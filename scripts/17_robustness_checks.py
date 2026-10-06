from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path("data/raw")
INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
N_TREAT = 150
MIN_LOLO_LINES = 15
LAMBDAS = np.logspace(-1, 3, 9)
MEK = ["MEK162", "GDC-0623", "TRAMETINIB", "AS-703026", "RO-4987655", "PD-0325901"]

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
drug = pd.read_parquet(INT / "drug_matrix.parquet")
dmeta = pd.read_csv(INT / "drug_meta.csv", index_col=0)
annot = pd.read_csv(INT / "tss_annotation.csv").dropna(subset=["gene"])
robust = pd.read_csv(OUT / "robust_silenced_genes.csv", index_col=0)

mut = pd.read_csv(RAW / "OmicsSomaticMutationsMatrixHotspot.csv", low_memory=False)
flag = "IsDefaultEntryForModel"
if flag in mut.columns:
    mut = mut[mut[flag].astype(str).str.lower().isin(["yes", "true", "1"])]
id_col = next(c for c in mut.columns if mut[c].astype(str).str.startswith("ACH-").all())
mut = mut.set_index(id_col)
mut = mut[~mut.index.duplicated()]
mut = mut[[c for c in mut.columns if str(c).endswith(")") and " (" in str(c)]]

lines = [c for c in cells.index if c in meth.columns and c in expr.columns
         and c in drug.columns and c in mut.index]
n_lines = len(lines)
print("cell lines:", n_lines, flush=True)


def normalized_kernel(x):
    k = x @ x.T
    return k / np.mean(np.diag(k))


def zscore_columns(x):
    sd = x.std(axis=0)
    keep = sd > 1e-9
    return (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]


def methylation_block(region_ids):
    block = meth.loc[region_ids, lines]
    block = block[block.isna().mean(axis=1) <= 0.10]
    block = block.apply(lambda r: r.fillna(r.mean()), axis=1)
    return block.to_numpy(dtype=float).T


lineage = cells.loc[lines, "OncotreeLineage"].to_numpy()
codes, _ = pd.factorize(lineage)
n_groups = int(codes.max()) + 1
K_lin = normalized_kernel(np.eye(n_groups)[codes])

mut_x = (mut.loc[lines].fillna(0).to_numpy(dtype=float) > 0).astype(float)
mut_x = mut_x[:, mut_x.sum(axis=0) >= 10]
K_mut = normalized_kernel(zscore_columns(mut_x))

ex = expr[lines]
ex = ex[~ex.isna().any(axis=1)].to_numpy(dtype=float).T
K_expr = normalized_kernel(zscore_columns(ex))

all_x = zscore_columns(methylation_block(list(meth.index)))
K_meth = normalized_kernel(all_x)

best = annot.sort_values("avg_coverage", ascending=False).drop_duplicates("gene").set_index("gene")
sil_regions = [best.loc[g, "TSS_id"] for g in robust.index
               if g in best.index and best.loc[g, "TSS_id"] in meth.index]
sil_x = zscore_columns(methylation_block(sil_regions))
print("methylation regions, all:", all_x.shape[1], "| silenced-gene regions:", sil_x.shape[1],
      flush=True)
K_sil = normalized_kernel(sil_x)

rng = np.random.default_rng(SEED)
perm = np.arange(n_lines)
for c in range(n_groups):
    members = np.where(codes == c)[0]
    perm[members] = rng.permutation(members)
K_sil_shuf = K_sil[perm][:, perm]

base = K_lin + K_mut
models = {
    "B_lineage_mut": base,
    "C_plus_expr": base + K_expr,
    "D_plus_meth_all": base + K_meth,
    "E_expr_meth_all": base + K_expr + K_meth,
    "Dsil_plus_meth_silenced": base + K_sil,
    "Esil_expr_meth_silenced": base + K_expr + K_sil,
    "Fsil_expr_meth_silenced_shuffled": base + K_expr + K_sil_shuf,
}
names = list(models)


def ridge_predict(k_train, k_test, y_train):
    mu = y_train.mean()
    yc = y_train - mu
    s, u = np.linalg.eigh(k_train)
    s = np.clip(s, 0, None)
    uty = u.T @ yc
    best_err, best_alpha = None, None
    for lam in LAMBDAS:
        w = 1.0 / (s + lam)
        alpha = u @ (w * uty)
        hdiag = (u**2) @ w
        err = float(np.mean((alpha / hdiag) ** 2))
        if best_err is None or err < best_err:
            best_err, best_alpha = err, alpha
    return k_test @ best_alpha + mu


size_counts = np.bincount(codes, minlength=n_groups)
big = [c for c in range(n_groups) if size_counts[c] >= MIN_LOLO_LINES]
print("tissues held out one at a time:", len(big), flush=True)

Y = drug[lines].to_numpy(dtype=float)
n_ok = (~np.isnan(Y)).sum(axis=1)
iqr = np.nanpercentile(Y, 75, axis=1) - np.nanpercentile(Y, 25, axis=1)
eligible = n_ok >= int(0.65 * n_lines)
order = np.argsort(-np.where(eligible, iqr, -np.inf))
chosen = list(order[:N_TREAT])
names_up = dmeta["Drug.Name"].str.upper().to_numpy()
for d in np.where(np.isin(names_up, MEK) & eligible)[0]:
    if d not in chosen:
        chosen.append(d)
print("treatments:", len(chosen), flush=True)

rcv_rows, lolo_rows = [], []
for count, d in enumerate(chosen, start=1):
    ok = np.where(~np.isnan(Y[d]))[0]
    y = stats.norm.ppf((stats.rankdata(Y[d, ok]) - 0.5) / len(ok))
    info = {"treatment": dmeta.index[d], "drug": dmeta.iloc[d]["Drug.Name"], "n": len(ok)}

    order_ok = np.random.default_rng(SEED + int(d)).permutation(len(ok))
    folds = np.array_split(order_ok, N_FOLDS)
    row = dict(info)
    for name, kern in models.items():
        pred = np.zeros(len(ok))
        for test in folds:
            train = np.setdiff1d(np.arange(len(ok)), test)
            pred[test] = ridge_predict(
                kern[np.ix_(ok[train], ok[train])], kern[np.ix_(ok[test], ok[train])], y[train]
            )
        row[name] = float(stats.spearmanr(pred, y)[0])
    rcv_rows.append(row)

    row_pool, row_within = dict(info), dict(info)
    for name, kern in models.items():
        preds, truth, within = [], [], []
        for h in big:
            test = np.where(codes[ok] == h)[0]
            if len(test) < 10:
                continue
            train = np.setdiff1d(np.arange(len(ok)), test)
            p = ridge_predict(
                kern[np.ix_(ok[train], ok[train])], kern[np.ix_(ok[test], ok[train])], y[train]
            )
            preds.extend(p)
            truth.extend(y[test])
            within.append(stats.spearmanr(p, y[test])[0])
        row_pool[name] = float(stats.spearmanr(preds, truth)[0]) if len(preds) > 10 else np.nan
        row_within[name] = float(np.nanmean(within)) if within else np.nan
    lolo_rows.append((row_pool, row_within))
    if count % 25 == 0:
        print(f"  done {count} of {len(chosen)}", flush=True)

rcv = pd.DataFrame(rcv_rows)
lolo_pool = pd.DataFrame([r[0] for r in lolo_rows])
lolo_within = pd.DataFrame([r[1] for r in lolo_rows])
rcv.to_csv(INT / "benchmark_checks_random_cv.csv", index=False)
lolo_pool.to_csv(INT / "benchmark_checks_lolo_pooled.csv", index=False)
lolo_within.to_csv(INT / "benchmark_checks_lolo_within.csv", index=False)

brng = np.random.default_rng(SEED)


def paired(table, m1, m2):
    diff = (table[m1] - table[m2]).dropna().to_numpy()
    boot = [np.median(brng.choice(diff, len(diff))) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.median(diff)), float(lo), float(hi), float((diff > 0).mean())


for title, table in [("Random 5-fold CV", rcv),
                     ("Leave-one-tissue-out, within-tissue average", lolo_within)]:
    print(f"{title}: median Spearman across treatments", flush=True)
    print(table[names].median().round(3).to_string(), flush=True)

M_E, M_C = "Esil_expr_meth_silenced", "C_plus_expr"
M_F, M_B = "Fsil_expr_meth_silenced_shuffled", "B_lineage_mut"
pairs = [
    ("Esil vs C: silenced-gene methylation added to expression", M_E, M_C),
    ("Esil vs Fsil: real vs shuffled silenced-gene methylation", M_E, M_F),
    ("Dsil vs B: silenced-gene methylation added to lineage+mutation", "Dsil_plus_meth_silenced",
     M_B),
    ("E vs C: all-region methylation added to expression", "E_expr_meth_all", M_C),
    ("C vs B: expression added to lineage+mutation", M_C, M_B),
]
summary = []
res_stats = {}
for title, table in [("random_cv", rcv), ("lolo_within", lolo_within)]:
    print(f"Paired differences, {title} (median, 95% CI, share improved):", flush=True)
    for label, m1, m2 in pairs:
        med, lo, hi, share = paired(table, m1, m2)
        res_stats[(title, m1, m2)] = (med, lo, hi)
        summary.append((title, label, med, lo, hi, share))
        print(f"  {label}: {med:+.4f} ({lo:+.4f} to {hi:+.4f}), {100 * share:.0f}%", flush=True)
cols = ["evaluation", "comparison", "median_diff", "ci_low", "ci_high", "share_improved"]
pd.DataFrame(summary, columns=cols).to_csv(OUT / "benchmark_checks_summary.csv", index=False)

gap = (rcv[M_E] - rcv[M_F]).dropna()
sigma = 1.4826 * float(np.median(np.abs(gap[gap < 0]))) if (gap < 0).any() else float("nan")
n_pos = int((gap > 3 * sigma).sum())
n_neg = int((gap < -3 * sigma).sum())
print(f"noise scale of the real-minus-shuffled gap (from its negative side): {sigma:.4f}",
      flush=True)
print(f"drugs where real beats shuffled by more than 3x noise: {n_pos}; "
      f"shuffled beats real by more than 3x noise: {n_neg}", flush=True)

r1 = res_stats[("random_cv", M_E, M_F)]
r2 = res_stats[("random_cv", M_E, M_C)]
r4 = res_stats[("lolo_within", M_E, M_F)]
ok1 = r1[0] > 0 and r1[1] > 0
ok2 = r2[0] > 0.01 and r2[1] > 0
ok3 = n_pos >= 3 and n_pos >= 2 * max(n_neg, 1)
ok4 = r4[0] > 0 and r4[1] > 0
print("DECISION RULE", flush=True)
print(f"  R1 real beats shuffled overall (random CV): {ok1}", flush=True)
print(f"  R2 adds more than +0.01 to expression overall (random CV): {ok2}", flush=True)
print(f"  R3 excess of drugs where real beats shuffled by > 3x noise ({n_pos} vs {n_neg}): {ok3}",
      flush=True)
print(f"  R4 real beats shuffled when whole tissues are held out (within-tissue): {ok4}",
      flush=True)
if ok1 and ok2 and ok4:
    verdict = "BROAD ADDED VALUE"
elif ok3 and ok4:
    verdict = "ADDED VALUE FOR A SUBSET OF DRUGS"
elif ok3:
    verdict = "SUBSET EFFECT THAT DOES NOT TRANSFER ACROSS TISSUES (EXPLORATORY)"
else:
    verdict = "NO ROBUST ADDED VALUE"
print("  VERDICT:", verdict, flush=True)

top = rcv.assign(gain=rcv[M_E] - rcv[M_F]).sort_values("gain", ascending=False).head(10)
print("Ten drugs with the largest gain, real vs shuffled (random CV):", flush=True)
print(top[["drug", "n", M_C, M_E, M_F, "gain"]].round(3).to_string(index=False), flush=True)

mek_mask = rcv["drug"].str.upper().isin(MEK)
print("MEK inhibitors, random CV (Spearman):", flush=True)
print(rcv[mek_mask][["drug", M_B, M_C, "E_expr_meth_all", M_E, M_F]].round(3).to_string(index=False),
      flush=True)
print("saved: results/tables/benchmark_checks_summary.csv", flush=True)
