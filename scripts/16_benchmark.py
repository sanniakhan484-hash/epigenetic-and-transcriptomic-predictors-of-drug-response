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
N_TREAT = 300
LAMBDAS = np.logspace(-1, 3, 9)
MEK = ["MEK162", "GDC-0623", "TRAMETINIB", "AS-703026", "RO-4987655", "PD-0325901"]

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
drug = pd.read_parquet(INT / "drug_matrix.parquet")
dmeta = pd.read_csv(INT / "drug_meta.csv", index_col=0)

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
print("cell lines with methylation, expression, mutation and drug data:", n_lines, flush=True)


def normalized_kernel(x):
    k = x @ x.T
    return k / np.mean(np.diag(k))


def zscore_columns(x):
    sd = x.std(axis=0)
    keep = sd > 1e-9
    return (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]


lineage = cells.loc[lines, "OncotreeLineage"].to_numpy()
codes, _ = pd.factorize(lineage)
K_lin = normalized_kernel(np.eye(int(codes.max()) + 1)[codes])

mut_x = (mut.loc[lines].fillna(0).to_numpy(dtype=float) > 0).astype(float)
mut_x = mut_x[:, mut_x.sum(axis=0) >= 10]
print("mutation features (genes mutated in at least 10 lines):", mut_x.shape[1], flush=True)
K_mut = normalized_kernel(zscore_columns(mut_x))

ex = expr[lines]
ex = ex[~ex.isna().any(axis=1)].to_numpy(dtype=float).T
print("expression features:", zscore_columns(ex).shape[1], flush=True)
K_expr = normalized_kernel(zscore_columns(ex))

me = meth.loc[meth.isna().mean(axis=1) <= 0.10, lines]
me = me.apply(lambda r: r.fillna(r.mean()), axis=1).to_numpy(dtype=float).T
print("methylation features:", zscore_columns(me).shape[1], flush=True)
K_meth = normalized_kernel(zscore_columns(me))

rng = np.random.default_rng(SEED)
perm = np.arange(n_lines)
for c in range(int(codes.max()) + 1):
    members = np.where(codes == c)[0]
    perm[members] = rng.permutation(members)
K_meth_shuf = K_meth[perm][:, perm]

models = {
    "A_lineage": K_lin,
    "B_lineage_mut": K_lin + K_mut,
    "C_lineage_mut_expr": K_lin + K_mut + K_expr,
    "D_lineage_mut_meth": K_lin + K_mut + K_meth,
    "E_all": K_lin + K_mut + K_expr + K_meth,
    "F_all_meth_shuffled": K_lin + K_mut + K_expr + K_meth_shuf,
}


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
print("treatments benchmarked:", len(chosen), "(top by spread, plus MEK inhibitors)", flush=True)

rows = []
for count, d in enumerate(chosen, start=1):
    ok = np.where(~np.isnan(Y[d]))[0]
    y = stats.norm.ppf((stats.rankdata(Y[d, ok]) - 0.5) / len(ok))
    order_ok = np.random.default_rng(SEED + int(d)).permutation(len(ok))
    folds = np.array_split(order_ok, N_FOLDS)
    result = {"treatment": dmeta.index[d], "drug": dmeta.iloc[d]["Drug.Name"], "n": len(ok)}
    for name, kern in models.items():
        pred = np.zeros(len(ok))
        for test in folds:
            train = np.setdiff1d(np.arange(len(ok)), test)
            k_tr = kern[np.ix_(ok[train], ok[train])]
            k_te = kern[np.ix_(ok[test], ok[train])]
            pred[test] = ridge_predict(k_tr, k_te, y[train])
        result[name] = float(stats.spearmanr(pred, y)[0])
    rows.append(result)
    if count % 50 == 0:
        print(f"  done {count} of {len(chosen)}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(INT / "benchmark_random_cv.csv", index=False)

names = list(models)
print("Median out-of-fold Spearman correlation across treatments:", flush=True)
print(res[names].median().round(3).to_string(), flush=True)

brng = np.random.default_rng(SEED)
comparisons = [
    ("D vs B: methylation added to lineage+mutation", "D_lineage_mut_meth", "B_lineage_mut"),
    ("C vs B: expression added to lineage+mutation", "C_lineage_mut_expr", "B_lineage_mut"),
    ("E vs C: methylation added to expression", "E_all", "C_lineage_mut_expr"),
    ("E vs F: real vs shuffled methylation", "E_all", "F_all_meth_shuffled"),
    ("F vs C: shuffled methylation vs none", "F_all_meth_shuffled", "C_lineage_mut_expr"),
    ("B vs A: mutations added to lineage", "B_lineage_mut", "A_lineage"),
]
summary = []
print("Paired comparisons (difference in Spearman, per treatment):", flush=True)
for label, m1, m2 in comparisons:
    diff = (res[m1] - res[m2]).to_numpy()
    boot = [np.median(brng.choice(diff, len(diff))) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    summary.append((label, float(np.median(diff)), float(lo), float(hi), float((diff > 0).mean())))
    print(f"  {label}: median {np.median(diff):+.4f} (95% CI {lo:+.4f} to {hi:+.4f}), "
          f"improved in {100 * (diff > 0).mean():.0f}% of treatments", flush=True)
cols = ["comparison", "median_diff", "ci_low", "ci_high", "share_improved"]
pd.DataFrame(summary, columns=cols).to_csv(OUT / "benchmark_summary.csv", index=False)

mek_rows = res[res["drug"].str.upper().isin(MEK)]
print("MEK inhibitors (Spearman per model):", flush=True)
print(mek_rows[["drug", "n"] + names].round(3).to_string(index=False), flush=True)
print("saved: results/tables/benchmark_summary.csv", flush=True)
