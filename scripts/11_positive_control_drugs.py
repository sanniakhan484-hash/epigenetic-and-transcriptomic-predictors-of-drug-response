from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
CUTOFF = 0.5
MIN_GROUP = 10

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
drug = pd.read_parquet(INT / "drug_matrix.parquet")
dmeta = pd.read_csv(INT / "drug_meta.csv", index_col=0)
annot = pd.read_csv(INT / "tss_annotation.csv").dropna(subset=["gene"])
cut = pd.read_csv(INT / "gene_meth_group_cutoffs.csv", index_col=0)

ok_all = pd.Series(True, index=cut.index)
for c in ["0.3", "0.5", "0.7"]:
    ok_all &= (cut[f"shift@{c}"] < -20) & (cut[f"fdr@{c}"] < 0.05)
robust = cut[ok_all]
print("robust silenced genes:", len(robust), flush=True)
robust.round(3).to_csv(OUT / "robust_silenced_genes.csv")
for g in ["SLFN11", "MGMT", "MLH1", "CDKN2A", "CDH1", "ESR1"]:
    print(f"  {g} in robust set: {g in robust.index}", flush=True)

best = annot.sort_values("avg_coverage", ascending=False).drop_duplicates("gene")
region = best.set_index("gene").TSS_id
lines = [c for c in meth.columns if c in expr.columns and c in drug.columns]
codes, _ = pd.factorize(cells.loc[lines, "OncotreeLineage"])
n_groups = int(codes.max()) + 1
print("cell lines with methylation, expression and drug data:", len(lines), flush=True)

want = [
    "TEMOZOLOMIDE", "TOPOTECAN", "IRINOTECAN", "SN-38", "ETOPOSIDE", "CAMPTOTHECIN",
    "CISPLATIN", "CARBOPLATIN", "OXALIPLATIN", "MITOXANTRONE", "DOXORUBICIN",
    "GEMCITABINE", "TALAZOPARIB", "OLAPARIB",
]
names = dmeta["Drug.Name"].str.upper()
sel = dmeta[names.str.contains("|".join(want), na=False)].copy()
sel["Drug.Name"] = sel["Drug.Name"].str.upper()
print("matching treatments found:", len(sel), flush=True)
print(sel.sort_values("Drug.Name")[["Drug.Name", "screen", "n_lines"]].to_string(), flush=True)


def remove_group_means(x, g):
    counts = np.bincount(g, minlength=n_groups)
    sums = np.bincount(g, weights=x, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    return x - means[g], int((counts > 0).sum())


def shift_test(m, y):
    """Methylated (m > CUTOFF) vs not, on the percentile of y, within lineage."""
    ok = ~np.isnan(m) & ~np.isnan(y)
    h = (m[ok] > CUTOFF).astype(float)
    n_high = int(h.sum())
    n_low = int(len(h) - n_high)
    if n_high < MIN_GROUP or n_low < MIN_GROUP:
        return n_high, n_low, np.nan, np.nan
    yr = stats.rankdata(y[ok]) / len(h)
    h_a, k = remove_group_means(h, codes[ok])
    y_a, _ = remove_group_means(yr, codes[ok])
    ssx = float((h_a**2).sum())
    if ssx < 1e-6:
        return n_high, n_low, np.nan, np.nan
    slope = float((h_a * y_a).sum() / ssx)
    resid = y_a - slope * h_a
    df = len(h) - k - 1
    se = np.sqrt(float((resid**2).sum()) / df / ssx)
    p = float(2 * stats.t.sf(abs(slope / se), df))
    return n_high, n_low, 100 * slope, p


def adjusted_rho(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 100:
        return np.nan
    xa, _ = remove_group_means(stats.rankdata(x[ok]), codes[ok])
    ya, _ = remove_group_means(stats.rankdata(y[ok]), codes[ok])
    if xa.std() < 1e-9 or ya.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(xa, ya)[0, 1])


rows = []
for gene in ["SLFN11", "MGMT"]:
    if gene not in region.index or gene not in expr.index:
        print(f"{gene}: not available in methylation/expression tables", flush=True)
        continue
    m = meth.loc[region[gene], lines].to_numpy(dtype=float)
    e = expr.loc[gene, lines].to_numpy(dtype=float)
    for drug_id, info in sel.iterrows():
        y = drug.loc[drug_id, lines].to_numpy(dtype=float)
        n_hi, n_lo, shift, p = shift_test(m, y)
        rows.append(
            (gene, info["Drug.Name"], info["screen"], n_hi, n_lo, shift, p, adjusted_rho(e, y))
        )

cols = ["gene", "drug", "screen", "n_meth", "n_unmeth", "shift_pct", "p", "rho_expr"]
res = pd.DataFrame(rows, columns=cols)
res.to_csv(OUT / "positive_control_drugs.csv", index=False)
print("shift_pct > 0: methylated lines less sensitive. rho_expr < 0: higher expression, more killing.",
      flush=True)
for gene, sub in res.groupby("gene"):
    print(f"Gene {gene}:", flush=True)
    print(sub.drop(columns="gene").sort_values("shift_pct").round(3).to_string(index=False),
          flush=True)
print("saved:", OUT / "positive_control_drugs.csv", flush=True)
