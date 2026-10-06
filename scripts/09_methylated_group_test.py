from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INT = Path("data/interim")
CUTOFF = 0.5
MIN_GROUP = 10

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
annot = pd.read_csv(INT / "tss_annotation.csv").dropna(subset=["gene"])
corr = pd.read_csv(INT / "gene_meth_expr.csv").set_index("gene")

annot = annot[annot.gene.isin(expr.index)]
best = annot.sort_values("avg_coverage", ascending=False).drop_duplicates("gene")
lines = [c for c in meth.columns if c in expr.columns]
M = meth.loc[best.TSS_id, lines].to_numpy(dtype=float)
E = expr.loc[best.gene, lines].to_numpy(dtype=float)
genes = best.gene.to_numpy()
codes, _ = pd.factorize(cells.loc[lines, "OncotreeLineage"])
n_groups = int(codes.max()) + 1


def remove_group_means(x, g):
    counts = np.bincount(g, minlength=n_groups)
    sums = np.bincount(g, weights=x, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    return x - means[g], int((counts > 0).sum())


rows = []
for i in range(len(genes)):
    ok = ~np.isnan(M[i]) & ~np.isnan(E[i])
    h = (M[i][ok] > CUTOFF).astype(float)
    n_high = int(h.sum())
    n_low = int(len(h) - n_high)
    if n_high < MIN_GROUP or n_low < MIN_GROUP:
        continue
    g = codes[ok]
    re = stats.rankdata(E[i][ok]) / len(h)
    h_a, k = remove_group_means(h, g)
    re_a, _ = remove_group_means(re, g)
    ssx = float((h_a**2).sum())
    if ssx < 1e-6:
        continue
    slope = float((h_a * re_a).sum() / ssx)
    resid = re_a - slope * h_a
    df = len(h) - k - 1
    if df < 10:
        continue
    se = np.sqrt(float((resid**2).sum()) / df / ssx)
    p = float(2 * stats.t.sf(abs(slope / se), df))
    rows.append((genes[i], n_high, n_low, 100 * slope, p))

res = pd.DataFrame(rows, columns=["gene", "n_methylated", "n_unmethylated", "shift_pct", "p"])
res["fdr"] = multipletests(res.p, method="fdr_bh")[1]
res = res.join(corr[["rho_adj", "fdr_adj"]], on="gene")
res.to_csv(INT / "gene_meth_group.csv", index=False)

print("genes testable (>=10 methylated and >=10 unmethylated lines):", len(res), flush=True)
print("genes with both measurements but too few in one group:", len(genes) - len(res), flush=True)
hit = (res.shift_pct < -20) & (res.fdr < 0.05)
print("group test: shift below -20 points and FDR < 0.05:", int(hit.sum()), flush=True)
up = (res.shift_pct > 20) & (res.fdr < 0.05)
print("group test: shift above +20 points and FDR < 0.05:", int(up.sum()), flush=True)
corr_hit = (res.rho_adj < -0.3) & (res.fdr_adj < 0.05)
print("of the testable genes, correlation hits (rho < -0.3):", int(corr_hit.sum()), flush=True)
print("genes that are hits in both tests:", int((hit & corr_hit).sum()), flush=True)
print("group-test hits missed by correlation:", int((hit & ~corr_hit).sum()), flush=True)
print("correlation hits missed by group test:", int((~hit & corr_hit).sum()), flush=True)

known = ["MLH1", "MGMT", "CDKN2A", "RASSF1", "CDH1", "ESR1", "BRCA1"]
cols = ["n_methylated", "n_unmethylated", "shift_pct", "fdr", "rho_adj"]
sub = res.set_index("gene").reindex(known)[cols]
print("Known genes (group sizes, shift in percentile points, FDR, correlation rho):", flush=True)
print(sub.round(3).to_string(), flush=True)
print("saved:", INT / "gene_meth_group.csv", flush=True)
