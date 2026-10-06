from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INT = Path("data/interim")
MIN_LINES = 100

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
annot = pd.read_csv(INT / "tss_annotation.csv")

annot = annot.dropna(subset=["gene"])
best = annot.sort_values("avg_coverage", ascending=False).drop_duplicates("gene")
best = best[best.gene.isin(expr.index)]
print("genes with methylation and expression:", len(best), flush=True)

lines = [c for c in meth.columns if c in expr.columns]
print("cell lines with both:", len(lines), flush=True)
M = meth.loc[best.TSS_id, lines].to_numpy(dtype=float)
E = expr.loc[best.gene, lines].to_numpy(dtype=float)
genes = best.gene.to_numpy()
tss = best.TSS_id.to_numpy()

lineage = cells.loc[lines, "OncotreeLineage"]
codes, _ = pd.factorize(lineage)
n_groups = int(codes.max()) + 1


def remove_group_means(x, g):
    counts = np.bincount(g, minlength=n_groups)
    sums = np.bincount(g, weights=x, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    return x - means[g], int((counts > 0).sum())


rows = []
for i in range(len(genes)):
    ok = ~np.isnan(M[i]) & ~np.isnan(E[i])
    n = int(ok.sum())
    if n < MIN_LINES:
        continue
    rm = stats.rankdata(M[i][ok])
    re = stats.rankdata(E[i][ok])
    if rm.std() == 0 or re.std() == 0:
        continue
    rho = float(np.corrcoef(rm, re)[0, 1])

    g = codes[ok]
    rm_a, k = remove_group_means(rm, g)
    re_a, _ = remove_group_means(re, g)
    if rm_a.std() < 1e-9 or re_a.std() < 1e-9:
        continue
    r = float(np.clip(np.corrcoef(rm_a, re_a)[0, 1], -0.999999, 0.999999))
    df = n - k - 1
    if df < 10:
        continue
    t = r * np.sqrt(df / (1 - r**2))
    p = float(2 * stats.t.sf(abs(t), df))
    rows.append((genes[i], tss[i], n, rho, r, p))

res = pd.DataFrame(rows, columns=["gene", "TSS_id", "n_lines", "rho_raw", "rho_adj", "p_adj"])
res["fdr_adj"] = multipletests(res.p_adj, method="fdr_bh")[1]
res.to_csv(INT / "gene_meth_expr.csv", index=False)

print("genes tested:", len(res), flush=True)
print("median rho (raw):", round(res.rho_raw.median(), 3), flush=True)
print("median rho (adjusted for lineage):", round(res.rho_adj.median(), 3), flush=True)
print("genes with raw rho < -0.3:", int((res.rho_raw < -0.3).sum()), flush=True)
print("genes with adjusted rho < -0.3:", int((res.rho_adj < -0.3).sum()), flush=True)
strong = (res.rho_adj < -0.3) & (res.fdr_adj < 0.05)
print("adjusted rho < -0.3 and FDR < 0.05:", int(strong.sum()), flush=True)
print("genes with adjusted rho > +0.3:", int((res.rho_adj > 0.3).sum()), flush=True)

known = ["MLH1", "MGMT", "CDKN2A", "RASSF1", "CDH1", "ESR1", "BRCA1"]
sub = res[res.gene.isin(known)].set_index("gene").reindex(known)
print("Known genes (n, raw rho, adjusted rho, FDR):", flush=True)
print(sub[["n_lines", "rho_raw", "rho_adj", "fdr_adj"]].round(3).to_string(), flush=True)
print("saved:", INT / "gene_meth_expr.csv", flush=True)
