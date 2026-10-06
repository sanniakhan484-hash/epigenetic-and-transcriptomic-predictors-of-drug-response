from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INT = Path("data/interim")
MIN_LINES = 100

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
annot = pd.read_csv(INT / "tss_annotation.csv").dropna(subset=["gene"])
annot = annot[annot.gene.isin(expr.index)].reset_index(drop=True)
print("TSS regions whose gene has expression data:", len(annot), flush=True)

lines = [c for c in meth.columns if c in expr.columns]
M = meth.loc[annot.TSS_id, lines].to_numpy(dtype=float)
E = expr.loc[annot.gene, lines].to_numpy(dtype=float)
codes, _ = pd.factorize(cells.loc[lines, "OncotreeLineage"])
n_groups = int(codes.max()) + 1


def remove_group_means(x, g):
    counts = np.bincount(g, minlength=n_groups)
    sums = np.bincount(g, weights=x, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    return x - means[g], int((counts > 0).sum())


def adjusted_rho(m, e):
    ok = ~np.isnan(m) & ~np.isnan(e)
    if ok.sum() < MIN_LINES:
        return np.nan
    rm = stats.rankdata(m[ok])
    re = stats.rankdata(e[ok])
    rm_a, _ = remove_group_means(rm, codes[ok])
    re_a, _ = remove_group_means(re, codes[ok])
    if rm_a.std() < 1e-9 or re_a.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(rm_a, re_a)[0, 1])


stat_rows = []
for i in range(len(annot)):
    m = M[i]
    valid = m[~np.isnan(m)]
    stat_rows.append(
        (
            len(valid),
            float(valid.mean()) if len(valid) else np.nan,
            float(valid.std()) if len(valid) else np.nan,
            int((valid > 0.5).sum()),
            adjusted_rho(m, E[i]),
        )
    )
cols = ["n", "mean_meth", "sd_meth", "n_above_half", "rho_adj"]
reg = pd.concat([annot, pd.DataFrame(stat_rows, columns=cols)], axis=1)
reg.to_csv(INT / "region_stats.csv", index=False)

show = ["TSS_id", "chr", "fpos", "strand", "avg_coverage", "n", "mean_meth", "sd_meth"]
show += ["n_above_half", "rho_adj"]
for gene in ["MLH1", "CDKN2A"]:
    print(f"All regions for {gene}:", flush=True)
    print(reg[reg.gene == gene][show].round(3).to_string(index=False), flush=True)

by_cov = reg.sort_values("avg_coverage", ascending=False).drop_duplicates("gene").set_index("gene")
by_sd = reg.sort_values("sd_meth", ascending=False).drop_duplicates("gene").set_index("gene")
multi = reg.gene.value_counts()
multi = multi[multi > 1].index
print("genes with more than one region:", len(multi), flush=True)
differ = int((by_cov.loc[multi, "TSS_id"] != by_sd.loc[multi, "TSS_id"]).sum())
print("of those, rules choose different regions:", differ, flush=True)

for name, pick in [("highest coverage", by_cov), ("most variable methylation", by_sd)]:
    print(f"Rule: {name}", flush=True)
    print("  genes with adjusted rho < -0.3:", int((pick.rho_adj < -0.3).sum()), flush=True)

known = ["MLH1", "MGMT", "CDKN2A", "RASSF1", "CDH1", "ESR1", "BRCA1"]
both = pd.DataFrame(
    {"rho_coverage_rule": by_cov.rho_adj, "rho_variability_rule": by_sd.rho_adj}
).reindex(known)
print("Known genes under each rule (adjusted rho):", flush=True)
print(both.round(3).to_string(), flush=True)
print("saved:", INT / "region_stats.csv", flush=True)
