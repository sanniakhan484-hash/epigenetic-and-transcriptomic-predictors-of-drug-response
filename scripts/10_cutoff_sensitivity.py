from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INT = Path("data/interim")
CUTOFFS = [0.3, 0.5, 0.7]
MIN_GROUP = 10

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
expr = pd.read_parquet(INT / "expression_matrix.parquet")
annot = pd.read_csv(INT / "tss_annotation.csv").dropna(subset=["gene"])

annot = annot[annot.gene.isin(expr.index)]
best = annot.sort_values("avg_coverage", ascending=False).drop_duplicates("gene")
lines = [c for c in meth.columns if c in expr.columns]
M = meth.loc[best.TSS_id, lines].to_numpy(dtype=float)
E = expr.loc[best.gene, lines].to_numpy(dtype=float)
genes = best.gene.to_numpy()
lineage = cells.loc[lines, "OncotreeLineage"].to_numpy()
codes, _ = pd.factorize(lineage)
n_groups = int(codes.max()) + 1


def remove_group_means(x, g):
    counts = np.bincount(g, minlength=n_groups)
    sums = np.bincount(g, weights=x, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    return x - means[g], int((counts > 0).sum())


def group_test(cutoff):
    rows = []
    for i in range(len(genes)):
        ok = ~np.isnan(M[i]) & ~np.isnan(E[i])
        h = (M[i][ok] > cutoff).astype(float)
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
        rows.append((genes[i], n_high, 100 * slope, p))
    res = pd.DataFrame(rows, columns=["gene", "n_methylated", "shift_pct", "p"])
    res["fdr"] = multipletests(res.p, method="fdr_bh")[1]
    return res.set_index("gene")


results = {}
hits = {}
for c in CUTOFFS:
    r = group_test(c)
    results[c] = r
    hits[c] = set(r.index[(r.shift_pct < -20) & (r.fdr < 0.05)])
    print(f"cutoff {c}: testable genes {len(r)}, hits (shift < -20, FDR < 0.05) {len(hits[c])}",
          flush=True)

for a, b in [(0.3, 0.5), (0.5, 0.7), (0.3, 0.7)]:
    both = len(hits[a] & hits[b])
    print(f"hits shared between cutoff {a} and {b}: {both} (union {len(hits[a] | hits[b])})",
          flush=True)
robust = hits[0.3] & hits[0.5] & hits[0.7]
print("hits at all three cutoffs:", len(robust), flush=True)

known = ["MLH1", "MGMT", "CDKN2A", "RASSF1", "CDH1", "ESR1", "BRCA1"]
shift = pd.DataFrame({f"shift@{c}": results[c].shift_pct for c in CUTOFFS}).reindex(known)
size = pd.DataFrame({f"n_meth@{c}": results[c].n_methylated for c in CUTOFFS}).reindex(known)
print("Known genes: shift in percentile points at each cutoff", flush=True)
print(shift.round(1).to_string(), flush=True)
print("Known genes: number of methylated lines at each cutoff", flush=True)
print(size.to_string(), flush=True)

print("Lineages of the methylated lines (cutoff 0.5), top 5:", flush=True)
for gene in ["MLH1", "CDKN2A", "MGMT"]:
    idx = np.where(genes == gene)[0]
    if len(idx) == 0:
        continue
    mask = M[idx[0]] > 0.5
    top = pd.Series(lineage[mask]).value_counts().head(5).to_dict()
    print(f"  {gene}: {top}", flush=True)

out = pd.DataFrame({f"shift@{c}": results[c].shift_pct for c in CUTOFFS})
for c in CUTOFFS:
    out[f"fdr@{c}"] = results[c].fdr
out.to_csv(INT / "gene_meth_group_cutoffs.csv")
print("saved:", INT / "gene_meth_group_cutoffs.csv", flush=True)
