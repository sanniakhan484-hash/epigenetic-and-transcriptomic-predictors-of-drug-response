from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
MARKERS = {
    "epithelial": ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "GRHL2"],
    "mesenchymal": ["VIM", "ZEB1", "ZEB2", "FN1", "CDH2", "SNAI2", "TWIST1"],
    "MAPK output": ["DUSP6", "ETV4", "ETV5", "SPRY2", "EGR1"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2"],
    "other": ["EGFR", "MYC", "ESR1", "ERBB2", "GATA3"],
}

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
pcs = pd.read_csv(INT / "methylation_pcs.csv", index_col=0)
expr = pd.read_parquet(INT / "expression_matrix.parquet")
meth = pd.read_parquet(INT / "methylation_matrix.parquet")
annot = pd.read_csv(INT / "tss_annotation.csv").set_index("TSS_id")


def subtype_group(label):
    s = str(label)
    if "TNBC" in s:
        return "TNBC"
    if s in ("HER2+", "ER-/PR-/HER2+"):
        return "HER2+ (ER-)"
    if "ER" in s:
        return "ER+"
    return "unknown"


def demean(a, onehot, counts):
    means = (a @ onehot) / np.maximum(counts, 1)
    return a - means @ onehot.T


def adjusted_corr(X, y, codes):
    """Lineage-adjusted Spearman correlation of each row of X with y."""
    onehot = np.eye(int(codes.max()) + 1)[codes]
    counts = onehot.sum(axis=0)
    xa = demean(stats.rankdata(X, axis=1), onehot, counts)
    ya = demean(stats.rankdata(y)[None, :], onehot, counts)[0]
    den = np.sqrt((xa**2).sum(axis=1) * (ya**2).sum())
    return (xa * ya).sum(axis=1) / den


pc2_all = pcs["PC2"]
lin = cells.loc[pc2_all.index, "OncotreeLineage"]
by_lin = pc2_all.groupby(lin).agg(["mean", "size"])
by_lin = by_lin[by_lin["size"] >= 10].sort_values("mean")
print("PC2 mean by lineage (lineages with at least 10 lines), lowest five:", flush=True)
print(by_lin.head(5).round(2).to_string(), flush=True)
print("highest five:", flush=True)
print(by_lin.tail(5).round(2).to_string(), flush=True)
breast = cells[cells.OncotreeLineage == "Breast"].copy()
breast["group"] = breast.ModelSubtypeFeatures.map(subtype_group)
b = pc2_all.reindex(breast.index).groupby(breast["group"]).agg(["mean", "size"])
print("Breast lines, PC2 by subtype group:", flush=True)
print(b.round(2).to_string(), flush=True)

lines_e = [c for c in pc2_all.index if c in expr.columns]
E = expr[lines_e]
E = E[~E.isna().any(axis=1)]
codes_e, _ = pd.factorize(cells.loc[lines_e, "OncotreeLineage"])
rho_e = pd.Series(
    adjusted_corr(E.to_numpy(dtype=float), pc2_all[lines_e].to_numpy(dtype=float), codes_e),
    index=E.index,
).dropna()
print(f"expression: {len(rho_e)} genes, {len(lines_e)} cell lines", flush=True)
print("Genes most POSITIVELY correlated with PC2 (lineage-adjusted rho):", flush=True)
print(rho_e.sort_values(ascending=False).head(15).round(3).to_string(), flush=True)
print("Genes most NEGATIVELY correlated with PC2:", flush=True)
print(rho_e.sort_values().head(15).round(3).to_string(), flush=True)
print("Marker genes (rho with PC2):", flush=True)
for group, genes in MARKERS.items():
    present = [g for g in genes if g in rho_e.index]
    txt = ", ".join(f"{g} {rho_e[g]:+.2f}" for g in present)
    print(f"  {group}: {txt}", flush=True)
top_e = pd.concat([rho_e.sort_values(ascending=False).head(50), rho_e.sort_values().head(50)])
top_e.rename("rho").to_csv(OUT / "pc2_top_expression_genes.csv")

keep = meth.isna().mean(axis=1) <= 0.10
M = meth.loc[keep, pc2_all.index]
M = M.apply(lambda r: r.fillna(r.mean()), axis=1)
codes_m, _ = pd.factorize(cells.loc[M.columns, "OncotreeLineage"])
rho_m = pd.Series(
    adjusted_corr(M.to_numpy(dtype=float), pc2_all[M.columns].to_numpy(dtype=float), codes_m),
    index=M.index,
).dropna()
names = annot.gene.reindex(rho_m.index)
print(f"methylation: {len(rho_m)} regions; with |rho| > 0.3: {int((rho_m.abs() > 0.3).sum())}",
      flush=True)
top_m = rho_m.reindex(rho_m.abs().sort_values(ascending=False).index).head(20)
print("Regions most aligned with PC2 (region, gene, rho):", flush=True)
for tss, r in top_m.items():
    print(f"  {tss}  {names[tss]}  {r:+.3f}", flush=True)
print("saved: results/tables/pc2_top_expression_genes.csv", flush=True)
