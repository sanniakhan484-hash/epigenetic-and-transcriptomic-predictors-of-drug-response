from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")
FILE = RAW / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

cells = pd.read_csv(INT / "cell_lines.csv")
ids = set(cells.ModelID)

head = pd.read_csv(FILE, nrows=0)
info_cols = list(head.columns[:6])
gene_cols = list(head.columns[6:])

print("Reading expression file...", flush=True)
dtypes = {c: "float32" for c in gene_cols}
df = pd.read_csv(FILE, usecols=info_cols[1:] + gene_cols, dtype=dtypes)

df = df[df.IsDefaultEntryForModel == "Yes"]
df = df[df.ModelID.isin(ids)]
print("cell lines kept:", df.ModelID.nunique(), "| rows:", len(df), flush=True)

expr = df.set_index("ModelID")[gene_cols]
expr.columns = [c.split(" (")[0] for c in gene_cols]
dup = expr.columns.duplicated()
print("gene symbols duplicated after removing IDs:", int(dup.sum()), flush=True)
expr = expr.loc[:, ~dup]

expr_t = expr.T
expr_t.to_parquet(INT / "expression_matrix.parquet")

print("matrix shape (genes x cell lines):", expr_t.shape, flush=True)
print("fraction missing:", round(float(expr_t.isna().to_numpy().mean()), 4), flush=True)
print("value range:", float(expr_t.min().min()), "to", float(expr_t.max().max()), flush=True)
low = int((expr_t.median(axis=1) < 0.5).sum())
print("genes with median expression below 0.5 (barely expressed):", low, flush=True)

annot = pd.read_csv(INT / "tss_annotation.csv")
meth_genes = set(annot.gene.dropna())
print("genes in methylation annotation:", len(meth_genes), flush=True)
print("of those, also in expression matrix:", len(meth_genes & set(expr_t.index)), flush=True)


def group(label):
    s = str(label)
    if "TNBC" in s:
        return "TNBC"
    if s in ("HER2+", "ER-/PR-/HER2+"):
        return "HER2+ (ER-)"
    if "ER" in s:
        return "ER+"
    return "unknown"


breast = cells[cells.OncotreeLineage == "Breast"].copy()
breast["group"] = breast.ModelSubtypeFeatures.map(group)
breast = breast[breast.ModelID.isin(expr_t.columns)]
print("breast lines with expression:", len(breast), flush=True)

print("Median expression by breast subtype group (marker check):", flush=True)
for gene in ["ESR1", "ERBB2", "BRCA1"]:
    if gene in expr_t.index:
        values = expr_t.loc[gene, breast.ModelID]
        med = values.groupby(breast["group"].to_numpy()).median()
        print(gene, med.round(2).to_dict(), flush=True)

print("saved:", INT / "expression_matrix.parquet", flush=True)
