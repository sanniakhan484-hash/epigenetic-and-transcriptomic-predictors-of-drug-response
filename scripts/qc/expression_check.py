from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")
FILE = RAW / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

print("Reading expression header...", flush=True)
head = pd.read_csv(FILE, nrows=0)
n_info = 6
info_cols = list(head.columns[:n_info])
gene_cols = list(head.columns[n_info:])
print("info columns:", info_cols, flush=True)
print("number of gene columns:", len(gene_cols), flush=True)
print("example gene column names:", gene_cols[:3], flush=True)

print("Reading profile columns only...", flush=True)
meta = pd.read_csv(FILE, usecols=info_cols)
print("profiles (rows):", len(meta), flush=True)
print("unique cell lines (ModelID):", meta.ModelID.nunique(), flush=True)

flag = "IsDefaultEntryForModel"
print(flag, "values:", meta[flag].value_counts(dropna=False).to_dict(), flush=True)
is_default = meta[flag].astype(str).str.lower().isin(["yes", "true", "1"])
dflt = meta[is_default]
per_line = dflt.groupby("ModelID").size()
print("cell lines with a default profile:", int(per_line.shape[0]), flush=True)
print("max default profiles per cell line:", int(per_line.max()), flush=True)

multi = meta.groupby("ModelID").size()
print("cell lines with more than one profile:", int((multi > 1).sum()), flush=True)

cells = pd.read_csv(INT / "cell_lines.csv")
have = set(dflt.ModelID)
print("of our 623 cell lines, with a default profile:", int(cells.ModelID.isin(have).sum()), flush=True)
breast = cells[cells.OncotreeLineage == "Breast"]
print("of our 27 breast lines, with a default profile:", int(breast.ModelID.isin(have).sum()), flush=True)

brca1 = [c for c in gene_cols if c.startswith("BRCA1 ")]
print("BRCA1 column:", brca1, flush=True)
