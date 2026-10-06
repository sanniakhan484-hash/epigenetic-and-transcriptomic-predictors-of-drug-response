from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")

cells = pd.read_csv(INT / "cell_lines.csv")
name_to_id = dict(zip(cells.CCLEName, cells.ModelID))
info_cols = ["TSS_id", "gene", "chr", "fpos", "tpos", "strand", "avg_coverage"]

print("Reading methylation (selected cell lines only)...", flush=True)
usecols = info_cols + list(name_to_id)
meth = pd.read_csv(
    RAW / "CCLE_RRBS_TSS_1kb_20180614.txt",
    sep="\t",
    na_values=["NA"],
    usecols=usecols,
    low_memory=False,
)

annot = meth[info_cols].copy()
annot.to_csv(INT / "tss_annotation.csv", index=False)

mat = meth.set_index("TSS_id")[list(name_to_id)].rename(columns=name_to_id)
before = int(mat.notna().to_numpy().sum())
mat = mat.apply(pd.to_numeric, errors="coerce").astype("float32")
after = int(mat.notna().to_numpy().sum())
print("text entries converted to missing:", before - after, flush=True)
mat.to_parquet(INT / "methylation_matrix.parquet")

print("matrix shape (TSS regions x cell lines):", mat.shape, flush=True)
print("duplicate TSS_ids:", int(mat.index.duplicated().sum()), flush=True)
print("overall fraction missing:", round(float(mat.isna().to_numpy().mean()), 3), flush=True)

row_na = mat.isna().mean(axis=1)
print("TSS regions with >50% missing:", int((row_na > 0.5).sum()), flush=True)

vals = mat.to_numpy().ravel()
vals = vals[~np.isnan(vals)]
print("value range:", float(vals.min()), "to", float(vals.max()), flush=True)
print("share of values below 0 or above 1:", round(float(((vals < 0) | (vals > 1)).mean()), 4), flush=True)

print(annot.avg_coverage.describe().to_string(), flush=True)
print("genes with more than one TSS region:", int((annot.gene.value_counts() > 1).sum()), flush=True)
print("saved: tss_annotation.csv and methylation_matrix.parquet in", INT, flush=True)
