from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

print("Reading Model.csv...", flush=True)
model = pd.read_csv(RAW / "Model.csv", low_memory=False)

print("Reading methylation header (cell line names)...", flush=True)
meth_header = pd.read_csv(RAW / "CCLE_RRBS_TSS_1kb_20180614.txt", sep="\t", nrows=0)
meth_lines = set(meth_header.columns[7:])

print("Reading drug matrix header (DepMap IDs)...", flush=True)
drug_header = pd.read_csv(
    RAW / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
    nrows=0,
    index_col=0,
)
drug_ids = set(drug_header.columns)

both = model[model.CCLEName.isin(meth_lines) & model.ModelID.isin(drug_ids)]
cols = ["ModelID", "CCLEName", "OncotreeLineage", "OncotreePrimaryDisease", "ModelSubtypeFeatures"]
table = both[cols].copy()
table.to_csv(OUT / "cell_lines.csv", index=False)

print("cell lines in all three tables:", len(table), flush=True)
print(table.OncotreeLineage.value_counts().head(12), flush=True)

breast = table[table.OncotreeLineage == "Breast"]
print("breast lines:", len(breast), flush=True)
print(breast.ModelSubtypeFeatures.value_counts(dropna=False).to_string(), flush=True)
print("saved to", OUT / "cell_lines.csv", flush=True)
