from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")

cells = pd.read_csv(INT / "cell_lines.csv")
ids = list(cells.ModelID)
breast_ids = list(cells.loc[cells.OncotreeLineage == "Breast", "ModelID"])

print("Reading drug matrix...", flush=True)
mat = pd.read_csv(
    RAW / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
    index_col=0,
    low_memory=False,
)
mat = mat[ids].apply(pd.to_numeric, errors="coerce").astype("float32")

print("Reading compound list...", flush=True)
comp = pd.read_csv(
    RAW / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv",
    low_memory=False,
)
comp = comp.drop_duplicates("IDs").set_index("IDs")
meta = comp.reindex(mat.index)[["Drug.Name", "screen", "dose", "MOA", "repurposing_target"]].copy()
meta["n_lines"] = mat.notna().sum(axis=1)
meta["n_breast"] = mat[breast_ids].notna().sum(axis=1)

mat.to_parquet(INT / "drug_matrix.parquet")
meta.to_csv(INT / "drug_meta.csv")

print("matrix shape (treatments x cell lines):", mat.shape, flush=True)
print("overall fraction missing:", round(float(mat.isna().to_numpy().mean()), 3), flush=True)
print("treatments with no drug name:", int(meta["Drug.Name"].isna().sum()), flush=True)
print(meta.screen.value_counts(dropna=False).to_string(), flush=True)

print("cell lines with data per treatment:", flush=True)
print(meta.n_lines.describe().to_string(), flush=True)
print("treatments with >=300 cell lines:", int((meta.n_lines >= 300).sum()), flush=True)
print("treatments with >=15 breast lines:", int((meta.n_breast >= 15).sum()), flush=True)

q = pd.Series(mat.to_numpy().ravel()).quantile([0.001, 0.01, 0.5, 0.99, 0.999])
print("value quantiles:", flush=True)
print(q.to_string(), flush=True)

dup = meta["Drug.Name"].value_counts()
print("drug names appearing more than once:", int((dup > 1).sum()), flush=True)

names = ["azacitidine", "decitabine", "guadecitabine", "vorinostat", "belinostat",
         "panobinostat", "romidepsin", "olaparib", "talazoparib", "niraparib", "rucaparib"]
hit = meta[meta["Drug.Name"].str.contains("|".join(names), case=False, na=False)]
print(hit[["Drug.Name", "screen", "dose", "n_lines", "n_breast"]].to_string(), flush=True)
print("saved: drug_matrix.parquet and drug_meta.csv in", INT, flush=True)
