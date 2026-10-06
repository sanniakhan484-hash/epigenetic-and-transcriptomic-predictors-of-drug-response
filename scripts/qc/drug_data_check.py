import pandas as pd

RAW = "data/raw/"

print("Reading compound list...", flush=True)
comp = pd.read_csv(RAW + "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv", low_memory=False)
print("compound list shape:", comp.shape, flush=True)
print("columns:", list(comp.columns), flush=True)
print(comp.iloc[:3, :6].to_string(), flush=True)
for col in ["screen", "dose"]:
    if col in comp.columns:
        print(comp[col].value_counts(dropna=False).head(8), flush=True)
if "Drug.Name" in comp.columns:
    print("unique drug names:", comp["Drug.Name"].nunique(), flush=True)

print("Reading data matrix...", flush=True)
mat = pd.read_csv(RAW + "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv", index_col=0, low_memory=False)
print("matrix shape (rows x columns):", mat.shape, flush=True)
print("first row labels:", list(mat.index[:3]), flush=True)
print("first column labels:", list(mat.columns[:3]), flush=True)
v = pd.to_numeric(pd.Series(mat.to_numpy().ravel()), errors="coerce")
print("fraction missing:", round(v.isna().mean(), 3), flush=True)
print(v.describe(), flush=True)

print("Joining with Model.csv and methylation...", flush=True)
model = pd.read_csv(RAW + "Model.csv", low_memory=False)
meth = pd.read_csv(RAW + "CCLE_RRBS_TSS_1kb_20180614.txt", sep="\t", nrows=0)
meth_lines = set(meth.columns[7:])
breast = model[model.OncotreeLineage == "Breast"]
with_drug = breast[breast.ModelID.isin(mat.columns)]
with_both = with_drug[with_drug.CCLEName.isin(meth_lines)]
print("breast lines in Model.csv:", len(breast), flush=True)
print("breast lines with drug data:", len(with_drug), flush=True)
print("breast lines with drug AND methylation:", len(with_both), flush=True)
print(with_both.ModelSubtypeFeatures.value_counts(dropna=False), flush=True)

names = ["azacitidine", "decitabine", "vorinostat", "belinostat", "panobinostat",
         "romidepsin", "olaparib", "talazoparib", "niraparib", "rucaparib"]
if "Drug.Name" in comp.columns:
    hit = comp[comp["Drug.Name"].str.contains("|".join(names), case=False, na=False)]
    cols = [c for c in ["IDs", "Drug.Name", "screen", "dose", "MOA"] if c in hit.columns]
    print(hit[cols].to_string(), flush=True)
