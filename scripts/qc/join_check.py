import pandas as pd

print("Reading Model.csv...", flush=True)
m = pd.read_csv("data/raw/Model.csv", low_memory=False)

print("Reading methylation header...", flush=True)
d = pd.read_csv("data/raw/CCLE_RRBS_TSS_1kb_20180614.txt", sep="\t", nrows=0)
cl = list(d.columns[7:])

mm = m[m.CCLEName.isin(cl)]
print("methylation lines:", len(cl), flush=True)
print("matched in Model.csv:", len(mm), flush=True)

b = mm[mm.OncotreeLineage == "Breast"]
print("breast matched:", len(b), flush=True)
print(b.OncotreeSubtype.value_counts(dropna=False), flush=True)
print(b.ModelSubtypeFeatures.value_counts(dropna=False).head(10), flush=True)
