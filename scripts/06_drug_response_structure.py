from pathlib import Path

import pandas as pd

INT = Path("data/interim")

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
mat = pd.read_parquet(INT / "drug_matrix.parquet")
meta = pd.read_csv(INT / "drug_meta.csv", index_col=0)

lin = cells.OncotreeLineage.reindex(mat.columns)
counts = lin.value_counts()
keep_lin = counts[counts >= 10].index
cols = lin[lin.isin(keep_lin)].index
g = lin[cols]
print("lineages with >=10 lines:", len(keep_lin), "| cell lines used:", len(cols), flush=True)

X = mat[cols].T
Xr = X.rank()

grand = Xr.mean()
gm = Xr.groupby(g).mean()
gn = Xr.groupby(g).count()
between = (gn * (gm - grand) ** 2).sum()
total = ((Xr - grand) ** 2).sum()
eta = (between / total).rename("eta2_lineage")

print("Share of drug response explained by lineage, across all treatments:", flush=True)
print(eta.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(3).to_string(), flush=True)
print("treatments with lineage explaining >30%:", int((eta > 0.30).sum()), "of", len(eta), flush=True)

names = [
    "AZACITIDINE",
    "DECITABINE",
    "GUADECITABINE",
    "VORINOSTAT",
    "BELINOSTAT",
    "PANOBINOSTAT",
    "ROMIDEPSIN",
    "OLAPARIB",
    "TALAZOPARIB",
    "NIRAPARIB",
    "RUCAPARIB",
]
ctrl = meta[meta["Drug.Name"].str.upper().isin(names)]

print("Positive-control drugs (eta2, breast median, sensitive lineages):", flush=True)
for drug_id, row in ctrl.iterrows():
    med = X[drug_id].groupby(g).median().dropna().sort_values()
    breast_med = med.get("Breast", float("nan"))
    most = ", ".join(f"{k} ({v:.2f})" for k, v in med.head(3).items())
    least = ", ".join(f"{k} ({v:.2f})" for k, v in med.tail(2).items())
    head = f"{row['Drug.Name']:<14} eta2={eta[drug_id]:.2f}"
    print(f"{head}  breast median={breast_med:.2f}", flush=True)
    print(f"    most sensitive: {most}", flush=True)
    print(f"    least sensitive: {least}", flush=True)

eta.to_csv(INT / "drug_lineage_eta2.csv")
print("saved:", INT / "drug_lineage_eta2.csv", flush=True)
