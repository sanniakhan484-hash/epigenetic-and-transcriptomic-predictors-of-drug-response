from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path("data/raw")
INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
MEK = ["PD-0325901", "REFAMETINIB", "SELUMETINIB", "TRAMETINIB"]
MAPK_GENES = ["BRAF", "NRAS", "KRAS", "HRAS"]
MIN_LINES = 100

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
pcs = pd.read_csv(INT / "methylation_pcs.csv", index_col=0)

mut = pd.read_csv(RAW / "OmicsSomaticMutationsMatrixHotspot.csv", low_memory=False)
flag = "IsDefaultEntryForModel"
if flag in mut.columns:
    mut = mut[mut[flag].astype(str).str.lower().isin(["yes", "true", "1"])]
id_col = next(c for c in mut.columns if mut[c].astype(str).str.startswith("ACH-").all())
mut = mut.set_index(id_col)
mut = mut[~mut.index.duplicated()]
mapk_cols = [c for c in mut.columns if str(c).split(" (")[0] in MAPK_GENES]

lines = [c for c in cells.index if c in pcs.index and c in mut.index]
mapk = (mut.loc[lines, mapk_cols].fillna(0) > 0).any(axis=1).to_numpy().astype(float)
pc2 = pcs.loc[lines, "PC2"].to_numpy(dtype=float)
codes, _ = pd.factorize(cells.loc[lines, "OncotreeLineage"])
K = int(codes.max()) + 1
print("cell lines with methylation PCs and mutation data:", len(lines), flush=True)

g = pd.read_csv(
    RAW / "sanger-dose-response.csv",
    usecols=["DATASET", "ARXSPAN_ID", "DRUG_NAME", "AUC_PUBLISHED"],
)
print("rows:", len(g), "| rows without a DepMap ID:", int(g.ARXSPAN_ID.isna().sum()), flush=True)
g = g.dropna(subset=["ARXSPAN_ID", "AUC_PUBLISHED"])
keys = ["DATASET", "DRUG_NAME", "ARXSPAN_ID"]
print("repeated cell line x drug x dataset rows (median taken):",
      int(g.duplicated(keys).sum()), flush=True)
g = g.groupby(keys, as_index=False)["AUC_PUBLISHED"].median()
print("combinations of dataset and drug:", g.groupby(["DATASET", "DRUG_NAME"]).ngroups, flush=True)


def residual(v, design):
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta


def partial_rho(x, y, design):
    rx = residual(stats.rankdata(x), design)
    ry = residual(stats.rankdata(y), design)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan, np.nan
    r = float(np.clip(np.corrcoef(rx, ry)[0, 1], -0.999999, 0.999999))
    df = len(x) - int(np.linalg.matrix_rank(design)) - 1
    t = r * np.sqrt(df / (1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), df))


rows = []
for (dataset, drug), sub in g.groupby(["DATASET", "DRUG_NAME"]):
    y = sub.set_index("ARXSPAN_ID").AUC_PUBLISHED.reindex(lines).to_numpy(dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < MIN_LINES:
        continue
    lin = np.eye(K)[codes[ok]]
    r, p = partial_rho(pc2[ok], y[ok], lin)
    r_m, p_m = partial_rho(pc2[ok], y[ok], np.column_stack([lin, mapk[ok]]))
    rows.append((dataset, drug, int(ok.sum()), r, p, r_m, p_m))

res = pd.DataFrame(rows, columns=["dataset", "drug", "n", "rho", "p", "rho_mut_adj", "p_mut_adj"])
res["rank_among_drugs"] = res.groupby("dataset").rho.rank(ascending=False)
res["drugs_in_dataset"] = res.groupby("dataset").rho.transform("size")
res.to_csv(INT / "gdsc_pc2_all_drugs.csv", index=False)
print("drugs tested (at least 100 of our cell lines):", len(res), flush=True)
print("PC2 vs drug response; rho > 0 means less killing at higher PC2", flush=True)

mek = res[res.drug.isin(MEK)].sort_values(["dataset", "drug"])
print("MEK inhibitors in GDSC:", flush=True)
print(mek.round(4).to_string(index=False), flush=True)
mek.to_csv(OUT / "gdsc_mek_replication.csv", index=False)

for dataset, sub in res.groupby("dataset"):
    top = sub.sort_values("rho", ascending=False).head(10)
    print(f"Top 10 drugs by PC2 correlation in {dataset} (of {len(sub)}):", flush=True)
    print(top[["drug", "n", "rho", "p"]].round(4).to_string(index=False), flush=True)
    print(f"  share of drugs with positive rho: {(sub.rho > 0).mean():.2f}", flush=True)
print("saved: results/tables/gdsc_mek_replication.csv", flush=True)
