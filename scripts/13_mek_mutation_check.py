from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path("data/raw")
INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
MAPK_GENES = ["BRAF", "NRAS", "KRAS", "HRAS"]
MEK = ["MEK162", "GDC-0623", "TRAMETINIB", "AS-703026", "RO-4987655", "PD-0325901"]

mut = pd.read_csv(RAW / "OmicsSomaticMutationsMatrixHotspot.csv", low_memory=False)
print("mutation file shape:", mut.shape, flush=True)
print("first 6 columns:", list(mut.columns[:6]), flush=True)

id_col = None
for c in mut.columns:
    if mut[c].astype(str).str.startswith("ACH-").all():
        id_col = c
        break
if id_col is None:
    raise SystemExit("could not find a column of ACH- IDs")

flag = "IsDefaultEntryForModel"
if flag in mut.columns:
    keep = mut[flag].astype(str).str.lower().isin(["yes", "true", "1"])
    print("rows marked as the default entry for the model:", int(keep.sum()), flush=True)
    mut = mut[keep]
mut = mut.set_index(id_col)
dups = int(mut.index.duplicated().sum())
print("cell lines in file:", len(mut), "| duplicated IDs after filtering:", dups, flush=True)
mut = mut[~mut.index.duplicated()]

symbol = {c: str(c).split(" (")[0] for c in mut.columns}
found = [c for c in mut.columns if symbol[c] in MAPK_GENES]
print("MAPK columns found:", found, flush=True)
if not found:
    raise SystemExit("no BRAF/NRAS/KRAS/HRAS columns found")
for c in found:
    vals = pd.Series(mut[c].to_numpy()).value_counts(dropna=False).to_dict()
    print(f"  {c} values: {vals}", flush=True)

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
pcs = pd.read_csv(INT / "methylation_pcs.csv", index_col=0)
drug = pd.read_parquet(INT / "drug_matrix.parquet")
dmeta = pd.read_csv(INT / "drug_meta.csv", index_col=0)

lines = [c for c in cells.index if c in mut.index and c in pcs.index and c in drug.columns]
print("cell lines with mutation, methylation and drug data:", len(lines), flush=True)
breast = [c for c in lines if cells.loc[c, "OncotreeLineage"] == "Breast"]
print("  of which breast:", len(breast), flush=True)

mapk = (mut.loc[lines, found].fillna(0) > 0).any(axis=1).to_numpy().astype(float)
print("MAPK-mutant lines (BRAF, NRAS, KRAS or HRAS):", int(mapk.sum()), flush=True)
lineage = cells.loc[lines, "OncotreeLineage"].to_numpy()
top = pd.Series(lineage[mapk > 0]).value_counts().head(6).to_dict()
print("  lineages of MAPK-mutant lines:", top, flush=True)

codes, _ = pd.factorize(lineage)
K = int(codes.max()) + 1
pc2 = pcs.loc[lines, "PC2"].to_numpy(dtype=float)


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


names = dmeta["Drug.Name"].str.upper()
rows = []
for drug_id in dmeta.index[names.isin(MEK)]:
    y = drug.loc[drug_id, lines].to_numpy(dtype=float)
    ok = ~np.isnan(y)
    lin = np.eye(K)[codes[ok]]
    with_mut = np.column_stack([lin, mapk[ok]])
    r_lin, p_lin = partial_rho(pc2[ok], y[ok], lin)
    r_mut, p_mut = partial_rho(pc2[ok], y[ok], with_mut)
    r_my, p_my = partial_rho(mapk[ok], y[ok], lin)
    rows.append((dmeta.loc[drug_id, "Drug.Name"], int(ok.sum()), r_lin, p_lin, r_mut, p_mut, r_my))

cols = ["drug", "n", "rho_PC2_lineage", "p", "rho_PC2_lineage_and_mutation", "p_adj", "rho_mutation_resp"]
res = pd.DataFrame(rows, columns=cols)
print("MEK inhibitors (PC2 vs response; rho > 0 means less killing):", flush=True)
print(res.round(4).to_string(index=False), flush=True)
res.to_csv(OUT / "mek_pc2_mutation_check.csv", index=False)

lin_all = np.eye(K)[codes]
r, p = partial_rho(pc2, mapk, lin_all)
print(f"PC2 vs MAPK-mutation status (lineage-adjusted): rho {r:.3f}, p {p:.3g}", flush=True)
print("saved:", OUT / "mek_pc2_mutation_check.csv", flush=True)
