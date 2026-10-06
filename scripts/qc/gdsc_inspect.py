from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")
TERMS = [
    "TRAMETINIB", "SELUMETINIB", "PD0325901", "PD-0325901", "REFAMETINIB", "RDEA119",
    "CI-1040", "CI1040", "BINIMETINIB", "MEK162", "COBIMETINIB", "PIMASERTIB", "TAK-733",
    "GDC-0623", "GDC0623", "AS-703026", "AS703026", "RO-4987655", "RO4987655", "MEK",
]

df = pd.read_csv(RAW / "sanger-dose-response.csv", low_memory=False)
print("shape:", df.shape, flush=True)
print("columns (type, unique values, share missing, example):", flush=True)
for c in df.columns:
    s = df[c]
    ex = s.dropna().iloc[0] if s.notna().any() else None
    print(f"  {c}: {s.dtype}, unique {s.nunique()}, missing {s.isna().mean():.2f}, "
          f"example {ex!r}", flush=True)

id_col = None
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        vals = df[c].dropna().astype(str)
        if len(vals) and vals.str.startswith("ACH-").mean() > 0.9:
            id_col = c
            break
print("DepMap ID column:", id_col, flush=True)

name_cols = [
    c for c in df.columns
    if not pd.api.types.is_numeric_dtype(df[c])
    and any(k in c.lower() for k in ["drug", "compound", "name"])
]
print("possible drug-name columns:", name_cols, flush=True)

pattern = "|".join(TERMS)
for c in name_cols:
    hit = df[df[c].astype(str).str.upper().str.contains(pattern, na=False)]
    if len(hit) == 0:
        continue
    per = hit.groupby(c)[id_col].nunique() if id_col else hit.groupby(c).size()
    print(f"MEK-like values in column {c} (number of cell lines):", flush=True)
    print(per.sort_values(ascending=False).head(25).to_string(), flush=True)

if id_col:
    cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
    have = set(df[id_col].dropna())
    ours = [c for c in cells.index if c in have]
    print("of our 623 cell lines, in this file:", len(ours), flush=True)
    breast = [c for c in ours if cells.loc[c, "OncotreeLineage"] == "Breast"]
    print("  of which breast:", len(breast), flush=True)
