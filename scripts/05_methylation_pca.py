from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

INT = Path("data/interim")

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
mat = pd.read_parquet(INT / "methylation_matrix.parquet")


def subtype_group(label):
    s = str(label)
    if "TNBC" in s:
        return "TNBC"
    if s in ("HER2+", "ER-/PR-/HER2+"):
        return "HER2+ (ER-)"
    if "ER" in s:
        return "ER+"
    return "unknown"


cells["subtype_group"] = cells.ModelSubtypeFeatures.map(subtype_group)
cells.loc[cells.OncotreeLineage != "Breast", "subtype_group"] = "not breast"

print("How breast labels were grouped:", flush=True)
breast_cells = cells[cells.OncotreeLineage == "Breast"]
print(breast_cells.groupby(["subtype_group", "ModelSubtypeFeatures"], dropna=False).size().to_string(), flush=True)

keep = mat.isna().mean(axis=1) <= 0.10
x = mat[keep]
top = x.var(axis=1).sort_values(ascending=False).index[:5000]
x = x.loc[top]
print("regions kept after missingness filter:", int(keep.sum()), "| used for PCA:", x.shape[0], flush=True)

X = x.T.fillna(x.mean(axis=1))

pca = PCA(n_components=10, random_state=42)
scores = pd.DataFrame(
    pca.fit_transform(X.to_numpy()),
    index=X.index,
    columns=[f"PC{i}" for i in range(1, 11)],
)


def eta2(values, groups):
    """Share of the variation in `values` that is explained by group membership (0 to 1)."""
    total = ((values - values.mean()) ** 2).sum()
    between = sum(len(v) * (v.mean() - values.mean()) ** 2 for _, v in values.groupby(groups))
    return between / total


lineage = cells.loc[scores.index, "OncotreeLineage"]
big = lineage.map(lineage.value_counts()) >= 5
res_lineage = {pc: eta2(scores.loc[big, pc], lineage[big]) for pc in scores.columns}

breast_ids = breast_cells.index
res_breast = {
    pc: eta2(scores.loc[breast_ids, pc], breast_cells.loc[breast_ids, "subtype_group"])
    for pc in scores.columns
}

out = pd.DataFrame(
    {
        "var_explained": pca.explained_variance_ratio_,
        "eta2_lineage": pd.Series(res_lineage),
        "eta2_breast_subtype": pd.Series(res_breast),
    }
)
print(out.round(3).to_string(), flush=True)

scores.join(cells[["OncotreeLineage", "subtype_group"]]).to_csv(INT / "methylation_pcs.csv")
print("saved:", INT / "methylation_pcs.csv", flush=True)
