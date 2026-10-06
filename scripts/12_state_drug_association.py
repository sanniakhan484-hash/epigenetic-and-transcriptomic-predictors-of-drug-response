from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
N_PERM = 10
SEED = 42
MIN_LINES = 100

meth = pd.read_parquet(INT / "methylation_matrix.parquet")
drug = pd.read_parquet(INT / "drug_matrix.parquet")
dmeta = pd.read_csv(INT / "drug_meta.csv", index_col=0)
cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
annot = pd.read_csv(INT / "tss_annotation.csv").dropna(subset=["gene"])
robust = pd.read_csv(OUT / "robust_silenced_genes.csv", index_col=0)
pcs = pd.read_csv(INT / "methylation_pcs.csv", index_col=0)

lines = [c for c in meth.columns if c in drug.columns and c in pcs.index]
n = len(lines)
print("cell lines:", n, "| treatments:", drug.shape[0], flush=True)

best = annot.sort_values("avg_coverage", ascending=False).drop_duplicates("gene").set_index("gene")
genes_r = [g for g in robust.index if g in best.index]
R = meth.loc[best.loc[genes_r, "TSS_id"], lines].to_numpy(dtype=float)
measured = ~np.isnan(R)
burden = ((R > 0.5) & measured).sum(axis=0) / np.maximum(measured.sum(axis=0), 1)
global_meth = np.nanmean(meth[lines].to_numpy(dtype=float), axis=0)
states = {
    "silencing_burden": burden,
    "global_methylation": global_meth,
    "PC1": pcs.loc[lines, "PC1"].to_numpy(dtype=float),
    "PC2": pcs.loc[lines, "PC2"].to_numpy(dtype=float),
    "PC3": pcs.loc[lines, "PC3"].to_numpy(dtype=float),
}
names = list(states)
S = np.vstack([states[k] for k in names])
print("robust silenced genes used for burden:", len(genes_r), flush=True)
assert not np.isnan(S).any(), "state variables contain missing values"

lineage = cells.loc[lines, "OncotreeLineage"].to_numpy()
codes, _ = pd.factorize(lineage)
K = int(codes.max()) + 1


def demean(a, onehot, counts):
    means = (a @ onehot) / np.maximum(counts, 1)
    return a - means @ onehot.T


def lineage_share(x):
    r = stats.rankdata(x)
    sizes = np.maximum(np.bincount(codes, minlength=K), 1)
    means = np.bincount(codes, weights=r, minlength=K) / sizes
    return 1 - ((r - means[codes]) ** 2).sum() / ((r - r.mean()) ** 2).sum()


print("State variables (share of each explained by lineage):", flush=True)
for k, v in states.items():
    print(f"  {k}: min {v.min():.3f}, median {np.median(v):.3f}, max {v.max():.3f}, "
          f"lineage share {lineage_share(v):.2f}", flush=True)

rng = np.random.default_rng(SEED)
blocks = [S]
for _ in range(N_PERM):
    idx = np.arange(n)
    for c in range(K):
        members = np.where(codes == c)[0]
        idx[members] = rng.permutation(members)
    blocks.append(S[:, idx])
X_all = np.vstack(blocks)

Y = drug[lines].to_numpy(dtype=float)
obs_rho = np.full((Y.shape[0], 5), np.nan)
obs_p = np.full((Y.shape[0], 5), np.nan)
obs_n = np.zeros(Y.shape[0], dtype=int)
null_p = np.full((Y.shape[0], N_PERM, 5), np.nan)

print("Testing all treatments...", flush=True)
for d in range(Y.shape[0]):
    ok = ~np.isnan(Y[d])
    m = int(ok.sum())
    if m < MIN_LINES:
        continue
    g = codes[ok]
    onehot = np.eye(K)[g]
    counts = onehot.sum(axis=0)
    k = int((counts > 0).sum())
    df = m - k - 1
    if df < 10:
        continue

    ya = demean(stats.rankdata(Y[d][ok])[None, :], onehot, counts)[0]
    xa = demean(stats.rankdata(X_all[:, ok], axis=1), onehot, counts)
    denom = np.sqrt((xa**2).sum(axis=1) * (ya**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.clip((xa * ya).sum(axis=1) / denom, -0.999999, 0.999999)
        t = r * np.sqrt(df / (1 - r**2))
    p = 2 * stats.t.sf(np.abs(t), df)
    obs_rho[d] = r[:5]
    obs_p[d] = p[:5]
    obs_n[d] = m
    null_p[d] = p[5:].reshape(N_PERM, 5)

tested = ~np.isnan(obs_p[:, 0])
print("treatments tested:", int(tested.sum()), "| tests:", int(tested.sum()) * 5, flush=True)

rows = []
for d in np.where(tested)[0]:
    for s in range(5):
        rows.append((dmeta.index[d], names[s], obs_n[d], obs_rho[d, s], obs_p[d, s]))
res = pd.DataFrame(rows, columns=["treatment", "state", "n_lines", "rho", "p"])
res["fdr"] = multipletests(res.p, method="fdr_bh")[1]
res = res.join(dmeta[["Drug.Name", "MOA", "screen"]], on="treatment")
res.to_csv(INT / "state_drug_assoc_all.csv", index=False)

hit = res.fdr < 0.05
print("associations with FDR < 0.05:", int(hit.sum()), flush=True)
print(res[hit].groupby("state").size().to_string(), flush=True)
if hit.any():
    p_star = float(res.loc[hit, "p"].max())
    null_counts = [(null_p[tested][:, i, :] <= p_star).sum() for i in range(N_PERM)]
    print(f"permutation check at p <= {p_star:.2e}: observed {int(hit.sum())} hits, "
          f"shuffled data average {np.mean(null_counts):.1f} "
          f"(estimated false discovery rate {np.mean(null_counts) / int(hit.sum()):.3f})",
          flush=True)

show = ["Drug.Name", "MOA", "state", "n_lines", "rho", "fdr", "screen"]
top = res.sort_values("p").head(25)
print("Top 25 associations (rho > 0: higher state, less killing; rho < 0: more killing):",
      flush=True)
print(top[show].round(4).to_string(index=False), flush=True)
res.sort_values("p").head(200)[["treatment"] + show].to_csv(OUT / "state_drug_top200.csv", index=False)

ctrl = ["AZACITIDINE", "DECITABINE", "GUADECITABINE", "VORINOSTAT", "BELINOSTAT",
        "PANOBINOSTAT", "ROMIDEPSIN", "OLAPARIB", "TALAZOPARIB", "NIRAPARIB", "RUCAPARIB"]
two = ["silencing_burden", "global_methylation"]
sub = res[res["Drug.Name"].str.upper().isin(ctrl) & res.state.isin(two)].copy()
sub["txt"] = sub.rho.round(2).astype(str) + " (FDR " + sub.fdr.round(3).astype(str) + ")"
print("Epigenetic and PARP drugs: rho (FDR) for two states:", flush=True)
print(sub.pivot_table(index="Drug.Name", columns="state", values="txt", aggfunc="first")
      .to_string(), flush=True)
print("saved: data/interim/state_drug_assoc_all.csv and results/tables/state_drug_top200.csv",
      flush=True)
