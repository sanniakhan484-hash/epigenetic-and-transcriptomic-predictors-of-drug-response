from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from drugresponse.stats import adjusted_corr

INT = Path("data/interim")
PROC = Path("data/processed")
TABLES = Path("results/tables")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERMILION, GREY, SKY = "#D55E00", "#7F7F7F", "#56B4E9"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 9,
})
MEK = ["MEK162", "GDC-0623", "TRAMETINIB", "AS-703026", "RO-4987655", "PD-0325901",
       "REFAMETINIB", "SELUMETINIB"]
MARKERS = {
    "Epithelial": ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "GRHL2"],
    "Mesenchymal": ["VIM", "ZEB1", "ZEB2", "FN1", "CDH2", "SNAI2", "TWIST1"],
    "MAPK output": ["DUSP6", "ETV4", "ETV5", "SPRY2", "EGR1"],
    "Proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2"],
}
GROUP_COLOR = {"Epithelial": BLUE, "Mesenchymal": VERMILION, "MAPK output": GREEN,
               "Proliferation": GREY}


def tag(ax, text):
    ax.text(-0.14, 1.06, text, transform=ax.transAxes, fontsize=11, fontweight="bold")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


genes = pd.read_csv(INT / "gene_meth_expr.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
bins = np.linspace(-0.7, 0.7, 57)
n_raw = int((genes.rho_raw < -0.3).sum())
n_adj = int((genes.rho_adj < -0.3).sum())
a.hist(genes.rho_raw, bins=bins, color=GREY, alpha=0.6, label=f"Raw ({n_raw:,} genes below -0.3)")
a.hist(genes.rho_adj, bins=bins, color=BLUE, alpha=0.7,
       label=f"Tissue-adjusted ({n_adj:,} below -0.3)")
a.axvline(-0.3, ymax=0.6, ls="--", lw=0.8, color="black")
a.set_xlabel("Spearman correlation, methylation vs expression")
a.set_ylabel("Genes")
a.set_ylim(top=a.get_ylim()[1] * 1.15)
a.legend(frameon=False, fontsize=8, loc="upper left")
tag(a, "a")
known = ["MGMT", "CDH1", "RASSF1", "ESR1", "CDKN2A", "MLH1", "BRCA1"]
sub = genes.set_index("gene").reindex(known).rho_adj.dropna().sort_values()
b.barh(sub.index, sub.values, color=[BLUE if v < 0 else ORANGE for v in sub.values])
b.axvline(0, color="black", lw=0.8)
b.set_xlabel("Tissue-adjusted correlation")
tag(b, "b")
save(fig, "fig1_silencing")

drugs = pd.read_parquet(PROC / "drugs.parquet")
gd = pd.read_csv(INT / "gdsc_pc2_all_drugs.csv")
drugs["key"] = drugs["drug"].astype(str).str.upper()
gd["key"] = gd["drug"].astype(str).str.upper()
prism = drugs[drugs["screen"].eq("REP.PRIMARY") & drugs["key"].isin(MEK)].set_index("key")["pc2_rho"]
names = sorted(set(prism.index) | set(gd.loc[gd["key"].isin(MEK), "key"]))
best = {n: max([prism.get(n, np.nan)] + list(gd.loc[gd["key"] == n, "rho"])) for n in names}
names = sorted(names, key=lambda n: best[n])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for i, n in enumerate(names):
    if n in prism.index:
        ax.scatter(prism[n], i, color=BLUE, marker="o", s=34, zorder=3)
    for ds, color, marker in [("GDSC1", ORANGE, "s"), ("GDSC2", GREEN, "D")]:
        v = gd[(gd["key"] == n) & (gd["dataset"] == ds)]["rho"]
        if len(v):
            ax.scatter(v.iloc[0], i, color=color, marker=marker, s=34, zorder=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Correlation of response with methylation pattern 2 (tissue-adjusted)")
for label, color, marker in [("PRISM", BLUE, "o"), ("GDSC1", ORANGE, "s"), ("GDSC2", GREEN, "D")]:
    ax.scatter([], [], color=color, marker=marker, s=34, label=label)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
save(fig, "fig2_mek_replication")

cells = pd.read_csv(INT / "cell_lines.csv").set_index("ModelID")
pcs = pd.read_csv(INT / "methylation_pcs.csv", index_col=0)
expr = pd.read_parquet(INT / "expression_matrix.parquet")
pc2 = pcs["PC2"]
lin = cells["OncotreeLineage"].reindex(pc2.index)
by = pc2.groupby(lin).agg(["mean", "size"])
by = by[by["size"] >= 10].sort_values("mean")
lines = [c for c in pc2.index if c in expr.columns]
E = expr[lines]
E = E[~E.isna().any(axis=1)]
codes, _ = pd.factorize(lin[lines])
rho = pd.Series(adjusted_corr(E.to_numpy(float), pc2[lines].to_numpy(float), codes), index=E.index)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
a.barh(by.index, by["mean"], color=SKY)
a.axvline(0, color="black", lw=0.8)
a.set_xlabel("Mean methylation pattern 2")
tag(a, "a")
ys, labels, colors = [], [], []
for group, members in MARKERS.items():
    for gene in members:
        if gene in rho.index:
            ys.append(rho[gene])
            labels.append(gene)
            colors.append(GROUP_COLOR[group])
pos = np.arange(len(ys))[::-1]
b.barh(pos, ys, color=colors)
b.set_yticks(pos)
b.set_yticklabels(labels, fontsize=7)
b.axvline(0, color="black", lw=0.8)
b.set_xlabel("Expression correlation with\npattern 2 (tissue-adjusted)")
handles = [Patch(color=color, label=group) for group, color in GROUP_COLOR.items()]
b.legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
tag(b, "b")
save(fig, "fig3_pc2")

bench = pd.read_csv(INT / "benchmark_random_cv.csv")
models = ["A_lineage", "B_lineage_mut", "C_lineage_mut_expr", "D_lineage_mut_meth",
          "E_all", "F_all_meth_shuffled"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
box = a.boxplot([bench[m].dropna() for m in models], tick_labels=list("ABCDEF"), showfliers=False,
                patch_artist=True, medianprops={"color": "black"})
for patch in box["boxes"]:
    patch.set_facecolor(SKY)
a.set_ylabel("Prediction accuracy (Spearman)")
a.set_xlabel("Model")
tag(a, "a")
contrasts = [("C", "B", "C - B"), ("D", "B", "D - B"), ("E", "C", "E - C"), ("E", "F", "E - F")]
col = dict(zip("ABCDEF", models))
diffs = [(bench[col[x]] - bench[col[y]]).dropna() for x, y, _ in contrasts]
box = b.boxplot(diffs, tick_labels=[c[2] for c in contrasts], showfliers=False,
                patch_artist=True, medianprops={"color": "black"})
for patch in box["boxes"]:
    patch.set_facecolor(ORANGE)
b.axhline(0, color="black", lw=0.8)
b.set_ylabel("Change in accuracy")
b.set_xlabel("Comparison")
tag(b, "b")
fig.subplots_adjust(wspace=0.45)
save(fig, "fig4_benchmark")

s = pd.read_csv(TABLES / "benchmark_checks_summary.csv")
s["key"] = s["comparison"].str.split(":").str[0]
plain = {
    "Esil vs C": "Silenced-gene methylation\nadded to expression",
    "Esil vs Fsil": "Real vs shuffled\nsilenced-gene methylation",
    "Dsil vs B": "Silenced-gene methylation\nadded to tissue + mutations",
    "E vs C": "All-region methylation\nadded to expression",
    "C vs B": "Expression added to\ntissue + mutations",
}
keys = [k for k in plain if k in set(s["key"])]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for ev, label, color, off in [("random_cv", "Random 5-fold", BLUE, 0.12),
                              ("lolo_within", "Tissue held out", ORANGE, -0.12)]:
    d = s[s["evaluation"] == ev].set_index("key").reindex(keys)
    y = np.arange(len(keys))[::-1] + off
    ax.errorbar(d["median_diff"], y,
                xerr=[d["median_diff"] - d["ci_low"], d["ci_high"] - d["median_diff"]],
                fmt="o", color=color, capsize=2.5, label=label)
ax.set_yticks(np.arange(len(keys))[::-1])
ax.set_yticklabels([plain[k] for k in keys])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Median change in accuracy across drugs (95% interval)")
ax.legend(frameon=False, loc="upper right")
save(fig, "fig5_robustness")
