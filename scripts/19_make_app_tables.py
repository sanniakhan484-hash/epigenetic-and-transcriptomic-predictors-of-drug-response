import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

INT = Path("data/interim")
OUT = Path("results/tables")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

dmeta = pd.read_csv(INT / "drug_meta.csv", index_col=0)
dmeta.index.name = "treatment"
dmeta = dmeta.reset_index()
state = pd.read_csv(INT / "state_drug_assoc_all.csv")
bench = pd.read_csv(INT / "benchmark_random_cv.csv")
chk = pd.read_csv(INT / "benchmark_checks_random_cv.csv")
gdsc = pd.read_csv(INT / "gdsc_pc2_all_drugs.csv")
g_expr = pd.read_csv(INT / "gene_meth_expr.csv")
g_cut = pd.read_csv(INT / "gene_meth_group_cutoffs.csv", index_col=0)
robust = pd.read_csv(OUT / "robust_silenced_genes.csv", index_col=0)

drugs = dmeta.rename(columns={"Drug.Name": "drug", "repurposing_target": "target"})
drugs = drugs[["treatment", "drug", "screen", "dose", "MOA", "target", "n_lines", "n_breast"]]
best = state.sort_values("fdr").drop_duplicates("treatment").set_index("treatment")
best = best[["state", "rho", "fdr"]].rename(
    columns={"state": "best_state", "rho": "best_rho", "fdr": "best_fdr"}
)
drugs = drugs.join(best, on="treatment")
n_sig = state[state.fdr < 0.05].groupby("treatment").size().rename("n_states_fdr05")
drugs = drugs.join(n_sig, on="treatment")
drugs["n_states_fdr05"] = drugs["n_states_fdr05"].fillna(0).astype(int)
pc2 = state[state.state == "PC2"].set_index("treatment")[["rho", "fdr"]]
drugs = drugs.join(pc2.rename(columns={"rho": "pc2_rho", "fdr": "pc2_fdr"}), on="treatment")

model_cols = ["A_lineage", "B_lineage_mut", "C_lineage_mut_expr", "D_lineage_mut_meth",
              "E_all", "F_all_meth_shuffled"]
drugs = drugs.join(bench.set_index("treatment")[model_cols].add_prefix("bench_"), on="treatment")
chk_t = chk.set_index("treatment")[
    ["C_plus_expr", "Esil_expr_meth_silenced", "Fsil_expr_meth_silenced_shuffled"]
]
chk_t.columns = ["chk_expression", "chk_expression_plus_silenced_meth", "chk_shuffled_control"]
chk_t["chk_net_gain"] = chk_t["chk_expression_plus_silenced_meth"] - chk_t["chk_expression"]
drugs = drugs.join(chk_t, on="treatment")

gdsc["key"] = gdsc["drug"].str.upper()
for value in ["rho", "p"]:
    wide = gdsc.pivot_table(index="key", columns="dataset", values=value, aggfunc="first")
    drugs["key"] = drugs["drug"].astype(str).str.upper()
    drugs = drugs.join(wide.add_prefix(f"gdsc_pc2_{value}_"), on="key")
drugs = drugs.drop(columns="key")


def replicated(row):
    if pd.isna(row.pc2_rho) or pd.isna(row.pc2_fdr) or row.pc2_fdr >= 0.05:
        return False
    for ds in ["GDSC1", "GDSC2"]:
        r = row.get(f"gdsc_pc2_rho_{ds}")
        p = row.get(f"gdsc_pc2_p_{ds}")
        if pd.notna(r) and pd.notna(p) and np.sign(r) == np.sign(row.pc2_rho) and p < 0.05:
            return True
    return False


def tier(row):
    if row.replicated:
        return "replicated (PRISM + GDSC)"
    if row.best_fdr < 0.05:
        return "single screen (FDR < 0.05)"
    if row.best_fdr < 0.25:
        return "weak (FDR < 0.25)"
    return "no association"


drugs["replicated"] = drugs.apply(replicated, axis=1)
drugs["tier"] = drugs.apply(tier, axis=1)
drugs.to_parquet(PROC / "drugs.parquet", index=False)

state[["treatment", "state", "n_lines", "rho", "p", "fdr"]].to_parquet(
    PROC / "state_assoc.parquet", index=False
)

cut = g_cut.reset_index()
genes = g_expr.merge(cut, on="gene", how="left")
genes["robust_silenced"] = genes["gene"].isin(robust.index)
genes.to_parquet(PROC / "genes.parquet", index=False)

for name in ["benchmark_summary.csv", "benchmark_checks_summary.csv"]:
    if (OUT / name).exists():
        shutil.copy(OUT / name, PROC / name)
meta = {
    "treatments": len(drugs),
    "genes": len(genes),
    "robust_silenced_genes": len(robust),
    "sources": {
        "methylation": "CCLE RRBS TSS 1kb (2018-06-14), hg19",
        "expression": "DepMap Public 26Q1",
        "mutations": "DepMap Public 26Q1 hotspot matrix",
        "drug response": "PRISM Repurposing 24Q2 extended primary (single 2.5 uM dose, 5 days)",
        "replication": "Sanger GDSC1 and GDSC2 via DepMap (AUC_PUBLISHED)",
    },
}
(PROC / "meta.json").write_text(json.dumps(meta, indent=2))

print("drugs table:", drugs.shape, flush=True)
print("reliability tiers:", flush=True)
print(drugs["tier"].value_counts().to_string(), flush=True)
print("drugs with benchmark results:", int(drugs["bench_C_lineage_mut_expr"].notna().sum()),
      "| with robustness-check results:", int(drugs["chk_expression"].notna().sum()), flush=True)
print("state associations:", len(state), "| genes:", len(genes),
      "| robust silenced genes in gene table:", int(genes["robust_silenced"].sum()), flush=True)
for f in sorted(PROC.iterdir()):
    print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB", flush=True)
