from pathlib import Path

import numpy as np
import pandas as pd

INT = Path("data/interim")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
M_E = "Esil_expr_meth_silenced"
M_C = "C_plus_expr"
M_F = "Fsil_expr_meth_silenced_shuffled"

rcv = pd.read_csv(INT / "benchmark_checks_random_cv.csv")
rcv["gain_vs_none"] = rcv[M_E] - rcv[M_C]
rcv["gain_vs_shuffled"] = rcv[M_E] - rcv[M_F]
rcv["shuffle_penalty"] = rcv[M_C] - rcv[M_F]
print("treatments:", len(rcv), flush=True)
print(f"median shuffle penalty (expression-only minus shuffled): "
      f"{rcv['shuffle_penalty'].median():+.4f}", flush=True)


def noise_scale(values):
    neg = values[values < 0]
    return 1.4826 * float(np.median(np.abs(neg))) if len(neg) else float("nan")


sig_none = noise_scale(rcv["gain_vs_none"])
sig_shuf = noise_scale(rcv["gain_vs_shuffled"])
pos_none = rcv[rcv["gain_vs_none"] > 3 * sig_none]
neg_none = rcv[rcv["gain_vs_none"] < -3 * sig_none]
pos_shuf = rcv[rcv["gain_vs_shuffled"] > 3 * sig_shuf]
print(f"gain vs shuffled (the rule's comparison): noise {sig_shuf:.4f}, "
      f"drugs above 3x noise: {len(pos_shuf)}", flush=True)
print(f"gain vs expression-only (net benefit): noise {sig_none:.4f}, "
      f"drugs above 3x noise: {len(pos_none)}, below -3x noise: {len(neg_none)}", flush=True)
only_penalty = pos_shuf[pos_shuf["gain_vs_none"] <= 3 * sig_none]
print(f"flagged by the rule but with no net benefit over expression-only: "
      f"{len(only_penalty)}", flush=True)

cols = ["treatment", "drug", "n", M_C, M_E, "gain_vs_none", "gain_vs_shuffled"]
print("Drugs with a net benefit from silenced-gene methylation:", flush=True)
print(pos_none.sort_values("gain_vs_none", ascending=False)[cols].round(3).to_string(index=False),
      flush=True)
print("Drugs where adding it hurt by more than 3x noise:", flush=True)
print(neg_none.sort_values("gain_vs_none")[cols].round(3).head(10).to_string(index=False),
      flush=True)
pos_none[cols].to_csv(OUT / "subset_drugs_net_benefit.csv", index=False)
print("saved: results/tables/subset_drugs_net_benefit.csv", flush=True)
