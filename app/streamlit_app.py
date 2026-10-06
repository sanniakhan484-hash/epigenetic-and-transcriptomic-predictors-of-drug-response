from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUTS = {
    "bench_A_lineage": "A",
    "bench_B_lineage_mut": "B",
    "bench_C_lineage_mut_expr": "C",
    "bench_D_lineage_mut_meth": "D",
    "bench_E_all": "E",
    "bench_F_all_meth_shuffled": "F",
}
LEGEND = (
    "A tissue only | B + mutations | C + expression | D + methylation (no expression) | "
    "E expression + methylation | F control with scrambled methylation. Higher is better."
)
TIER_TEXT = {
    "replicated (PRISM + GDSC)": "Seen in two independent drug screens.",
    "single screen (FDR < 0.05)": "Seen in one screen only. Treat it as a lead.",
    "weak (FDR < 0.25)": "A weak signal that would not hold up under strict correction.",
    "no association": "No link to any of the epigenetic measures tested.",
}
PLAIN = {
    "C vs B": "Expression added to tissue + mutations",
    "D vs B": "Methylation added to tissue + mutations",
    "E vs C": "Methylation added to expression",
    "E vs F": "Real vs scrambled methylation",
    "F vs C": "Scrambled methylation vs none",
    "B vs A": "Mutations added to tissue",
}
MEASURES = {
    "silencing_burden": "Share of silenced genes that are methylated",
    "global_methylation": "Overall methylation level",
    "PC1": "Methylation pattern 1 (PC1)",
    "PC2": "Methylation pattern 2 (PC2)",
    "PC3": "Methylation pattern 3 (PC3)",
}

st.set_page_config(page_title="Drug response predictors", layout="wide")


@st.cache_data
def load():
    return (
        pd.read_parquet(PROC / "drugs.parquet"),
        pd.read_parquet(PROC / "state_assoc.parquet"),
        pd.read_parquet(PROC / "genes.parquet"),
        pd.read_csv(PROC / "benchmark_summary.csv"),
    )


def friendly(name):
    for key, text in PLAIN.items():
        if str(name).startswith(key):
            return text
    return str(name)


def fmt_fdr(x):
    if pd.isna(x):
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


drugs, state, genes, summary = load()
page = st.sidebar.radio("Go to", ["Findings", "Drugs", "Genes", "About"])
st.sidebar.caption("Research tool built on cancer cell line data. Not medical advice.")

if page == "Findings":
    st.title("Does DNA methylation help predict how cancer cells respond to drugs?")
    st.write(
        "Gene expression explains drug response far better than DNA methylation. Once "
        "expression is known, methylation adds almost nothing, though it can stand in when "
        "expression is missing."
    )
    table = summary.copy()
    table["What was added"] = table["comparison"].map(friendly)
    gains = dict(zip(summary["comparison"].str[:6], summary["median_diff"]))
    cols = st.columns(3)
    for col, (key, label) in zip(cols, [("C vs B", "Expression, on top of tissue + mutations"),
                                        ("D vs B", "Methylation, on top of tissue + mutations"),
                                        ("E vs C", "Methylation, on top of expression")]):
        col.metric(label, f"{gains[key]:+.3f}" if key in gains else "n/a")
    st.bar_chart(table.set_index("What was added")["median_diff"], horizontal=True)
    st.caption("Change in prediction accuracy (Spearman correlation), median across drugs, "
               "tested on cell lines the model had not seen.")
    show = table[["What was added", "median_diff", "ci_low", "ci_high", "share_improved"]]
    show = show.rename(columns={"median_diff": "Median change", "ci_low": "Low (95%)",
                                "ci_high": "High (95%)", "share_improved": "Share of drugs better"})
    st.dataframe(show.round(3), hide_index=True)

elif page == "Drugs":
    st.title("Drugs")
    with st.expander("Filter by evidence"):
        tiers = sorted(drugs["tier"].unique())
        chosen = st.multiselect("Evidence level", tiers, default=tiers)
    sub = drugs[drugs["tier"].isin(chosen)].copy()
    if sub.empty:
        st.info("No drugs match this filter.")
        st.stop()
    sub["label"] = sub["drug"].astype(str) + " [" + sub["screen"].astype(str) + "] " + sub["treatment"]
    row = sub[sub["label"] == st.selectbox("Search for a drug", sub["label"].tolist())].iloc[0]
    st.subheader(str(row["drug"]))
    st.write(f"**Evidence:** {row['tier']}. {TIER_TEXT.get(row['tier'], '')}")
    moa = row["MOA"] if pd.notna(row["MOA"]) else "not listed"
    st.write(f"**Mechanism:** {moa}  \n**Cell lines with data:** {int(row['n_lines'])}")
    st.markdown("#### How well can each kind of data predict this drug?")
    vals = {INPUTS[c]: row[c] for c in INPUTS if c in row.index and pd.notna(row[c])}
    if vals:
        data = pd.DataFrame({"Input": list(vals), "Accuracy": list(vals.values())})
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X("Input:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Accuracy:Q", title="Prediction accuracy (Spearman)"),
        )
        st.altair_chart(chart)
        st.caption(LEGEND)
    else:
        st.info("This drug wasn't part of the prediction benchmark, which covered 303 drugs "
                "with the widest range of responses.")
    if pd.notna(row.get("chk_net_gain")):
        st.write(f"Adding methylation of the silenced genes to expression changes accuracy "
                 f"by {row['chk_net_gain']:+.3f}.")
    st.markdown("#### Link to epigenetic measures")
    assoc = state[state["treatment"] == row["treatment"]].drop(columns="treatment").copy()
    assoc["state"] = assoc["state"].map(lambda s: MEASURES.get(s, s))
    assoc = assoc.rename(columns={"state": "Measure", "n_lines": "Cell lines",
                                  "rho": "Correlation", "p": "p-value", "fdr": "FDR"})
    st.dataframe(assoc.round(4), hide_index=True)
    gd = {c.replace("gdsc_pc2_rho_", ""): row[c] for c in row.index
          if c.startswith("gdsc_pc2_rho_") and pd.notna(row[c])}
    if gd:
        st.write("Second screen (GDSC), correlation with methylation pattern 2: "
                 + ", ".join(f"{k} {v:+.2f}" for k, v in gd.items()))
    st.caption("Only methylation pattern 2 could be checked in the second screen.")

elif page == "Genes":
    st.title("Genes")
    gene = st.selectbox("Search for a gene", sorted(genes["gene"].astype(str).unique()))
    g = genes[genes["gene"] == gene].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Methylation vs expression", f"{g['rho_adj']:+.2f}",
              help="Correlation across cell lines of the same tissue. Negative means more "
                   "methylation goes with less expression.")
    c2.metric("FDR", fmt_fdr(g["fdr_adj"]))
    c3.metric("In the 1,128 silenced genes", "yes" if g["robust_silenced"] else "no")
    s5, f5 = g.get("shift@0.5"), g.get("fdr@0.5")
    if g["robust_silenced"]:
        st.write("Methylation reliably switches this gene down.")
    elif pd.notna(s5) and pd.notna(f5) and s5 > 20 and f5 < 0.05:
        st.write("Methylated cell lines express this gene more, the opposite of silencing.")
    elif g["rho_adj"] < -0.3:
        st.write("More methylation goes with less expression, but not strongly enough "
                 "for the silenced set.")
    else:
        st.write("No clear link between methylation and expression for this gene.")
    rows = []
    for cutoff in ["0.3", "0.5", "0.7"]:
        shift = g.get(f"shift@{cutoff}")
        rows.append({
            "Methylation above": cutoff,
            "Change in expression (percentile points)": "n/a" if pd.isna(shift) else f"{shift:+.1f}",
            "FDR": fmt_fdr(g.get(f"fdr@{cutoff}")),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("How much higher (+) or lower (-) the gene's expression is in cell lines whose "
               "methylation is above each cutoff, compared with lines of the same tissue below "
               "it. n/a means too few cell lines on one side to test.")

else:
    st.title("About")
    st.markdown(
        "**What this is.** A comparison of what different kinds of molecular data add to "
        "predicting how cancer cell lines respond to drugs, with a focus on DNA methylation.\n\n"
        "**Where the data comes from.** Methylation from the Cancer Cell Line Encyclopedia "
        "(2018). Gene expression and mutations from DepMap release 26Q1. Drug response from the "
        "PRISM Repurposing screen (one dose, five days). A second screen, GDSC, is used to "
        "confirm results.\n\n"
        "**How predictions are tested.** Each model learns from some cell lines and is scored on "
        "others it has not seen. Controls with scrambled methylation show what real methylation "
        "adds.\n\n"
        "**Limits.** Only 27 breast cell lines have every measurement, so results cover cancer "
        "in general. A single dose suits some drugs poorly; DNA-demethylating drugs need longer. "
        "Everything here is a correlation in cell lines. Any drug shown is a lead for lab "
        "testing, not a recommendation."
    )
