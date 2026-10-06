# Epigenetic and Transcriptomic Predictors of Drug Response in Cancer Cell Lines

Does DNA methylation predict how cancer cell lines respond to drugs beyond what tissue type, mutations and gene expression already explain? This project answers that on public data (CCLE, DepMap, PRISM, GDSC) and includes a small web app for browsing the results drug by drug and gene by gene.

## Findings

| Question | Answer |
|---|---|
| Does expression predict drug response beyond tissue and mutations? | Yes, by a wide margin: +0.13 in median Spearman correlation across 303 drugs, better for 99% of them. |
| Does methylation add to tissue and mutations? | Modestly: +0.03 (all regions), +0.04 (the 1,128 genes silenced by methylation), +0.06 when whole tissues are held out. |
| Does methylation add to expression? | No. The difference is slightly negative in every version tested (-0.003 to -0.011). |
| Is any drug signal linked to methylation? | One cluster: MEK inhibitors associate with a methylation pattern (PC2). Correlations are 0.24 to 0.29 in PRISM and 0.09 to 0.21 in GDSC. PC2 tracks cell identity (epithelial-like to non-epithelial), and expression explains the effect. |
| Do known biomarkers behave as expected? | In the expected direction: MGMT methylation with temozolomide (11 percentile points more killing, p = 0.009) and SLFN11 with DNA-damaging drugs (mostly not significant). |

Expression carries nearly all of the predictive information. Methylation holds real but largely redundant signal, and is most useful when expression is not available.

## Explorer app

```bash
git clone https://github.com/sanniakhan484-hash/epigenetic-and-transcriptomic-predictors-of-drug-response.git
cd epigenetic-and-transcriptomic-predictors-of-drug-response
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,app]"
streamlit run app/streamlit_app.py
```

The app reads the small tables in `data/processed/`, so it runs without downloading any raw data. Pages: Findings, Drugs (prediction by input type, links to epigenetic measures, GDSC replication, evidence level), Genes, About.

## Reproducing the analysis

Download the public input files listed in [data/README.md](data/README.md) into `data/raw/`, then run the scripts in order from the repository root, for example `python scripts/01_build_cell_lines.py`.

| Script | What it does |
|---|---|
| 01 to 03 | Cell line table, methylation matrix, drug response matrix |
| 04 | Expression matrix |
| 05 and 06 | Methylation principal components; structure of drug response across tissues |
| 07 to 10 | Genes silenced by methylation: correlation, region choice, methylated-group test, cutoff sensitivity |
| 11 | Known biomarkers (MGMT, SLFN11) |
| 12 | Scan of epigenetic state variables against drug response |
| 13 and 14 | MEK inhibitor signal: mutation check and GDSC replication |
| 15 | What methylation pattern 2 represents |
| 16 and 17 | Prediction benchmark and robustness checks (each takes about 10 to 20 minutes) |
| 18 | Net-benefit comparison against expression alone (post hoc) |
| 19 | Tables for the app |
| 20 | Figures |
| `qc/` | Checks on the input files, run while building the tables |

Checks: `ruff check .` and `pytest -q`.

## Repository layout

```
scripts/          analysis scripts, run in numbered order
src/drugresponse/ shared statistics code
tests/            unit tests
app/              Streamlit explorer
data/processed/   small result tables used by the app
results/          summary tables and figures
```

## Limitations

- Pan-cancer: only 27 breast cell lines have methylation, expression, mutation and drug data.
- Drug response is a single 2.5 µM dose over 5 days, a poor fit for DNMT inhibitors, which need several cell divisions.
- Benchmarked drugs were chosen for wide response spread.
- Cell lines are not patients, and the results are correlations, not causal effects.
- The benchmark gives every input type equal weight; other designs might value methylation differently.
- Any drug named in the app is a hypothesis for experimental follow-up, not a recommendation.

## Data sources

Methylation: CCLE reduced-representation bisulfite sequencing (2018, hg19). Expression and mutations: DepMap Public 26Q1. Drug response: PRISM Repurposing 24Q2 (Corsello et al., 2020, doi:10.1038/s43018-019-0018-6). Replication: Sanger GDSC1 and GDSC2 as distributed by DepMap.

## License

MIT. See [LICENSE](LICENSE).
