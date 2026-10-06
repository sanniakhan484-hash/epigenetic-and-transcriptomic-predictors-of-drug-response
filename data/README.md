# Data

Raw and intermediate data are not stored in this repository. Download these public files into `data/raw/` (create the folder if it does not exist):

| File | Source on the DepMap portal (https://depmap.org/portal/data_page/) |
|---|---|
| `CCLE_RRBS_TSS_1kb_20180614.txt` | All Data, Methylation (RRBS) |
| `Model.csv` | Current release (DepMap Public 26Q1), Model, Conditions, and Mapping |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | Current release, Expression |
| `OmicsSomaticMutationsMatrixHotspot.csv` | Current release, Mutations |
| `Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv` | All Data, PRISM Repurposing Primary Screen |
| `Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv` | All Data, PRISM Repurposing Primary Screen |
| `sanger-dose-response.csv` | All Data, Sanger GDSC1 and GDSC2 |

The scripts write intermediate tables to `data/interim/` (created on first run). `data/processed/` holds the small final tables used by the app.
