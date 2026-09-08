# NHIS Mental Health Access Study (2020–2024)

**Among people experiencing mental health symptoms, who is unable to access care because of the cost? Findings from NHIS: 2020–2024**

## Overview

This repository contains the analysis code for a pooled cross-sectional study using the National Health Interview Survey (NHIS) 2020–2024 public use data. The study examines sociodemographic and identity-based predictors of cost-related delayed mental health care among U.S. adults who report depression or anxiety symptoms.

**Outcome:** Delayed counseling or therapy due to cost (MHTHDLY_A)  
**Inclusion criteria:** Adults reporting any frequency of depression (DEPFREQ_A) or anxiety (ANXFREQ_A) symptoms (codes 1–4)  
**Key exposures:** Sexual orientation, insurance status, race/ethnicity, sex, age, education, survey year

## Repository Structure

```
nhis-mh-access/
├── scripts/
│   ├── nhis_build_pooled.py    # Step 1–4: build analytic dataset, recode variables
│   ├── nhis_table1.py          # Step 5:   weighted Table 1 (sample characteristics)
│   ├── nhis_regression.py      # Step 6:   weighted logistic regression (Table 2)
│   └── nhis_sensitivity.py     # Steps 7+: six pre-registered sensitivity analyses
├── output/                     # Analysis output tables (CSV)
│   ├── table1.csv
│   ├── table2_regression.csv
│   └── sa[1-6]_*.csv
├── data/                       # NOT committed — place raw NHIS files here (see below)
└── README.md
```

## Data

Raw NHIS public use data files are **not included** in this repository. Download each year's Sample Adult CSV file from NCHS:

- [2020 NHIS](https://www.cdc.gov/nchs/nhis/2020nhis.htm)
- [2021 NHIS](https://www.cdc.gov/nchs/nhis/2021nhis.htm)
- [2022 NHIS](https://www.cdc.gov/nchs/nhis/2022nhis.htm)
- [2023 NHIS](https://www.cdc.gov/nchs/nhis/2023nhis.htm)
- [2024 NHIS](https://www.cdc.gov/nchs/nhis/2024nhis.htm)

Unzip each file into `data/` following this folder structure:

```
data/
├── adult20csv/adult20.csv
├── adult21csv/adult21.csv
├── adult22csv/adult22.csv
├── adult23csv/adult23.csv
└── adult24csv/adult24.csv
```

To use a different data location, set the `NHIS_DATA_DIR` environment variable before running any script:

```bash
export NHIS_DATA_DIR="/path/to/your/nhis/data"
```

## How to Run

Scripts must be run in order. Each script reads the analytic CSV produced by `nhis_build_pooled.py`.

```bash
python scripts/nhis_build_pooled.py   # builds data/nhis_analytic_2020_2024.csv
python scripts/nhis_table1.py         # writes output/table1.csv
python scripts/nhis_regression.py     # writes output/table2_regression.csv
python scripts/nhis_sensitivity.py    # writes output/sa1–sa6 CSVs
```

## Dependencies

- Python 3.9+
- pandas
- rpy2 (Python–R bridge)
- R 4.5+ with the `survey` package

Install Python dependencies:

```bash
pip install pandas rpy2
```

Install R `survey` package (run once in R):

```r
install.packages("survey")
```

**Note on rpy2 and R version:** If you upgrade R, reinstall rpy2 to re-link against the new libraries:

```bash
R_HOME=/Library/Frameworks/R.framework/Resources pip install --force-reinstall rpy2
```

## Survey Design Notes

- Pooled 5-year weights: `WTFA_5YR = WTFA_A / 5` (standard NCHS pooling approach)
- Strata and PSU IDs are made year-unique to prevent cross-year conflation: `PPSU_POOL = year_PSTRAT_PPSU`
- `options(survey.lonely.psu = "adjust")` applied in all scripts
- Regression uses `svyglm` with `quasibinomial(logit)` family

## Sensitivity Analyses

| SA | Description |
|----|-------------|
| SA1 | Restrict symptom inclusion to codes 1–3 (at least monthly; drops "a few times a year") |
| SA2 | Year-stratified regression — separate model per survey year using single-year weights |
| SA3 | SGM binary — collapse Gay/Lesbian + Bisexual + Queer or Questioning into a single "SGM" category |
| SA4 | Missing indicator — replace NA covariates with "Missing" category instead of listwise deletion |
| SA5 | Unweighted regression — standard `glm()` without survey weights or design |
| SA6 | Alternative race/ethnicity — use HISPALLP_A (NCHS pre-built combined variable) in place of manual HISP_A + RACEALLP_A combination |

## AI Assistance

Code in this repository was drafted and iteratively developed with assistance from **Claude** (Anthropic) and **Codex** (OpenAI). Both tools were used to write, read, and validate the analysis scripts — including survey design setup, variable recoding logic, and rpy2 integration. All code was subsequently reviewed and validated by the lead author (N. Singh-Dias) prior to use.

## Authors

Noel Dias, working with Dr. Alyssa Falise
*(formerly published as Noel Singh-Dias)*

## License

This project is for academic research purposes. NHIS public use data are freely available from the National Center for Health Statistics (NCHS). See NCHS data use restrictions for terms of use.
