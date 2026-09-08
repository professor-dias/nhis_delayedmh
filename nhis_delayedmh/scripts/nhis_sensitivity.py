"""
Study: Among people experiencing mental health symptoms, who is unable to access care
       because of the cost? Findings from NHIS: 2020–2024

nhis_sensitivity.py — Six sensitivity analyses for the main logistic regression

SA1: Symptom frequency threshold  — restrict inclusion to codes 1–3 (≥ monthly)
SA2: Year-stratified regression   — separate svyglm per survey year (2020–2024)
SA3: SGM binary orientation       — collapse Gay/Lesbian + Bisexual + Queer → "SGM"
SA4: Missing indicator            — replace NA covariates with "Missing" category
SA5: Unweighted regression        — standard glm(), no survey weights or design
SA6: Alternative race/ethnicity   — HISPALLP_A (NCHS pre-built variable)
"""

import os
os.environ["RPY2_CFFI_MODE"] = "ABI"

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = Path(os.getenv("NHIS_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR    = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_FILE    = DATA_DIR / "nhis_analytic_2020_2024.csv"

print("Loading analytic file...")
nhis = pd.read_csv(INPUT_FILE)
print(f"  Full analytic sample: {len(nhis):,} rows\n")

importr("survey")

# ─── Python recoding helpers (used in SA6 raw-file reload) ────────────────────
def recode_age_group(age):
    if pd.isna(age):  return None
    if age < 35:      return "18–34"
    if age < 50:      return "35–49"
    if age < 65:      return "50–64"
    return                   "65+"

def recode_education(code):
    if pd.isna(code) or code in [97, 98, 99]: return None
    code = int(code)
    if code in [0, 1, 2]:  return "Less than High School"
    if code in [3, 4]:     return "High School / GED"
    if code in [5, 6, 7]:  return "Some College / Associate"
    if code == 8:          return "Bachelor's Degree"
    if code in [9, 10]:    return "Graduate Degree"
    return None

# ─── Label map (R coefficient names → APA display labels) ────────────────────
LABEL_MAP = {
    "age_group35–49":                  "Age: 35–49 (ref: 18–34)",
    "age_group50–64":                  "Age: 50–64",
    "age_group65+":                         "Age: 65+",
    "sexFemale":                            "Sex: Female (ref: Male)",
    "race_ethHispanic":                     "Race: Hispanic (ref: NH White)",
    "race_ethNH Black":                     "Race: NH Black",
    "race_ethNH Asian":                     "Race: NH Asian",
    "race_ethNH AIAN":                      "Race: NH AIAN",
    "race_ethNH Other/Multiracial":         "Race: NH Other/Multiracial",
    "educationLess than High School":       "Education: < HS (ref: HS/GED)",
    "educationSome College / Associate":    "Education: Some College / Assoc.",
    "educationBachelor's Degree":           "Education: Bachelor's",
    "educationGraduate Degree":             "Education: Graduate",
    "orientationGay or Lesbian":            "Orientation: Gay/Lesbian (ref: Straight)",
    "orientationBisexual":                  "Orientation: Bisexual",
    "orientationQueer or Questioning":      "Orientation: Queer/Questioning",
    "orientation_binSGM":                   "Orientation: SGM (ref: Straight)",
    "insuredUninsured":                     "Insurance: Uninsured (ref: Insured)",
    "year2021":                             "Year: 2021 (ref: 2020)",
    "year2022":                             "Year: 2022",
    "year2023":                             "Year: 2023",
    "year2024":                             "Year: 2024",
}

LABEL_MAP_MISSING = {
    **LABEL_MAP,
    "age_groupMissing":    "Age: Missing",
    "sexMissing":          "Sex: Missing",
    "race_ethMissing":     "Race: Missing",
    "educationMissing":    "Education: Missing",
    "orientationMissing":  "Orientation: Missing",
    "insuredMissing":      "Insurance: Missing",
}

def label_results(df, lmap=None):
    if lmap is None:
        lmap = LABEL_MAP
    df = df.copy()
    df["Variable"] = df["Variable"].map(lambda x: lmap.get(x, x))
    return df

def print_sa(title, df):
    print("\n" + "─" * 75)
    print(title)
    print("─" * 75)
    print(df.to_string(index=False))

def save_sa(df, path, title, n):
    cols  = df.columns.tolist()
    blank = pd.DataFrame([[""] * len(cols)], columns=cols)
    out   = pd.concat([
        pd.DataFrame([[title]        + [""] * (len(cols) - 1)], columns=cols),
        pd.DataFrame([[f"n = {n:,}"] + [""] * (len(cols) - 1)], columns=cols),
        blank, df, blank,
    ], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"  ✓ Saved: {Path(path).name}")

# ─── Shared R helpers ─────────────────────────────────────────────────────────
ro.r("""
library(survey)

options(survey.lonely.psu = "adjust")

fmt_p <- function(p) {
    if (is.na(p)) return("")
    if (p < .001) return("< .001")
    return(formatC(p, digits = 3, format = "f"))
}

extract_results <- function(model) {
    coefs   <- summary(model)$coefficients
    p_col   <- if ("Pr(>|t|)" %in% colnames(coefs)) "Pr(>|t|)" else "Pr(>|z|)"
    OR      <- exp(coefs[, "Estimate"])
    CI_low  <- exp(coefs[, "Estimate"] - 1.96 * coefs[, "Std. Error"])
    CI_high <- exp(coefs[, "Estimate"] + 1.96 * coefs[, "Std. Error"])
    p_vals  <- coefs[, p_col]
    results <- data.frame(
        Variable = rownames(coefs),
        OR       = round(OR,      2),
        CI_lower = round(CI_low,  2),
        CI_upper = round(CI_high, 2),
        p        = sapply(p_vals, fmt_p),
        row.names = NULL,
        stringsAsFactors = FALSE
    )
    results$CI <- paste0("[", results$CI_lower, ", ", results$CI_upper, "]")
    results[results$Variable != "(Intercept)", c("Variable", "OR", "CI", "p")]
}

set_standard_factors <- function(d) {
    d$mh_delay    <- factor(d$mh_delay,    levels = c("No", "Yes"))
    d$age_group   <- factor(d$age_group,   levels = c("18–34", "35–49", "50–64", "65+"))
    d$sex         <- factor(d$sex,         levels = c("Male", "Female"))
    d$race_eth    <- factor(d$race_eth,
                            levels = c("NH White", "Hispanic", "NH Black",
                                       "NH Asian", "NH AIAN", "NH Other/Multiracial"))
    d$education   <- factor(d$education,
                            levels = c("High School / GED", "Less than High School",
                                       "Some College / Associate",
                                       "Bachelor's Degree", "Graduate Degree"))
    d$orientation <- factor(d$orientation,
                            levels = c("Straight", "Gay or Lesbian",
                                       "Bisexual", "Queer or Questioning"))
    d$insured     <- factor(d$insured, levels = c("Insured", "Uninsured"))
    d
}
""")

MAIN_FORMULA = (
    'mh_delay == "Yes" ~ age_group + sex + race_eth + education + '
    'orientation + insured + year'
)
YEAR_FORMULA = (
    'mh_delay == "Yes" ~ age_group + sex + race_eth + education + '
    'orientation + insured'
)

# ══════════════════════════════════════════════════════════════════════════════
# SA1: Symptom Frequency Threshold (codes 1–3 only; drop "a few times a year")
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("SA1: Symptom Frequency Threshold — codes 1–3 only (at least monthly)")
print("=" * 75)

sa1 = nhis[
    nhis["DEPFREQ_A"].isin([1, 2, 3]) | nhis["ANXFREQ_A"].isin([1, 2, 3])
].copy()
print(f"  Main sample (codes 1–4): {len(nhis):,}")
print(f"  SA1  sample (codes 1–3): {len(sa1):,}  (dropped: {len(nhis) - len(sa1):,})")

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["sa1"] = sa1

ro.r(f"""
options(survey.lonely.psu = "adjust")
svy_sa1 <- svydesign(id=~PPSU_POOL, strata=~PSTRAT_POOL, weights=~WTFA_5YR,
                     data=sa1, nest=TRUE)
svy_sa1 <- subset(svy_sa1,
    !is.na(mh_delay) & !is.na(age_group) & !is.na(sex) & !is.na(race_eth) &
    !is.na(education) & !is.na(orientation) & !is.na(insured))

d <- set_standard_factors(svy_sa1$variables)
d$year <- factor(d$year, levels=c(2020,2021,2022,2023,2024))
svy_sa1$variables <- d

n_sa1     <- nrow(svy_sa1$variables)
model_sa1 <- svyglm({repr(MAIN_FORMULA)}, design=svy_sa1, family=quasibinomial(link="logit"))
res_sa1   <- extract_results(model_sa1)
""")

with (ro.default_converter + pandas2ri.converter).context():
    res_sa1 = ro.globalenv["res_sa1"]
    n_sa1   = int(ro.globalenv["n_sa1"][0])

res_sa1 = label_results(res_sa1)
print_sa(f"SA1: Symptom Threshold — Codes 1–3 Only (n = {n_sa1:,})", res_sa1)
save_sa(res_sa1, OUTPUT_DIR / "sa1_symptom_threshold.csv",
        "SA1: Symptom Threshold (codes 1–3 only — at least monthly symptoms)", n_sa1)


# ══════════════════════════════════════════════════════════════════════════════
# SA2: Year-Stratified Regression (separate svyglm per year; no year covariate)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("SA2: Year-Stratified Regression (separate model per year)")
print("=" * 75)

all_yr_results = []

for year in [2020, 2021, 2022, 2023, 2024]:
    yr_df = nhis[nhis["year"] == year].copy()

    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["yr_df"] = yr_df

    ro.r(f"""
    options(survey.lonely.psu = "adjust")
    svy_yr <- svydesign(id=~PPSU, strata=~PSTRAT, weights=~WTFA_A,
                        data=yr_df, nest=TRUE)
    svy_yr <- subset(svy_yr,
        !is.na(mh_delay) & !is.na(age_group) & !is.na(sex) & !is.na(race_eth) &
        !is.na(education) & !is.na(orientation) & !is.na(insured))

    d <- set_standard_factors(svy_yr$variables)
    svy_yr$variables <- d

    n_yr <- nrow(svy_yr$variables)

    model_yr <- tryCatch(
        svyglm({repr(YEAR_FORMULA)}, design=svy_yr, family=quasibinomial(link="logit")),
        error = function(e) {{ message("ERROR: ", conditionMessage(e)); NULL }}
    )
    res_yr <- if (is.null(model_yr)) {{
        data.frame(Variable="MODEL FAILED", OR=NA_real_, CI=NA_character_, p=NA_character_,
                   stringsAsFactors=FALSE)
    }} else {{
        extract_results(model_yr)
    }}
    """)

    with (ro.default_converter + pandas2ri.converter).context():
        res_yr = ro.globalenv["res_yr"]
        n_yr   = int(ro.globalenv["n_yr"][0])

    res_yr = label_results(res_yr)
    res_yr.insert(0, "Year", str(year))
    res_yr.insert(1, "n",    n_yr)
    all_yr_results.append(res_yr)
    print(f"\n  {year} (n = {n_yr:,})")
    print(res_yr[["Variable", "OR", "CI", "p"]].to_string(index=False))

sa2_combined = pd.concat(all_yr_results, ignore_index=True)
sa2_path = OUTPUT_DIR / "sa2_year_stratified.csv"
sa2_combined.to_csv(sa2_path, index=False)
print(f"\n  ✓ Saved: {sa2_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# SA3: SGM Binary Orientation (Straight vs. any SGM)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("SA3: SGM Binary Orientation (Straight vs. any SGM)")
print("=" * 75)

sa3 = nhis.copy()
sa3["orientation_bin"] = sa3["orientation"].map({
    "Straight":               "Straight",
    "Gay or Lesbian":         "SGM",
    "Bisexual":               "SGM",
    "Queer or Questioning":   "SGM",
})

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["sa3"] = sa3

BINARY_FORMULA = (
    'mh_delay == "Yes" ~ age_group + sex + race_eth + education + '
    'orientation_bin + insured + year'
)

ro.r(f"""
options(survey.lonely.psu = "adjust")
svy_sa3 <- svydesign(id=~PPSU_POOL, strata=~PSTRAT_POOL, weights=~WTFA_5YR,
                     data=sa3, nest=TRUE)
svy_sa3 <- subset(svy_sa3,
    !is.na(mh_delay) & !is.na(age_group) & !is.na(sex) & !is.na(race_eth) &
    !is.na(education) & !is.na(orientation_bin) & !is.na(insured))

d <- set_standard_factors(svy_sa3$variables)
d$orientation_bin <- factor(d$orientation_bin, levels=c("Straight","SGM"))
d$year            <- factor(d$year, levels=c(2020,2021,2022,2023,2024))
svy_sa3$variables <- d

n_sa3     <- nrow(svy_sa3$variables)
model_sa3 <- svyglm({repr(BINARY_FORMULA)}, design=svy_sa3, family=quasibinomial(link="logit"))
res_sa3   <- extract_results(model_sa3)
""")

with (ro.default_converter + pandas2ri.converter).context():
    res_sa3 = ro.globalenv["res_sa3"]
    n_sa3   = int(ro.globalenv["n_sa3"][0])

res_sa3 = label_results(res_sa3)
print_sa(f"SA3: SGM Binary — Straight vs. Any SGM (n = {n_sa3:,})", res_sa3)
save_sa(res_sa3, OUTPUT_DIR / "sa3_sgm_binary.csv",
        "SA3: SGM Binary Orientation (Straight vs. SGM collapsed)", n_sa3)


# ══════════════════════════════════════════════════════════════════════════════
# SA4: Missing Indicator (NA covariates → "Missing" category; no listwise drop)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("SA4: Missing Indicator Approach")
print("=" * 75)

sa4 = nhis.copy()
for col in ["age_group", "sex", "race_eth", "education", "orientation", "insured"]:
    sa4[col] = sa4[col].fillna("Missing")

cc_n = nhis.dropna(subset=["mh_delay","age_group","sex","race_eth",
                             "education","orientation","insured"]).shape[0]
mi_n = sa4[sa4["mh_delay"].notna()].shape[0]
print(f"  Complete-case model n:       {cc_n:,}")
print(f"  Missing-indicator model n:   {mi_n:,}  (gain: {mi_n - cc_n:,})")

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["sa4"] = sa4

ro.r(f"""
options(survey.lonely.psu = "adjust")
svy_sa4 <- svydesign(id=~PPSU_POOL, strata=~PSTRAT_POOL, weights=~WTFA_5YR,
                     data=sa4, nest=TRUE)
svy_sa4 <- subset(svy_sa4, !is.na(mh_delay))

d <- svy_sa4$variables
d$mh_delay    <- factor(d$mh_delay,    levels=c("No","Yes"))
d$age_group   <- factor(d$age_group,   levels=c("18–34","35–49","50–64","65+","Missing"))
d$sex         <- factor(d$sex,         levels=c("Male","Female","Missing"))
d$race_eth    <- factor(d$race_eth,
                        levels=c("NH White","Hispanic","NH Black","NH Asian",
                                 "NH AIAN","NH Other/Multiracial","Missing"))
d$education   <- factor(d$education,
                        levels=c("High School / GED","Less than High School",
                                 "Some College / Associate","Bachelor's Degree",
                                 "Graduate Degree","Missing"))
d$orientation <- factor(d$orientation,
                        levels=c("Straight","Gay or Lesbian","Bisexual",
                                 "Queer or Questioning","Missing"))
d$insured     <- factor(d$insured, levels=c("Insured","Uninsured","Missing"))
d$year        <- factor(d$year,    levels=c(2020,2021,2022,2023,2024))
svy_sa4$variables <- d

n_sa4     <- nrow(svy_sa4$variables)
model_sa4 <- svyglm({repr(MAIN_FORMULA)}, design=svy_sa4, family=quasibinomial(link="logit"))
res_sa4   <- extract_results(model_sa4)
""")

with (ro.default_converter + pandas2ri.converter).context():
    res_sa4 = ro.globalenv["res_sa4"]
    n_sa4   = int(ro.globalenv["n_sa4"][0])

res_sa4 = label_results(res_sa4, lmap=LABEL_MAP_MISSING)
print_sa(f"SA4: Missing Indicator (n = {n_sa4:,})", res_sa4)
save_sa(res_sa4, OUTPUT_DIR / "sa4_missing_indicator.csv",
        "SA4: Missing Indicator (NA covariates coded as 'Missing' category)", n_sa4)


# ══════════════════════════════════════════════════════════════════════════════
# SA5: Unweighted Regression (standard glm; no survey weights or design)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("SA5: Unweighted Regression (glm, no survey design)")
print("=" * 75)

sa5 = nhis.dropna(subset=["mh_delay","age_group","sex","race_eth",
                            "education","orientation","insured"]).copy()

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["sa5"] = sa5

ro.r(f"""
d5 <- set_standard_factors(sa5)
d5$year <- factor(d5$year, levels=c(2020,2021,2022,2023,2024))

n_sa5     <- nrow(d5)
model_sa5 <- glm({repr(MAIN_FORMULA)}, data=d5, family=binomial(link="logit"))
res_sa5   <- extract_results(model_sa5)
""")

with (ro.default_converter + pandas2ri.converter).context():
    res_sa5 = ro.globalenv["res_sa5"]
    n_sa5   = int(ro.globalenv["n_sa5"][0])

res_sa5 = label_results(res_sa5)
print_sa(f"SA5: Unweighted Regression (n = {n_sa5:,})", res_sa5)
save_sa(res_sa5, OUTPUT_DIR / "sa5_unweighted.csv",
        "SA5: Unweighted Regression (glm, no survey weights or design)", n_sa5)


# ══════════════════════════════════════════════════════════════════════════════
# SA6: Alternative Race/Ethnicity — HISPALLP_A (NCHS pre-built variable)
# Raw files reloaded to access HISPALLP_A, which is not in the analytic CSV.
# HISPALLP_A codes: 1=Hispanic, 2=NH White, 3=NH Black, 4=NH Asian,
#                   5=NH AIAN only, 6=NH AIAN + other group,
#                   7=Other single and multiple races, 97/98/99=missing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("SA6: Alternative Race/Ethnicity — HISPALLP_A (NCHS pre-built variable)")
print("=" * 75)

VARS_SA6 = [
    "DEPFREQ_A", "ANXFREQ_A", "MHTHDLY_A", "AGEP_A", "SEX_A",
    "HISPALLP_A", "EDUCP_A", "ORIENT_A", "NOTCOV_A", "WTFA_A", "PSTRAT", "PPSU",
]

HISPALLP_MAP = {
    1: "Hispanic",
    2: "NH White",
    3: "NH Black",
    4: "NH Asian",
    5: "NH AIAN",
    6: "NH Other/Multiracial",   # NH AIAN and any other group
    7: "NH Other/Multiracial",   # other single and multiple races
}

year_frames_sa6 = []
for year in [2020, 2021, 2022, 2023, 2024]:
    yy       = str(year)[2:]
    filepath = DATA_DIR / f"adult{yy}csv" / f"adult{yy}.csv"
    df       = pd.read_csv(filepath, low_memory=False)

    if year == 2020:
        df = df.rename(columns={"EDUC_A": "EDUCP_A"})

    missing_cols = [v for v in VARS_SA6 if v not in df.columns]
    if missing_cols:
        print(f"  WARNING: {year} is missing {missing_cols} — skipping this year")
        continue

    df = df[VARS_SA6].copy()
    df["year"] = year
    year_frames_sa6.append(df)
    print(f"  {year}: {len(df):,} rows loaded")

sa6_raw = pd.concat(year_frames_sa6, ignore_index=True)

sa6 = sa6_raw[
    sa6_raw["DEPFREQ_A"].isin([1, 2, 3, 4]) | sa6_raw["ANXFREQ_A"].isin([1, 2, 3, 4])
].copy()
print(f"\n  SA6 symptomatic sample: {len(sa6):,}")

print("\n  HISPALLP_A value counts (pooled):")
print(sa6["HISPALLP_A"].value_counts().sort_index().to_string())

sa6["WTFA_5YR"]    = sa6["WTFA_A"] / 5
sa6["PSTRAT_POOL"] = sa6["year"].astype(str) + "_" + sa6["PSTRAT"].astype(str)
sa6["PPSU_POOL"]   = (sa6["year"].astype(str) + "_" +
                      sa6["PSTRAT"].astype(str) + "_" +
                      sa6["PPSU"].astype(str))

sa6["mh_delay"]    = sa6["MHTHDLY_A"].map({1: "Yes", 2: "No"})
sa6["age"]         = sa6["AGEP_A"].where(~sa6["AGEP_A"].isin([97, 98, 99]))
sa6["age_group"]   = sa6["age"].map(recode_age_group)
sa6["sex"]         = sa6["SEX_A"].map({1: "Male", 2: "Female"})
sa6["race_eth"]    = sa6["HISPALLP_A"].map(HISPALLP_MAP)
sa6["EDUCP_A"]     = sa6["EDUCP_A"].replace(11, 10)
sa6["education"]   = sa6["EDUCP_A"].map(recode_education)
sa6["orientation"] = sa6["ORIENT_A"].map({
    1: "Gay or Lesbian",
    2: "Straight",
    3: "Bisexual",
    4: "Queer or Questioning",
    5: "Queer or Questioning",
})
sa6["insured"]     = sa6["NOTCOV_A"].map({1: "Uninsured", 2: "Insured"})

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["sa6"] = sa6

ro.r(f"""
options(survey.lonely.psu = "adjust")
svy_sa6 <- svydesign(id=~PPSU_POOL, strata=~PSTRAT_POOL, weights=~WTFA_5YR,
                     data=sa6, nest=TRUE)
svy_sa6 <- subset(svy_sa6,
    !is.na(mh_delay) & !is.na(age_group) & !is.na(sex) & !is.na(race_eth) &
    !is.na(education) & !is.na(orientation) & !is.na(insured))

d <- set_standard_factors(svy_sa6$variables)
d$year <- factor(d$year, levels=c(2020,2021,2022,2023,2024))
svy_sa6$variables <- d

n_sa6     <- nrow(svy_sa6$variables)
model_sa6 <- svyglm({repr(MAIN_FORMULA)}, design=svy_sa6, family=quasibinomial(link="logit"))
res_sa6   <- extract_results(model_sa6)
""")

with (ro.default_converter + pandas2ri.converter).context():
    res_sa6 = ro.globalenv["res_sa6"]
    n_sa6   = int(ro.globalenv["n_sa6"][0])

res_sa6 = label_results(res_sa6)
print_sa(f"SA6: Alternative Race/Ethnicity — HISPALLP_A (n = {n_sa6:,})", res_sa6)
save_sa(res_sa6, OUTPUT_DIR / "sa6_alt_race.csv",
        "SA6: Alternative Race/Ethnicity Variable (HISPALLP_A — NCHS pre-built)", n_sa6)


# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("All 6 sensitivity analyses complete.")
print("Output CSVs written to: output/")
print("  sa1_symptom_threshold.csv  — SA1: codes 1–3 only")
print("  sa2_year_stratified.csv    — SA2: separate model per year")
print("  sa3_sgm_binary.csv         — SA3: Straight vs. SGM binary")
print("  sa4_missing_indicator.csv  — SA4: missing category approach")
print("  sa5_unweighted.csv         — SA5: glm without survey design")
print("  sa6_alt_race.csv           — SA6: HISPALLP_A race variable")
print("=" * 75)
