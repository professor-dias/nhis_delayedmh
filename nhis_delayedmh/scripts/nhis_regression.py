"""
Study: Among people experiencing mental health symptoms, who is unable to access care
       because of the cost? Findings from NHIS: 2020–2024

Step 6: Weighted logistic regression
        Outcome: Delayed mental health care due to cost (mh_delay)
        Family:  quasibinomial(logit) via svyglm — appropriate for binary outcome
                 with complex survey design
        Output:  Odds Ratios, 95% CI, p-values — APA style table
"""

import os
os.environ["RPY2_CFFI_MODE"] = "ABI"

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = Path(os.getenv("NHIS_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_FILE   = DATA_DIR / "nhis_analytic_2020_2024.csv"
OUTPUT_CSV   = OUTPUT_DIR / "table2_regression.csv"

# ─── Load data and pass to R ───────────────────────────────────────────────────
print("Loading analytic file...")
nhis = pd.read_csv(INPUT_FILE)

importr("survey")

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["nhis"] = nhis

# ─── Survey design + regression entirely in R ─────────────────────────────────
ro.r("""
library(survey)

options(survey.lonely.psu = "adjust")

svy <- svydesign(
    id      = ~PPSU_POOL,
    strata  = ~PSTRAT_POOL,
    weights = ~WTFA_5YR,
    data    = nhis,
    nest    = TRUE
)

# Restrict to complete cases on all model variables
svy_model <- subset(svy,
    !is.na(mh_delay)    &
    !is.na(age_group)   &
    !is.na(sex)         &
    !is.na(race_eth)    &
    !is.na(education)   &
    !is.na(orientation) &
    !is.na(insured)
)

# ── Set factor levels and reference categories ────────────────────────────────
d <- svy_model$variables

d$mh_delay    <- factor(d$mh_delay,    levels = c("No", "Yes"))
d$age_group   <- factor(d$age_group,   levels = c("18–34", "35–49",
                                                   "50–64", "65+"))
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
d$insured     <- factor(d$insured,     levels = c("Insured", "Uninsured"))
d$year        <- factor(d$year,        levels = c(2020, 2021, 2022, 2023, 2024))

svy_model$variables <- d

# ── Fit weighted logistic regression ─────────────────────────────────────────
model <- svyglm(
    mh_delay == "Yes" ~
        age_group + sex + race_eth + education + orientation + insured + year,
    design = svy_model,
    family = quasibinomial(link = "logit")
)

# ── Extract ORs, 95% CIs, p-values ───────────────────────────────────────────
coefs   <- summary(model)$coefficients
OR      <- exp(coefs[, "Estimate"])
CI_low  <- exp(coefs[, "Estimate"] - 1.96 * coefs[, "Std. Error"])
CI_high <- exp(coefs[, "Estimate"] + 1.96 * coefs[, "Std. Error"])
p_vals  <- coefs[, "Pr(>|t|)"]

# ── APA p-value formatter ─────────────────────────────────────────────────────
fmt_p <- function(p) {
    if (is.na(p))  return("")
    if (p < .001)  return("< .001")
    return(formatC(p, digits = 3, format = "f"))
}

# ── Build clean label map ─────────────────────────────────────────────────────
label_map <- c(
    "(Intercept)"                              = "(Intercept)",
    "age_group35–49"                      = "Age: 35–49 (ref: 18–34)",
    "age_group50–64"                      = "Age: 50–64",
    "age_group65+"                             = "Age: 65+",
    "sexFemale"                                = "Sex: Female (ref: Male)",
    "race_ethHispanic"                         = "Race/Ethnicity: Hispanic (ref: NH White)",
    "race_ethNH Black"                         = "Race/Ethnicity: NH Black",
    "race_ethNH Asian"                         = "Race/Ethnicity: NH Asian",
    "race_ethNH AIAN"                          = "Race/Ethnicity: NH AIAN",
    "race_ethNH Other/Multiracial"             = "Race/Ethnicity: NH Other/Multiracial",
    "educationLess than High School"           = "Education: Less than HS (ref: HS/GED)",
    "educationSome College / Associate"        = "Education: Some College / Associate",
    "educationBachelor's Degree"               = "Education: Bachelor's Degree",
    "educationGraduate Degree"                 = "Education: Graduate Degree",
    "orientationGay or Lesbian"                = "Orientation: Gay or Lesbian (ref: Straight)",
    "orientationBisexual"                      = "Orientation: Bisexual",
    "orientationQueer or Questioning"          = "Orientation: Queer or Questioning",
    "insuredUninsured"                         = "Insurance: Uninsured (ref: Insured)",
    "year2021"                                 = "Year: 2021 (ref: 2020)",
    "year2022"                                 = "Year: 2022",
    "year2023"                                 = "Year: 2023",
    "year2024"                                 = "Year: 2024"
)

row_names <- rownames(coefs)
labels    <- ifelse(row_names %in% names(label_map), label_map[row_names], row_names)

# ── Assemble results table ────────────────────────────────────────────────────
results <- data.frame(
    Variable = labels,
    OR       = round(OR,      2),
    CI_lower = round(CI_low,  2),
    CI_upper = round(CI_high, 2),
    p        = sapply(p_vals, fmt_p),
    row.names = NULL,
    stringsAsFactors = FALSE
)

# Format CI as [lower, upper] string for display
results$"95% CI" <- paste0("[", results$CI_lower, ", ", results$CI_upper, "]")
results <- results[, c("Variable", "OR", "95% CI", "p")]

# Remove intercept from display table
results <- results[results$Variable != "(Intercept)", ]

# Model fit info
n_obs <- nrow(svy_model$variables)
""")

# ─── Pull to Python and save ───────────────────────────────────────────────────
with (ro.default_converter + pandas2ri.converter).context():
    results = ro.globalenv["results"]
    n_obs   = int(ro.globalenv["n_obs"][0])

# APA title and note
print("\n" + "=" * 75)
print("TABLE 2")
print("Weighted Logistic Regression: Predictors of Delayed Mental Health Care")
print("Due to Cost, NHIS 2020–2024")
print(f"n = {n_obs:,} (complete cases)")
print("=" * 75)
print(results.to_string(index=False))
print()
print("Note. OR = odds ratio; CI = confidence interval.")
print("Reference categories: 18–34 (age); Male (sex); NH White (race/ethnicity);")
print("High School/GED (education); Straight (orientation);")
print("Insured (insurance); 2020 (year).")
print("Estimates weighted using 5-year pooled survey weights (WTFA_5YR).")
print("Survey design accounts for stratification and clustering.")

# Save
title_row    = pd.DataFrame([["Table 2", "", "", ""]], columns=results.columns)
subtitle_row = pd.DataFrame([[
    "Weighted Logistic Regression: Predictors of Delayed Mental Health Care Due to Cost, NHIS 2020-2024",
    "", "", ""]], columns=results.columns)
n_row        = pd.DataFrame([[f"n = {n_obs:,} (complete cases on all model variables)",
                               "", "", ""]], columns=results.columns)
blank        = pd.DataFrame([["", "", "", ""]], columns=results.columns)
note_row     = pd.DataFrame([[
    "Note. OR = odds ratio; CI = 95% confidence interval. Reference categories: "
    "Male (sex); NH White (race/ethnicity); High School/GED (education); "
    "Straight (orientation); Insured (insurance); 2020 (year). "
    "Estimates weighted using 5-year pooled survey weights. "
    "Survey design accounts for stratification and clustering (svyglm, quasibinomial).",
    "", "", ""]], columns=results.columns)

output = pd.concat([title_row, subtitle_row, n_row, blank,
                    results, blank, note_row], ignore_index=True)
output.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved: {OUTPUT_CSV.name}")
