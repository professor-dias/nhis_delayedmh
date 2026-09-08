"""
Study: Among people experiencing mental health symptoms, who is unable to access care
       because of the cost? Findings from NHIS: 2020–2024

Step 5: Table 1 — Weighted Sample Characteristics by Delayed Mental Health Care Due to Cost
        APA style, complex survey-weighted estimates via R survey package
"""

import os
os.environ["RPY2_CFFI_MODE"] = "ABI"  # must be set before any rpy2 import

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
OUTPUT_CSV   = OUTPUT_DIR / "table1.csv"

# ─── Load data and pass to R ───────────────────────────────────────────────────
print("Loading analytic file...")
nhis = pd.read_csv(INPUT_FILE)

survey = importr("survey")

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["nhis"] = nhis

# ─── Build survey design and compute Table 1 entirely in R ────────────────────
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

# Restrict to non-missing outcome
svy_clean <- subset(svy, !is.na(mh_delay))

# Set factor levels so table rows appear in a logical order
svy_clean$variables$mh_delay    <- factor(svy_clean$variables$mh_delay,
                                          levels = c("Yes", "No"))
svy_clean$variables$age_group   <- factor(svy_clean$variables$age_group,
                                          levels = c("18–34", "35–49",
                                                     "50–64", "65+"))
svy_clean$variables$sex         <- factor(svy_clean$variables$sex,
                                          levels = c("Male", "Female"))
svy_clean$variables$race_eth    <- factor(svy_clean$variables$race_eth,
                                          levels = c("NH White", "Hispanic", "NH Black",
                                                     "NH Asian", "NH AIAN",
                                                     "NH Other/Multiracial"))
svy_clean$variables$education   <- factor(svy_clean$variables$education,
                                          levels = c("Less than High School",
                                                     "High School / GED",
                                                     "Some College / Associate",
                                                     "Bachelor's Degree",
                                                     "Graduate Degree"))
svy_clean$variables$orientation <- factor(svy_clean$variables$orientation,
                                          levels = c("Straight", "Gay or Lesbian",
                                                     "Bisexual", "Queer or Questioning"))
svy_clean$variables$insured     <- factor(svy_clean$variables$insured,
                                          levels = c("Insured", "Uninsured"))

# ── Unweighted sample sizes for column headers ─────────────────────────────
total_n   <- nrow(svy_clean$variables)
delayed_n <- sum(svy_clean$variables$mh_delay == "Yes", na.rm = TRUE)
notdel_n  <- sum(svy_clean$variables$mh_delay == "No",  na.rm = TRUE)

# ── APA p-value formatter ──────────────────────────────────────────────────
fmt_p <- function(p) {
    if (is.na(p))   return("")
    if (p < .001)   return("< .001")
    if (p < .01)    return(paste0(formatC(p, digits=3, format="f")))
    return(formatC(p, digits=3, format="f"))
}

# ── Row builders ───────────────────────────────────────────────────────────
rows <- list()

header_row <- function(label, p_val = NA) {
    rows[[length(rows) + 1]] <<- data.frame(
        Variable    = label,
        Overall     = "",
        Not_Delayed = "",
        Delayed     = "",
        p           = fmt_p(p_val),
        stringsAsFactors = FALSE
    )
}

data_row <- function(label, ov, nd, dl) {
    rows[[length(rows) + 1]] <<- data.frame(
        Variable    = paste0("  ", label),
        Overall     = ov,
        Not_Delayed = nd,
        Delayed     = dl,
        p           = "",
        stringsAsFactors = FALSE
    )
}

# ── Categorical variable helper ────────────────────────────────────────────
# Reports: unweighted n + weighted % (overall); weighted % by outcome group
cat_rows <- function(var, label) {
    f_ov   <- as.formula(paste0("~", var))
    f_cr   <- as.formula(paste0("~", var, " + mh_delay"))

    tab_ov   <- svytable(f_ov, svy_clean)
    tab_cr   <- svytable(f_cr, svy_clean)
    prop_ov  <- prop.table(tab_ov) * 100
    prop_cr  <- prop.table(tab_cr, margin = 2) * 100  # column %

    test_p   <- tryCatch(
        svychisq(f_cr, svy_clean, statistic = "Chisq")$p.value,
        error = function(e) NA
    )

    # Unweighted n per level (for overall column)
    uw_counts <- table(svy_clean$variables[[var]], useNA = "no")

    header_row(paste0(label, ", n (%)"), test_p)

    for (lvl in levels(svy_clean$variables[[var]])) {
        uw_n   <- if (lvl %in% names(uw_counts)) as.integer(uw_counts[[lvl]]) else 0L
        ov_pct <- if (lvl %in% names(prop_ov))   round(as.numeric(prop_ov[[lvl]]), 1) else 0

        dl_pct <- if (lvl %in% rownames(prop_cr) && "Yes" %in% colnames(prop_cr))
                      round(as.numeric(prop_cr[lvl, "Yes"]), 1) else 0
        nd_pct <- if (lvl %in% rownames(prop_cr) && "No"  %in% colnames(prop_cr))
                      round(as.numeric(prop_cr[lvl, "No"]),  1) else 0

        data_row(
            lvl,
            sprintf("%s (%.1f%%)", format(uw_n, big.mark = ","), ov_pct),
            sprintf("%.1f%%", nd_pct),
            sprintf("%.1f%%", dl_pct)
        )
    }
}

cat_rows("age_group",   "Age Group")
cat_rows("sex",         "Sex")
cat_rows("race_eth",    "Race/Ethnicity")
cat_rows("education",   "Education")
cat_rows("orientation", "Sexual Orientation")
cat_rows("insured",     "Insurance Status")

# ── Assemble final table ───────────────────────────────────────────────────
table1 <- do.call(rbind, rows)

colnames(table1) <- c(
    "Variable",
    sprintf("Overall (n = %s)",     format(total_n,   big.mark = ",")),
    sprintf("Not Delayed (n = %s)", format(notdel_n,  big.mark = ",")),
    sprintf("Delayed (n = %s)",     format(delayed_n, big.mark = ",")),
    "p"
)
""")

# ─── Pull table to Python and save ────────────────────────────────────────────
with (ro.default_converter + pandas2ri.converter).context():
    table1 = ro.globalenv["table1"]

# Add APA title and note as flanking rows
title_row = pd.DataFrame([["Table 1", "", "", "", ""] +
                           [""] * (len(table1.columns) - 5)],
                          columns=table1.columns)

note_row = pd.DataFrame([[
    "Note. Estimates are weighted to represent the U.S. adult population. "
    "Unweighted n reported with weighted % for categorical variables. "
    "p-values from weighted chi-square tests (categorical variables). "
    "NHIS = National Health Interview Survey.",
    "", "", "", ""]], columns=table1.columns)

subtitle_row = pd.DataFrame([[
    "Weighted Sample Characteristics by Delayed Mental Health Care Due to Cost, NHIS 2020–2024",
    "", "", "", ""]], columns=table1.columns)

blank = pd.DataFrame([["", "", "", "", ""]], columns=table1.columns)

output = pd.concat([title_row, subtitle_row, blank, table1, blank, note_row],
                   ignore_index=True)

output.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("TABLE 1")
print("Weighted Sample Characteristics by Delayed Mental Health Care Due to Cost")
print("NHIS 2020–2024")
print("=" * 80)
print(table1.to_string(index=False))
print("\nNote. Unweighted n, weighted % for all variables.")
print("p-values from weighted chi-square tests (svychisq).")
print(f"\n✓ Saved: {OUTPUT_CSV.name}")
