"""
Study: Among people experiencing mental health symptoms, who is unable to access care
       because of the cost? Findings from NHIS: 2020–2024

Step 1: Build pooled analytic dataset (all 5 years, 12 target variables)
Step 2: Apply inclusion filter — keep only adults reporting symptoms
Step 3: Adjust survey weights for pooling + set up complex survey design via rpy2
Step 4: Recode all variables into labeled categories
"""

import os
os.environ["RPY2_CFFI_MODE"] = "ABI"  # must be before any rpy2 import

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
# Raw NHIS year folders live in data/ by default.
# Override by setting NHIS_DATA_DIR environment variable.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = Path(os.getenv("NHIS_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE  = DATA_DIR / "nhis_analytic_2020_2024.csv"

# ─── Target variables ──────────────────────────────────────────────────────────
VARS = [
    "DEPFREQ_A",   # inclusion: depression frequency
    "ANXFREQ_A",   # inclusion: anxiety frequency
    "MHTHDLY_A",   # outcome:   delayed mental health care
    "AGEP_A",      # age
    "SEX_A",       # sex
    "RACEALLP_A",  # race
    "HISP_A",      # Hispanic ethnicity
    "EDUCP_A",     # education  (NOTE: named EDUC_A in 2020 — renamed below)
    "ORIENT_A",    # sexual orientation
    "NOTCOV_A",    # uninsured (not covered)
    "WTFA_A",      # survey weight
    "PSTRAT",      # strata
    "PPSU",        # primary sampling unit
]

# ─── Load, clean, and label each year ──────────────────────────────────────────
year_frames = []

for year in [2020, 2021, 2022, 2023, 2024]:
    yy = str(year)[2:]
    filepath = DATA_DIR / f"adult{yy}csv" / f"adult{yy}.csv"

    df = pd.read_csv(filepath, low_memory=False)

    # 2020 uses EDUC_A instead of EDUCP_A — rename before selecting
    if year == 2020:
        df = df.rename(columns={"EDUC_A": "EDUCP_A"})

    df = df[VARS].copy()
    df["year"] = year
    year_frames.append(df)
    print(f"  {year}: {len(df):,} rows loaded")

# ─── Stack all years ───────────────────────────────────────────────────────────
nhis_pooled = pd.concat(year_frames, ignore_index=True)

print(f"\nPooled dataset: {nhis_pooled.shape[0]:,} rows × {nhis_pooled.shape[1]} columns")
print("\nRows per year:")
print(nhis_pooled["year"].value_counts().sort_index().to_string())

# ─── Step 2: Inclusion filter ──────────────────────────────────────────────────
# Keep adults who report any frequency of depression OR anxiety symptoms.
# DEPFREQ_A / ANXFREQ_A coding:
#   1 = Daily  2 = Weekly  3 = Monthly  4 = A few times a year
#   5 = Never  7 = Refused  8 = Not Ascertained  9 = Don't Know
# Retain codes 1–4 on either variable (OR logic).
# Drop: 5 (never symptomatic), 7/8/9 (cannot determine symptom status).

SYMPTOM_CODES = [1, 2, 3, 4]

n_before = len(nhis_pooled)

nhis_symptoms = nhis_pooled[
    nhis_pooled["DEPFREQ_A"].isin(SYMPTOM_CODES) |
    nhis_pooled["ANXFREQ_A"].isin(SYMPTOM_CODES)
].copy()

n_after   = len(nhis_symptoms)
n_dropped = n_before - n_after

print(f"\n── Inclusion filter ───────────────────────────────────────────")
print(f"  Before : {n_before:,} rows")
print(f"  After  : {n_after:,} rows")
print(f"  Dropped: {n_dropped:,} ({n_dropped / n_before:.1%}) — never symptomatic or unknown")
print(f"\nRetained by year:")
print(nhis_symptoms["year"].value_counts().sort_index().to_string())

# ─── Step 3: Survey weight adjustment ─────────────────────────────────────────
# Divide individual-year weights by 5 to produce correct pooled estimates.
# This is the standard NCHS approach for pooling NHIS years.
nhis_symptoms["WTFA_5YR"] = nhis_symptoms["WTFA_A"] / 5

# Make strata and PSU IDs unique across years.
# PSTRAT and PPSU codes reset each year — without this, R treats e.g. strata 101
# from 2020 and strata 101 from 2021 as the same strata, producing wrong SEs.
nhis_symptoms["PSTRAT_POOL"] = nhis_symptoms["year"].astype(str) + "_" + nhis_symptoms["PSTRAT"].astype(str)
nhis_symptoms["PPSU_POOL"]   = (nhis_symptoms["year"].astype(str) + "_" +
                                nhis_symptoms["PSTRAT"].astype(str) + "_" +
                                nhis_symptoms["PPSU"].astype(str))

# ─── Step 4: Variable recoding ─────────────────────────────────────────────────

df = nhis_symptoms.copy()

# ── Outcome: delayed mental health care due to cost ───────────────────────────
df["mh_delay"] = df["MHTHDLY_A"].map({1: "Yes", 2: "No"})
# 7/8/9 → NaN (not mapped, become NaN automatically)

# ── Age: continuous (kept for reference) + categorical groups ────────────────
df["age"] = df["AGEP_A"].where(~df["AGEP_A"].isin([97, 98, 99]))
# Note: 85 = top-coded (85+), kept as numeric 85

def recode_age_group(age):
    if pd.isna(age):  return None
    if age < 35:      return "18–34"
    if age < 50:      return "35–49"
    if age < 65:      return "50–64"
    return                   "65+"

df["age_group"] = df["age"].map(recode_age_group)

# ── Sex ───────────────────────────────────────────────────────────────────────
df["sex"] = df["SEX_A"].map({1: "Male", 2: "Female"})

# ── Race/ethnicity: Hispanic identity overrides race ─────────────────────────
# HISP_A=1 → Hispanic regardless of RACEALLP_A
# RACEALLP_A 7/8/9 → NaN
def recode_race_eth(row):
    if row["HISP_A"] == 1:
        return "Hispanic"
    if row["RACEALLP_A"] == 1:
        return "NH White"
    if row["RACEALLP_A"] == 2:
        return "NH Black"
    if row["RACEALLP_A"] == 3:
        return "NH Asian"
    if row["RACEALLP_A"] in [4, 5]:
        return "NH AIAN"
    if row["RACEALLP_A"] == 6:
        return "NH Other/Multiracial"
    return None  # 7/8/9 or unresolvable

df["race_eth"] = df.apply(recode_race_eth, axis=1)

# ── Education: harmonize 2020 code 11 → 10, then group ───────────────────────
# 2020 had separate codes for Professional (10) and Doctoral (11).
# 2021–2024 combined them into code 10. Recode before grouping.
df["EDUCP_A"] = df["EDUCP_A"].replace(11, 10)

def recode_education(code):
    if pd.isna(code) or code in [97, 98, 99]:
        return None
    code = int(code)
    if code in [0, 1, 2]:  return "Less than High School"
    if code in [3, 4]:     return "High School / GED"
    if code in [5, 6, 7]:  return "Some College / Associate"
    if code == 8:          return "Bachelor's Degree"
    if code in [9, 10]:    return "Graduate Degree"
    return None

df["education"] = df["EDUCP_A"].map(recode_education)

# ── Sexual orientation ────────────────────────────────────────────────────────
# Codes 4 (Something else) and 5 (I don't know) → "Queer or Questioning"
# Codes 7/8 → NaN
df["orientation"] = df["ORIENT_A"].map({
    1: "Gay or Lesbian",
    2: "Straight",
    3: "Bisexual",
    4: "Queer or Questioning",
    5: "Queer or Questioning",
})

# ── Insurance status ──────────────────────────────────────────────────────────
df["insured"] = df["NOTCOV_A"].map({1: "Uninsured", 2: "Insured"})
# 9 → NaN

# ─── Replace working dataframe with recoded version ───────────────────────────
nhis_symptoms = df

print("\n── Step 4: Recoding complete ──────────────────────────────────")
print(f"\n  age (continuous — kept for reference):")
print(f"    Mean  : {nhis_symptoms['age'].mean():.1f}")
print(f"    Median: {nhis_symptoms['age'].median():.0f}")
print(f"    Range : {nhis_symptoms['age'].min():.0f}–{nhis_symptoms['age'].max():.0f}")
print(f"    NaN   : {nhis_symptoms['age'].isna().sum():,}")
for col in ["age_group", "mh_delay", "sex", "race_eth", "education", "orientation", "insured"]:
    print(f"\n  {col}:")
    print(nhis_symptoms[col].value_counts(dropna=False).to_string())

# ─── Save analytic file ────────────────────────────────────────────────────────
nhis_symptoms.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved analytic file: {OUTPUT_FILE.name}")

# ─── Step 3 (cont): Survey design via rpy2 ────────────────────────────────────
print("\nLoading R survey package...")
survey = importr("survey")

with (ro.default_converter + pandas2ri.converter).context():
    ro.globalenv["nhis"] = nhis_symptoms

ro.r("""
options(survey.lonely.psu = "adjust")

svy_nhis <- svydesign(
    id      = ~PPSU_POOL,
    strata  = ~PSTRAT_POOL,
    weights = ~WTFA_5YR,
    data    = nhis,
    nest    = TRUE
)
""")

print(ro.r("cat('Survey design ready:\\n'); print(summary(svy_nhis))"))
print("\n✓ Survey design object 'svy_nhis' is ready for analysis.")
