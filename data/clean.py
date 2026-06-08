import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = "adni_24m_progression_dataset.csv"
OUTPUT_PATH = "adni_24m_progression_dataset_cleaned.csv"

ECOG_COLS = ["EcogSPMem", "EcogPtMem"]
DERIVED_COL = "EcogMem_discrepancy"

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_PATH)
print("Initial shape:", df.shape)

# =========================================================
# 2. BASIC SANITY CHECK
# =========================================================

def print_summary(df, cols):
    for col in cols:
        print(f"\n{col} summary:")
        print(df[col].describe())
        print("Unique (sample):", df[col].dropna().unique()[:10])

print_summary(df, ECOG_COLS + [DERIVED_COL])

# =========================================================
# 3. FORCE NUMERIC (SAFE)
# =========================================================

for col in ECOG_COLS:
    df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

# =========================================================
# 4. CLEAN INVALID ECOG VALUES (NO WARNING)
# =========================================================

def clean_ecog(df, col):
    invalid_mask = (df[col] < 1) | (df[col] > 4)
    print(f"{col}: removing {invalid_mask.sum()} invalid values")
    df.loc[invalid_mask, col] = np.nan

for col in ECOG_COLS:
    clean_ecog(df, col)

# =========================================================
# 5. RECOMPUTE DISCREPANCY (AUTHORITATIVE)
# =========================================================

df.loc[:, DERIVED_COL] = df["EcogSPMem"] - df["EcogPtMem"]

# =========================================================
# 6. HANDLE EXTREME DISCREPANCY
# =========================================================

df.loc[:, DERIVED_COL] = df[DERIVED_COL].clip(-3, 3)

# =========================================================
# 7. OPTIONAL: ROBUST OUTLIER CLIPPING
# (safe because ECOG is bounded anyway)
# =========================================================

def clip_outliers(df, col, lower_q=0.01, upper_q=0.99):
    low = df[col].quantile(lower_q)
    high = df[col].quantile(upper_q)
    df.loc[:, col] = df[col].clip(low, high)

for col in ECOG_COLS + [DERIVED_COL]:
    clip_outliers(df, col)

# =========================================================
# 8. MISSING VALUE HANDLING
# (keep flags even if no missing → production consistency)
# =========================================================

for col in ECOG_COLS + [DERIVED_COL]:
    missing_count = df[col].isna().sum()
    print(f"{col}: missing = {missing_count}")

    # Always create missing flag (important for deployment)
    df.loc[:, f"{col}_missing"] = df[col].isna().astype(int)

    # Median imputation
    median_val = df[col].median()
    df.loc[:, col] = df[col].fillna(median_val)

# =========================================================
# 9. FINAL CHECK
# =========================================================

print("\nAfter cleaning:")
print_summary(df, ECOG_COLS + [DERIVED_COL])

# =========================================================
# 10. SAVE CLEAN DATA
# =========================================================

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")