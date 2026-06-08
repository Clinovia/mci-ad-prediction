import pandas as pd

# =========================================================
# 1. LOAD DATA
# =========================================================

main_df = pd.read_csv("adni_24m_progression_dataset.csv")
adni_merge = pd.read_csv("ADNIMERGE.csv")

print("Main dataset shape:", main_df.shape)
print("ADNIMERGE shape:", adni_merge.shape)

# =========================================================
# 2. SELECT BASELINE FROM ADNIMERGE
# =========================================================

adni_bl = adni_merge[adni_merge["VISCODE"] == "bl"].copy()

# Keep only needed columns
cols_needed = ["RID", "RAVLT_immediate", "EcogSPMem"]
adni_bl = adni_bl[cols_needed]

# Ensure numeric
for col in ["RAVLT_immediate", "EcogSPMem"]:
    adni_bl[col] = pd.to_numeric(adni_bl[col], errors="coerce")

print("Baseline ADNIMERGE shape:", adni_bl.shape)

# =========================================================
# 3. MERGE INTO MAIN DATASET
# =========================================================

df = main_df.merge(adni_bl, on="RID", how="left")

print("After merge shape:", df.shape)

# =========================================================
# 4. CREATE ECOG DISCREPANCY
# =========================================================

# Ensure EcogPtMem exists
if "EcogPtMem" not in df.columns:
    raise ValueError("EcogPtMem not found in main dataset")

df["EcogMem_discrepancy"] = df["EcogSPMem"] - df["EcogPtMem"]

# =========================================================
# 5. OPTIONAL: HANDLE MISSING VALUES
# =========================================================

# Fill with median (same strategy as your model)
for col in ["RAVLT_immediate", "EcogSPMem", "EcogMem_discrepancy"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# =========================================================
# 6. SAVE UPDATED DATASET
# =========================================================

df.to_csv("adni_24m_progression_dataset.csv", index=False)

print("\nUpdated dataset saved: adni_24m_progression_dataset.csv")

# =========================================================
# 7. VERIFY
# =========================================================

print("\nNew columns added:")
print(df[["RAVLT_immediate", "EcogSPMem", "EcogMem_discrepancy"]].head())

print("\nEcogSPMem stats:")
print(df["EcogSPMem"].describe())

print("\nUnique values (first 20):")
print(df["EcogSPMem"].unique()[:20])