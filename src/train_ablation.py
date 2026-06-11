from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =========================================================
# PROJECT SETUP
# =========================================================

CURRENT_DIR  = Path(__file__).resolve().parent         
PROJECT_ROOT = CURRENT_DIR.parent                      
DATA_PATH    = PROJECT_ROOT / "data" / "adni_24m_progression_dataset_filled.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"
OUTPUT_CSV   = RESULTS_DIR / "ablation_results.csv"

sys.path.append(str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

from src.preprocessor_ablation import (
    Stage1Preprocessor,
    DEMOGRAPHIC_FEATURES,
    GENETIC_FEATURES,
    COGNITIVE_FEATURES,
    FUNCTIONAL_FEATURES,
    MEMORY_FEATURES,
    BASE_FEATURES,
)


# =========================================================
# CONFIG
# =========================================================

TARGET       = "Target_24m"
RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

if not DATA_PATH.exists():
    sys.exit(
        f"[ERROR] Data file not found at {DATA_PATH}\n"
        "Please download the ADNI dataset and place it in the data/ directory.\n"
        "Access available at: https://adni.loni.usc.edu"
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET])
y  = df[TARGET].astype(int)

print(f"\nLoaded: {DATA_PATH}")
print(f"Rows  : {len(df)}")
print(f"Progressors   : {y.sum()} ({100 * y.mean():.1f}%)")
print(f"Non-progressors: {(1 - y).sum()} ({100 * (1 - y.mean()):.1f}%)")


# =========================================================
# FEATURE GROUPS
# =========================================================
# BASE_FEATURES = AGE, PTGENDER, PTEDUCAT, RAVLT_immediate, MMSE, EcogSPTotal
# APOE4 added explicitly at each level to isolate its marginal contribution

ABLATION_MODELS = {

    # --- Demographics only ---
    "Demographics":
        DEMOGRAPHIC_FEATURES,

    "Demographics + APOE4":
        DEMOGRAPHIC_FEATURES + GENETIC_FEATURES,

    # --- + RAVLT immediate ---
    "Demographics + RAVLT":
        DEMOGRAPHIC_FEATURES + MEMORY_FEATURES,

    "Demographics + RAVLT + APOE4":
        DEMOGRAPHIC_FEATURES + MEMORY_FEATURES + GENETIC_FEATURES,

    # --- + MMSE ---
    "Demographics + RAVLT + MMSE":
        DEMOGRAPHIC_FEATURES + MEMORY_FEATURES + COGNITIVE_FEATURES,

    "Demographics + RAVLT + MMSE + APOE4":
        DEMOGRAPHIC_FEATURES + MEMORY_FEATURES + COGNITIVE_FEATURES + GENETIC_FEATURES,

    # --- + EcogSPTotal (= BASE_FEATURES, full minimal set) ---
    "Demographics + RAVLT + MMSE + EcogSPTotal":
        BASE_FEATURES,

    "Demographics + RAVLT + MMSE + EcogSPTotal + APOE4":
        BASE_FEATURES + GENETIC_FEATURES,
}


# =========================================================
# SANITY CHECK
# =========================================================

print("\nABLATION_MODELS — resolved feature sets:")
for name, feats in ABLATION_MODELS.items():
    print(f"  ({len(feats):2d} features) {name}")
    print(f"    → {feats}")


# =========================================================
# TRAIN / EVALUATE
# =========================================================

print("\n" + "=" * 60)
print("ABLATION STUDY")
print("=" * 60)

results = []

for model_name, features in ABLATION_MODELS.items():

    print(f"\n--- {model_name} ---")

    preprocessor = Stage1Preprocessor(selected_features=features)
    X = preprocessor.fit_transform(df[features])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=1.0,
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"AUC = {auc:.4f}")

    results.append({
        "Model":        model_name,
        "Num_Features": len(features),
        "AUC":          round(float(auc), 4),
        "Features":     ", ".join(features),
    })


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(results_df[["Model", "Num_Features", "AUC"]].to_string(index=False))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")