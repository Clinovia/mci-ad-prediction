"""
XGBoost_SHAP.py
---------------
SHAP interpretability analysis for the XGBoost full model (Stage 2).

Computes TreeExplainer SHAP values on the full nine-feature XGBoost model
and saves ranked feature importance, raw SHAP values, and plots.

Usage
-----
    python src/XGBoost_SHAP.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =========================================================
# PROJECT SETUP
# =========================================================

CURRENT_DIR  = Path(__file__).resolve().parent          # src/
PROJECT_ROOT = CURRENT_DIR.parent                       # stage1_clinical/
DATA_PATH    = PROJECT_ROOT / "data" / "adni_24m_progression_dataset_filled.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"
OUTPUT_DIR   = RESULTS_DIR / "shap"

sys.path.append(str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

from src.preprocessor import Stage1Preprocessor, FULL_FEATURES


# =========================================================
# CONFIG
# =========================================================

TARGET_COL   = "Target_24m"
RANDOM_STATE = 42
TOP_N        = 20           # features shown in plots


# =========================================================
# MODEL
# =========================================================

def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =========================================================
# SHAP ANALYSIS
# =========================================================

def run_shap_analysis(
    df: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("SHAP ANALYSIS — FULL MODEL")
    print("==============================")

    # ── Preprocess ────────────────────────────────────────────
    preprocessor  = Stage1Preprocessor(features=FULL_FEATURES)
    X             = preprocessor.fit_transform(df)
    feature_names = list(X.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # ── Train / test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ── Fit ───────────────────────────────────────────────────
    model = build_xgb()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"  Model fitted  |  train={len(X_train)}  test={len(X_test)}")

    # ── SHAP values (TreeExplainer — exact for XGB) ───────────
    explainer  = shap.TreeExplainer(model)
    shap_exp   = explainer(X_test)      # shap.Explanation object
    shap_array = shap_exp.values        # (n_test, n_features)

    # ── 1. Beeswarm plot ──────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_exp, max_display=TOP_N, show=False)
    plt.title("SHAP Summary — Full XGB (beeswarm)", fontsize=13)
    plt.tight_layout()
    p = output_dir / "shap_beeswarm.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")

    # ── 2. Mean |SHAP| bar chart ──────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_exp, max_display=TOP_N, show=False)
    plt.title("SHAP Feature Importance — Full XGB (mean |SHAP|)", fontsize=13)
    plt.tight_layout()
    p = output_dir / "shap_bar.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")

    # ── 3. Raw SHAP matrix ────────────────────────────────────
    shap_df = pd.DataFrame(shap_array, columns=feature_names)
    p = output_dir / "shap_values.csv"
    shap_df.to_csv(p, index=False)
    print(f"  Saved → {p}")

    # ── 4. Ranked mean |SHAP| table ───────────────────────────
    mean_abs = (
        pd.Series(
            np.abs(shap_array).mean(axis=0),
            index=feature_names,
            name="mean_abs_shap",
        )
        .sort_values(ascending=False)
    )
    p = output_dir / "shap_importance.csv"
    mean_abs.rename_axis("feature").reset_index().to_csv(p, index=False)
    print(f"  Saved → {p}")

    print("\n  Ranked feature importance (mean |SHAP|):")
    print(mean_abs.round(4).to_string())

    X_test.to_csv(output_dir / "X_test.csv", index=False)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    if not DATA_PATH.exists():
        sys.exit(
            f"[ERROR] Data file not found at {DATA_PATH}\n"
            "Please download the ADNI dataset and place it in the data/ directory.\n"
            "Access available at: https://adni.loni.usc.edu"
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL])
    y  = df[TARGET_COL].values

    print(f"\nLoaded: {DATA_PATH}")
    print(f"Rows  : {len(df)}")
    print(f"Progressors    : {int(y.sum())} ({100 * y.mean():.1f}%)")
    print(f"Non-progressors: {int((1 - y).sum())} ({100 * (1 - y.mean()):.1f}%)")

    run_shap_analysis(df, y, output_dir=OUTPUT_DIR)