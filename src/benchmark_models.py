import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from preprocessor import (
    Stage1Preprocessor,
    MINIMAL_FEATURES,
    FULL_FEATURES,
)

from evaluate_classifier import (
    evaluate_feature_set,
    print_banner,
)

TARGET_COL = "Target_24m"

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "adni_24m_progression_dataset_filled.csv"

# ============================================================
# MODELS
# ============================================================

def build_lr():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",
         LogisticRegression(
             max_iter=2000,
             solver="lbfgs",
             class_weight="balanced",
             random_state=42,
         )),
    ])


def build_rf():

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_xgb():

    return XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.0,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = pd.read_csv(DATA_PATH)

    y = df[TARGET_COL].values

    models = [
        ("LR", build_lr),
        ("RF", build_rf),
        ("XGB", build_xgb),
    ]

    feature_sets = [
        ("Minimal", MINIMAL_FEATURES),
        ("Full", FULL_FEATURES),
    ]

    results = []

    for model_name, model_builder in models:

        for feature_name, feature_list in feature_sets:

            results.append(
                evaluate_feature_set(
                    df=df,
                    y=y,
                    estimator=model_builder(),
                    preprocessor_cls=Stage1Preprocessor,
                    features=feature_list,
                    model_name=f"{model_name}-{feature_name}",
                )
            )

    print_banner("MODEL COMPARISON")

    summary_df = pd.DataFrame([
        {
            "Model": r.name,
            "AUC": round(r.auc_oof, 3),
            "Sens": round(r.metrics["sensitivity"], 3),
            "Spec": round(r.metrics["specificity"], 3),
            "Acc": round(r.metrics["accuracy"], 3),
            "Brier": round(r.metrics["brier"], 3),
            "BSS": round(r.bss, 3),
            "N_Features": len(r.features),
        }
        for r in results
    ])

    print(
        summary_df
        .sort_values("AUC", ascending=False)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()