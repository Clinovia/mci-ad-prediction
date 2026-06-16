"""
Shared evaluation utilities.

Supports:
    - Logistic Regression
    - Random Forest
    - XGBoost

Features:
    - Fold-level preprocessing (no leakage)
    - Stratified 5-fold CV
    - Out-of-fold predictions
    - ROC/AUC
    - Sensitivity
    - Specificity
    - Accuracy
    - Brier Score
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    brier_score_loss,
)

from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIG
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EvaluationResult:
    name: str
    features: list

    fold_aucs: list

    auc_oof: float
    mean_auc: float
    std_auc: float

    threshold: float

    metrics: dict

    brier_null: float
    bss: float

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray

    oof_probs: np.ndarray

    y_true: np.ndarray
    y_prob: np.ndarray
    threshold: float


# ============================================================
# HELPERS
# ============================================================

def print_banner(text):
    print(f"\n{'=' * 70}")
    print(text)
    print(f"{'=' * 70}")


def youden_threshold(
    fpr,
    tpr,
    thresholds,
):
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def metrics_at_threshold(
    y_true,
    y_prob,
    threshold,
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return {
        "threshold": threshold,
        "sensitivity":
            tp / (tp + fn)
            if (tp + fn)
            else 0.0,

        "specificity":
            tn / (tn + fp)
            if (tn + fp)
            else 0.0,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "brier":
            brier_score_loss(
                y_true,
                y_prob,
            ),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cv(
    *,
    df,
    y,
    estimator,
    preprocessor_cls,
    features,
    verbose=True,
):
    """
    Returns:
        fold_aucs
        oof_probs
        feature_names
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_aucs = []
    oof_probs = np.zeros(len(y))

    feature_names = None

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv.split(df, y),
        start=1,
    ):

        train_df = (
            df.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        val_df = (
            df.iloc[val_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_train = y[train_idx]
        y_val = y[val_idx]

        preprocessor = preprocessor_cls(
            features=features
        )

        X_train_df = (
            preprocessor.fit_transform(
                train_df
            )
        )

        X_val_df = (
            preprocessor.transform(
                val_df
            )
        )

        if feature_names is None:

            feature_names = list(
                X_train_df.columns
            )

            if verbose:
                print(
                    f"Features ({len(feature_names)}):"
                )
                print(feature_names)

        X_train = (
            X_train_df.values
            .astype(float)
        )

        X_val = (
            X_val_df.values
            .astype(float)
        )

        estimator.fit(
            X_train,
            y_train,
        )

        probs = (
            estimator
            .predict_proba(X_val)[:, 1]
        )

        oof_probs[val_idx] = probs

        fold_auc = roc_auc_score(
            y_val,
            probs,
        )

        fold_aucs.append(
            fold_auc
        )

        if verbose:
            print(
                f"Fold {fold_idx}: "
                f"AUC={fold_auc:.4f}"
            )

    return (
        fold_aucs,
        oof_probs,
        feature_names,
    )


# ============================================================
# OOF METRICS
# ============================================================

def calculate_oof_metrics(
    y,
    oof_probs,
):

    auc_oof = roc_auc_score(
        y,
        oof_probs,
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        oof_probs,
    )

    threshold = youden_threshold(
        fpr,
        tpr,
        thresholds,
    )

    metrics = metrics_at_threshold(
        y,
        oof_probs,
        threshold,
    )

    brier_null = brier_score_loss(
        y,
        np.full(
            len(y),
            y.mean(),
        ),
    )

    bss = (
        1.0
        - metrics["brier"]
        / brier_null
    )

    return {
        "auc_oof": auc_oof,
        "threshold": threshold,
        "metrics": metrics,
        "brier_null": brier_null,
        "bss": bss,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


# ============================================================
# PUBLIC API
# ============================================================

def evaluate_feature_set(
    *,
    df,
    y,
    estimator,
    preprocessor_cls,
    features,
    model_name,
    verbose=True,
):

    if verbose:
        print_banner(model_name)

    fold_aucs, oof_probs, feature_names = run_cv(
        df=df,
        y=y,
        estimator=estimator,
        preprocessor_cls=preprocessor_cls,
        features=features,
        verbose=verbose,
    )

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    oof = calculate_oof_metrics(
        y,
        oof_probs,
    )

    if verbose:

        print(
            f"\nMean CV AUC: "
            f"{mean_auc:.4f} ± {std_auc:.4f}"
        )

        print(
            f"OOF AUC     : "
            f"{oof['auc_oof']:.4f}"
        )

        print(
            f"Sensitivity : "
            f"{oof['metrics']['sensitivity']:.4f}"
        )

        print(
            f"Specificity : "
            f"{oof['metrics']['specificity']:.4f}"
        )

        print(
            f"Accuracy    : "
            f"{oof['metrics']['accuracy']:.4f}"
        )

        print(
            f"Brier Score : "
            f"{oof['metrics']['brier']:.4f}"
        )

    return EvaluationResult(
        name=model_name,
        features=features,

        fold_aucs=fold_aucs,

        auc_oof=oof["auc_oof"],
        mean_auc=mean_auc,
        std_auc=std_auc,

        threshold=oof["threshold"],

        metrics=oof["metrics"],

        brier_null=oof["brier_null"],
        bss=oof["bss"],

        fpr=oof["fpr"],
        tpr=oof["tpr"],
        thresholds=oof["thresholds"],

        oof_probs=oof_probs,

        y_true=y,
        y_prob=oof_probs,
    )