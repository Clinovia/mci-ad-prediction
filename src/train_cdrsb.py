"""
train_cdrsb.py
--------------
Evaluates CDR-SB as a clinical baseline comparator for Table 2.

CDR-SB (Clinical Dementia Rating — Sum of Boxes) is the standard
clinical staging tool used in every AD trial. A fixed threshold is
applied to baseline CDR-SB to classify 24-month progression risk.

Strategy
--------
  1. Sweep CDR-SB thresholds to find the Youden-optimal cutoff
     (mirrors the threshold selection used in the XGBoost models).
  2. Report AUC, sensitivity, specificity, accuracy, and Brier score
     at that threshold — ready for Table 2.
  3. Also report a literature-standard threshold (CDR-SB >= 1.5)
     for reference.

Usage
-----
    python src/train_cdrsb.py
    python src/train_cdrsb.py --cdrsb-col CDRSB   # if column name differs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# =========================================================
# PROJECT SETUP
# =========================================================

CURRENT_DIR  = Path(__file__).resolve().parent          # src/
PROJECT_ROOT = CURRENT_DIR.parent                       # stage1_clinical/
DATA_PATH    = PROJECT_ROOT / "data" / "adni_24m_progression_dataset_filled.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"
OUTPUT_CSV   = RESULTS_DIR / "baseline_cdrsb_results.csv"


# =========================================================
# CONFIG
# =========================================================

TARGET_COL  = "Target_24m"
CDRSB_COL   = "CDRSB"
LIT_THRESH  = 1.5               # literature-standard clinical threshold


# =========================================================
# HELPERS
# =========================================================

def youden_threshold(fpr, tpr, thresholds):
    """Return threshold maximising Youden Index (sens + spec - 1)."""
    J   = tpr - fpr
    idx = np.argmax(J)
    return thresholds[idx], tpr[idx], 1 - fpr[idx]


def metrics_at_threshold(y_true, y_score, threshold):
    """Confusion-matrix metrics. Brier filled by caller."""
    y_pred          = (y_score >= threshold).astype(int)
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = accuracy_score(y_true, y_pred)
    return dict(
        threshold=threshold,
        sensitivity=sens,
        specificity=spec,
        accuracy=acc,
        brier=None,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def print_result(label, auc, m):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  AUC          : {auc:.4f}")
    print(f"  Threshold    : {m['threshold']:.4f}")
    print(f"  Sensitivity  : {m['sensitivity']:.4f}")
    print(f"  Specificity  : {m['specificity']:.4f}")
    print(f"  Accuracy     : {m['accuracy']:.4f}")
    print(f"  Brier Score  : {m['brier']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="CDR-SB clinical baseline comparator for Table 2"
    )
    parser.add_argument(
        "--cdrsb-col",
        default=CDRSB_COL,
        help="Column name for CDR-SB in the CSV (default: CDRSB)",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        sys.exit(
            f"[ERROR] Data file not found at {DATA_PATH}\n"
            "Please download the ADNI dataset and place it in the data/ directory.\n"
            "Access available at: https://adni.loni.usc.edu"
        )

    print(f"\nReading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape   : {df.shape}")
    print(f"  Columns (first 20): {list(df.columns[:20])}")

    # ── Column checks ─────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        sys.exit(f"[ERROR] Target column '{TARGET_COL}' not found.")

    cdrsb_col = args.cdrsb_col
    if cdrsb_col not in df.columns:
        candidates = [c for c in df.columns
                      if "cdrsb" in c.lower() or "cdr" in c.lower()]
        if candidates:
            cdrsb_col = candidates[0]
            print(f"  [INFO] '{args.cdrsb_col}' not found — using '{cdrsb_col}' instead.")
        else:
            sys.exit(
                f"[ERROR] No CDR-SB column found.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Pass the correct name with --cdrsb-col COLNAME"
            )

    print(f"\n  CDR-SB column : '{cdrsb_col}'")
    print(f"  CDR-SB unique values (first 10): "
          f"{sorted(df[cdrsb_col].dropna().unique())[:10]}")

    # ── Prepare analytic sample ────────────────────────────────────────
    analytic              = df[[TARGET_COL, cdrsb_col]].copy()
    analytic[cdrsb_col]   = pd.to_numeric(analytic[cdrsb_col], errors="coerce")
    before                = len(analytic)
    analytic              = analytic.dropna()
    after                 = len(analytic)

    print(f"\n  Rows with both CDR-SB and outcome : {after} "
          f"(dropped {before - after} with missing values)")

    y_true  = analytic[TARGET_COL].values
    y_score = analytic[cdrsb_col].values       # higher = more impaired = higher risk

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    print(f"  Progressors    : {n_pos} ({100 * n_pos / after:.1f}%)")
    print(f"  Non-progressors: {n_neg} ({100 * n_neg / after:.1f}%)")

    # ── ROC / AUC ─────────────────────────────────────────────────────
    auc                    = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds   = roc_curve(y_true, y_score)

    # ── Null Brier ─────────────────────────────────────────────────────
    prevalence  = y_true.mean()
    brier_null  = brier_score_loss(
        y_true, np.full_like(y_true, prevalence, dtype=float)
    )

    # ── Youden-optimal threshold ───────────────────────────────────────
    opt_thresh, _, _ = youden_threshold(fpr, tpr, thresholds)

    # Min-max normalise CDR-SB to [0, 1] as probability proxy for Brier
    cdrsb_min = y_score.min()
    cdrsb_max = y_score.max()
    y_prob = (
        (y_score - cdrsb_min) / (cdrsb_max - cdrsb_min)
        if cdrsb_max > cdrsb_min
        else np.full_like(y_score, 0.5, dtype=float)
    )
    brier = brier_score_loss(y_true, y_prob)
    bss   = 1.0 - (brier / brier_null)

    # ── Results: Youden-optimal ────────────────────────────────────────
    m_opt          = metrics_at_threshold(y_true, y_score, opt_thresh)
    m_opt["brier"] = brier
    print_result(f"CDR-SB — Youden-optimal threshold (≥ {opt_thresh:.2f})", auc, m_opt)

    # ── Results: literature threshold (CDR-SB >= 1.5) ─────────────────
    m_lit          = metrics_at_threshold(y_true, y_score, LIT_THRESH)
    m_lit["brier"] = brier
    print_result(f"CDR-SB — Literature threshold (≥ {LIT_THRESH})", auc, m_lit)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  Null Brier Score  : {brier_null:.4f}")
    print(f"  Brier Skill Score : {bss:.4f}  (1 - BS/BS_null)")

    print(f"\n{'=' * 60}")
    print("  TABLE 2 ROW — CDR-SB (Youden-optimal threshold)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<42} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'Brier':>7}")
    print(
        f"  {'CDR-SB threshold (clinical baseline)':<42} "
        f"{auc:>6.3f} "
        f"{m_opt['sensitivity']:>6.3f} "
        f"{m_opt['specificity']:>6.3f} "
        f"{m_opt['brier']:>7.3f}"
    )
    print(f"{'=' * 60}\n")

    # ── Save ───────────────────────────────────────────────────────────
    out = pd.DataFrame([
        {
            "Model":             "CDR-SB threshold (clinical baseline)",
            "Threshold_type":    "Youden-optimal",
            "Threshold_value":   round(float(opt_thresh), 3),
            "AUC":               round(float(auc), 4),
            "Sensitivity":       round(float(m_opt["sensitivity"]), 4),
            "Specificity":       round(float(m_opt["specificity"]), 4),
            "Accuracy":          round(float(m_opt["accuracy"]), 4),
            "Brier_Score":       round(float(brier), 4),
            "Brier_Null":        round(float(brier_null), 4),
            "Brier_Skill_Score": round(float(bss), 4),
            "N":                 after,
            "N_pos":             n_pos,
            "N_neg":             n_neg,
        },
        {
            "Model":             "CDR-SB threshold (literature, >= 1.5)",
            "Threshold_type":    "Literature",
            "Threshold_value":   LIT_THRESH,
            "AUC":               round(float(auc), 4),
            "Sensitivity":       round(float(m_lit["sensitivity"]), 4),
            "Specificity":       round(float(m_lit["specificity"]), 4),
            "Accuracy":          round(float(m_lit["accuracy"]), 4),
            "Brier_Score":       round(float(brier), 4),
            "Brier_Null":        round(float(brier_null), 4),
            "Brier_Skill_Score": round(float(bss), 4),
            "N":                 after,
            "N_pos":             n_pos,
            "N_neg":             n_neg,
        },
    ])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()