import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    brier_score_loss,
)

N_BOOTSTRAPS = 2000
RANDOM_STATE = 42


def bootstrap_metrics(y_true, y_prob, threshold=0.5):

    rng = np.random.RandomState(RANDOM_STATE)

    aucs = []
    sensitivities = []
    specificities = []
    accuracies = []
    briers = []

    y_pred = (y_prob >= threshold).astype(int)

    for _ in range(N_BOOTSTRAPS):

        indices = rng.choice(
            len(y_true),
            size=len(y_true),
            replace=True,
        )

        yt = y_true[indices]
        yp_prob = y_prob[indices]
        yp = y_pred[indices]

        # Skip samples containing only one class
        if len(np.unique(yt)) < 2:
            continue

        aucs.append(
            roc_auc_score(yt, yp_prob)
        )

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
        ).ravel()

        sensitivities.append(
            tp / (tp + fn)
        )

        specificities.append(
            tn / (tn + fp)
        )

        accuracies.append(
            accuracy_score(yt, yp)
        )

        briers.append(
            brier_score_loss(
                yt,
                yp_prob,
            )
        )

    return {
        "AUC": aucs,
        "Sensitivity": sensitivities,
        "Specificity": specificities,
        "Accuracy": accuracies,
        "Brier": briers,
    }


def summarize(metric_name, values):

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)
    mean = np.mean(values)

    print(
        f"{metric_name}: "
        f"{mean:.3f} "
        f"(95% CI {lower:.3f}-{upper:.3f})"
    )


def main():

    results_dir = (
        Path(__file__).resolve().parent.parent
        / "results"
    )

    prediction_files = [
        results_dir / "LR-Full_oof_predictions.csv",
        results_dir / "LR-Minimal_oof_predictions.csv",
        results_dir / "RF-Full_oof_predictions.csv",
        results_dir / "RF-Minimal_oof_predictions.csv",
    ]

    # Ensure all files exist
    prediction_files = [
        f for f in prediction_files
        if f.exists()
    ]

    if not prediction_files:
        raise FileNotFoundError(
            "No prediction files found."
        )

    for file in prediction_files:

        print("\n" + "=" * 60)
        print(file.stem)
        print("=" * 60)

        df = pd.read_csv(file)

        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        metrics = bootstrap_metrics(
            y_true,
            y_prob,
        )

        for name, values in metrics.items():
            summarize(name, values)


if __name__ == "__main__":
    main()