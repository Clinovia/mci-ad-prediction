import numpy as np
import pandas as pd


BASE_FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "MMSE",
    "EcogSPTotal",
    "RAVLT_immediate",
]

GENDER_MAP = {
    "Male": 1,
    "Female": 0,
}


class Stage1Preprocessor:

    def __init__(self):

        self.imputation_values = {}
        self.feature_order = None

    def _normalize_gender(self, series):

        # Already numeric?
        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        valid_numeric = numeric.isin([0, 1])

        result = numeric.copy()

        # Only map string values where numeric conversion failed
        mask = ~valid_numeric

        result.loc[mask] = (
            series.loc[mask]
            .map(GENDER_MAP)
        )

        return pd.to_numeric(
            result,
            errors="coerce"
        )

    def fit(self, df: pd.DataFrame):

        model_df = df.copy()

        # -------------------------
        # PTGENDER diagnostics
        # -------------------------
        print("\nPTGENDER RAW")
        print(
            model_df["PTGENDER"]
            .value_counts(dropna=False)
        )

        model_df["PTGENDER"] = (
            self._normalize_gender(
                model_df["PTGENDER"]
            )
        )

        print("\nPTGENDER NORMALIZED")
        print(
            model_df["PTGENDER"]
            .value_counts(dropna=False)
        )

        if model_df["PTGENDER"].notna().sum() == 0:
            raise ValueError(
                "PTGENDER became entirely NaN "
                "after normalization."
            )

        model_df["PTGENDER_missing"] = (
            model_df["PTGENDER"]
            .isna()
            .astype(int)
        )

        # -------------------------
        # Numeric features
        # -------------------------
        for col in BASE_FEATURES:

            if col == "PTGENDER":
                continue

            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce"
            )

            median = model_df[col].median()

            if np.isnan(median):
                median = 0.0

            self.imputation_values[col] = (
                float(median)
            )

        # -------------------------
        # Gender median
        # -------------------------
        gender_median = (
            model_df["PTGENDER"]
            .median()
        )

        if np.isnan(gender_median):
            gender_median = 0.0

        self.imputation_values[
            "PTGENDER"
        ] = float(gender_median)

        self.feature_order = (
            BASE_FEATURES
            + ["PTGENDER_missing"]
        )

        print("\nFEATURE ORDER")
        print(self.feature_order)

        return self

    def transform(self, df: pd.DataFrame):

        model_df = df.copy()

        model_df["PTGENDER"] = (
            self._normalize_gender(
                model_df["PTGENDER"]
            )
        )

        model_df["PTGENDER_missing"] = (
            model_df["PTGENDER"]
            .isna()
            .astype(int)
        )

        for col in BASE_FEATURES:

            if col == "PTGENDER":
                continue

            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce"
            )

            model_df[col] = (
                model_df[col]
                .fillna(
                    self.imputation_values[col]
                )
            )

        model_df["PTGENDER"] = (
            model_df["PTGENDER"]
            .fillna(
                self.imputation_values[
                    "PTGENDER"
                ]
            )
        )

        result = model_df[
            self.feature_order
        ]

        # -------------------------
        # Diagnostics
        # -------------------------
        print("\nPTGENDER AFTER TRANSFORM")
        print(
            result["PTGENDER"]
            .value_counts(dropna=False)
        )

        print("\nPTGENDER_missing")
        print(
            result["PTGENDER_missing"]
            .value_counts(dropna=False)
        )

        return result

    def fit_transform(self, df):

        return self.fit(df).transform(df)