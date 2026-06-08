import numpy as np
import pandas as pd


MINIMAL_FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "MMSE",
    "EcogSPTotal",
    "RAVLT_immediate",
]


FULL_FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "APOE4",
    "MMSE",
    "EcogSPTotal",
    "EcogMem_discrepancy",
    "RAVLT_forgetting",
    "RAVLT_immediate",
]


GENDER_MAP = {
    "Male": 1,
    "Female": 0,
}


class Stage1Preprocessor:

    def __init__(
        self,
        features,
        verbose=False,
    ):
        self.features = list(features)
        self.verbose = verbose

        self.imputation_values = {}
        self.feature_order = None

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _normalize_gender(self, series):

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        valid_numeric = numeric.isin([0, 1])

        result = numeric.copy()

        mask = ~valid_numeric

        result.loc[mask] = (
            series.loc[mask]
            .map(GENDER_MAP)
        )

        return pd.to_numeric(
            result,
            errors="coerce",
        )

    # =========================================================
    # FIT
    # =========================================================

    def fit(self, df: pd.DataFrame):

        model_df = df.copy()

        self._log("\nPTGENDER RAW")
        self._log(
            model_df["PTGENDER"]
            .value_counts(dropna=False)
        )

        model_df["PTGENDER"] = (
            self._normalize_gender(
                model_df["PTGENDER"]
            )
        )

        self._log("\nPTGENDER NORMALIZED")
        self._log(
            model_df["PTGENDER"]
            .value_counts(dropna=False)
        )

        if model_df["PTGENDER"].notna().sum() == 0:
            raise ValueError(
                "PTGENDER became entirely NaN "
                "after normalization."
            )


        # -------------------------------------
        # Learn imputations
        # -------------------------------------

        for col in self.features:

            if col == "PTGENDER":
                continue

            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce",
            )

            median = model_df[col].median()

            if np.isnan(median):
                median = 0.0

            self.imputation_values[col] = (
                float(median)
            )

        gender_median = (
            model_df["PTGENDER"]
            .median()
        )

        if np.isnan(gender_median):
            gender_median = 0.0

        self.imputation_values[
            "PTGENDER"
        ] = float(gender_median)


        self._log("\nFEATURE ORDER")
        self._log(self.feature_order)
        self.feature_order = list(self.features)

        return self

    # =========================================================
    # TRANSFORM
    # =========================================================

    def transform(self, df: pd.DataFrame):

        if self.feature_order is None:
            raise RuntimeError(
                "Preprocessor must be fitted first."
            )

        model_df = df.copy()

        model_df["PTGENDER"] = (
            self._normalize_gender(
                model_df["PTGENDER"]
            )
        )


        for col in self.features:

            if col == "PTGENDER":
                continue

            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce",
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

        self._log("\nPTGENDER AFTER TRANSFORM")
        self._log(
            result["PTGENDER"]
            .value_counts(dropna=False)
        )

        return result

    # =========================================================
    # FIT + TRANSFORM
    # =========================================================

    def fit_transform(
        self,
        df: pd.DataFrame,
    ):
        return self.fit(df).transform(df)