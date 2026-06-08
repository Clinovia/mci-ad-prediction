import numpy as np
import pandas as pd


DEMOGRAPHIC_FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
]

GENETIC_FEATURES = [
    "APOE4",
]

COGNITIVE_FEATURES = [
    "MMSE",
]

FUNCTIONAL_FEATURES = [
    "EcogSPTotal",
]

MEMORY_FEATURES = [
    "RAVLT_immediate",
]


BASE_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + MEMORY_FEATURES
    + COGNITIVE_FEATURES
    + FUNCTIONAL_FEATURES
)


GENDER_MAP = {
    "Male": 1,
    "Female": 0,
}


class Stage1Preprocessor:

    def __init__(
        self,
        selected_features=None,
        verbose=False,
    ):

        self.selected_features = (
            selected_features
            if selected_features is not None
            else BASE_FEATURES
        )

        self.verbose = verbose

        self.imputation_values = {}
        self.feature_order = None

    def _normalize_gender(self, series):

        numeric = pd.to_numeric(
            series,
            errors="coerce"
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
            errors="coerce"
        )

    def fit(self, df):

        model_df = df.copy()

        if "PTGENDER" in self.selected_features:

            model_df["PTGENDER"] = (
                self._normalize_gender(
                    model_df["PTGENDER"]
                )
            )

            if (
                model_df["PTGENDER"]
                .notna()
                .sum()
                == 0
            ):
                raise ValueError(
                    "PTGENDER became all NaN"
                )

            gender_median = (
                model_df["PTGENDER"]
                .median()
            )

            if np.isnan(gender_median):
                gender_median = 0.0

            self.imputation_values[
                "PTGENDER"
            ] = float(
                gender_median
            )

        for col in self.selected_features:

            if col == "PTGENDER":
                continue

            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce"
            )

            median = (
                model_df[col]
                .median()
            )

            if np.isnan(median):
                median = 0.0

            self.imputation_values[
                col
            ] = float(median)

        self.feature_order = (
            self.selected_features.copy()
        )

        if self.verbose:

            print(
                "\nFEATURE ORDER"
            )
            print(
                self.feature_order
            )

        return self

    def transform(self, df):

        model_df = df.copy()

        if "PTGENDER" in self.selected_features:

            model_df["PTGENDER"] = (
                self._normalize_gender(
                    model_df["PTGENDER"]
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

        for col in self.selected_features:

            if col == "PTGENDER":
                continue

            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce"
            )

            model_df[col] = (
                model_df[col]
                .fillna(
                    self.imputation_values[
                        col
                    ]
                )
            )

        return model_df[
            self.feature_order
        ]

    def fit_transform(self, df):

        return (
            self.fit(df)
            .transform(df)
        )