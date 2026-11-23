import pandas as pd

from .cleaning.cleaning import DataCleaner
from .cleaning.validation import CAT_COLS, validate_data
from .transformation.encoding import encode_features, fit_encoder


class DTLPreprocessor:
    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        self.cleaner = DataCleaner().fit(X)
        features = self._features(X)
        self.encoder = fit_encoder(features, CAT_COLS)
        encoded = encode_features(features, self.encoder, CAT_COLS).astype(float)
        self.columns = encoded.columns.tolist()
        return encoded

    def transform(self, X):
        features = self._features(X)
        return encode_features(features, self.encoder, CAT_COLS).loc[:, self.columns].astype(float)

    def _features(self, X):
        features = self.cleaner.transform(X)
        validated = validate_data(X)
        for column in validated.columns:
            features[column + "_missing"] = validated[column].isna().astype(float)
        for column in CAT_COLS:
            features[column] = validated[column].fillna("__missing__")
        arrival = pd.to_datetime({
            "year": features.arrival_year,
            "month": features.arrival_month,
            "day": features.arrival_day,
        })
        features["booking_ordinal"] = (
            (arrival - pd.Timestamp("2000-01-01")).dt.days - features.days_before_arrival
        )
        return features
