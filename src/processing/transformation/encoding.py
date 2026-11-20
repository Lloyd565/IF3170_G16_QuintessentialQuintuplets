import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def fit_encoder(X_train, cat_cols):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[cat_cols])
    return encoder


def encode_features(X, encoder, cat_cols):
    encoded = encoder.transform(X[cat_cols])
    encoded_features = pd.DataFrame(
        encoded,
        index=X.index,
        columns=encoder.get_feature_names_out(cat_cols),
    )
    numeric_features = X.drop(columns=cat_cols)
    return pd.concat([numeric_features, encoded_features], axis=1)