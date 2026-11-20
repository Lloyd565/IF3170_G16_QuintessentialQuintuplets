from sklearn.preprocessing import StandardScaler



def fit_scaler(X_train, num_cols):
    return StandardScaler().fit(X_train[num_cols])

def scale_features(X, scaler, num_cols):
    hasil = X.copy()
    hasil[num_cols] = scaler.transform(X[num_cols])
    return hasil
