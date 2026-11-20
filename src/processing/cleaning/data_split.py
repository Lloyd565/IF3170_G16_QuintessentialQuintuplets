from sklearn.model_selection import train_test_split

ID_COL = "id"
TARGET_COL = "is_canceled"
VAL_SIZE = 0.2
SEED = 42


def split_data(train, test):
    X = train.drop(columns=[ID_COL, TARGET_COL]).copy()
    y = train[TARGET_COL].copy()

    X_test = test.drop(columns=[ID_COL]).copy()
    id_test = test[ID_COL].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED
    )
    keep = ~X_train.duplicated()
    X_train, y_train = X_train[keep], y_train[keep]

    return X_train, X_val, y_train, y_val, X_test, id_test
