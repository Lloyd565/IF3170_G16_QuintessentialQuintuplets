from ..cleaning.validation import BINARY_COLS, CAT_COLS, FEATURE_COLS, NUM_COLS
from .encoding import encode_features, fit_encoder
from .scaling import fit_scaler, scale_features


def transformation(X, transformer=None, fit=False):
    if fit:
        transformer = {}
    elif transformer is None:
        raise ValueError("Butuh transformer yang sudah di-fit.")

    if not X.columns.is_unique:
        raise ValueError("Nama kolom fitur harus unik.")

    expected_cols = FEATURE_COLS if fit else transformer["input_cols"]
    missing_cols = [col for col in expected_cols if col not in X.columns]
    extra_cols = [col for col in X.columns if col not in expected_cols]
    if missing_cols or extra_cols:
        raise ValueError(f"Kolom fitur tidak sesuai. Kurang: {missing_cols}. Ekstra: {extra_cols}.")

    if X.isna().any().any():
        raise ValueError("Masih ada nilai kosong. Jalankan cleaning sebelum transformasi.")

    if not X[BINARY_COLS].isin([0, 1]).all().all():
        raise ValueError("Fitur biner harus berisi 0 atau 1 setelah cleaning.")

    if fit:
        num_cols = [col for col in NUM_COLS if col not in BINARY_COLS]
        transformer = {
            "encoder": fit_encoder(X, CAT_COLS),
            "scaler": fit_scaler(X, num_cols),
            "num_cols": num_cols,
            "input_cols": X.columns.tolist(),
        }

    encoded = encode_features(X, transformer["encoder"], CAT_COLS)
    scaled = scale_features(encoded, transformer["scaler"], transformer["num_cols"])

    if fit:
        transformer["columns"] = scaled.columns.tolist()
    return scaled.loc[:, transformer["columns"]], transformer
