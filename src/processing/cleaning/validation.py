import calendar

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "meal_package",
    "needs_parking",
    "room_category",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "booking_channel",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

CAT_COLS = [
    "meal_package",
    "room_category",
    "booking_channel",
]

NUM_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "needs_parking",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

INT_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "needs_parking",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "special_requests",
]

NON_NEG_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

BINARY_COLS = [
    "needs_parking",
    "is_returning_guest",
]

MIN_YEAR = 1
MIN_MONTH = 1
MAX_MONTH = 12
MIN_DAY = 1
SAFE_DAY = 28

def validate_data(data, valid_cats=None, logs=None):
    check_cols(data)
    data = data[FEATURE_COLS].copy()

    to_num_cols(data, logs)
    clean_cats(data, logs)
    if valid_cats is not None:
        new_cat_to_nan(data, valid_cats, logs)

    neg_to_nan(data, logs)
    decimal_to_nan(data, logs)
    bad_binary_to_nan(data, logs)
    zero_guests_to_nan(data, logs)
    bad_year_to_nan(data, logs)
    bad_month_to_nan(data, logs)
    bad_day_to_nan(data, logs)
    fix_dates(data, logs)
    return data

def log_step(logs, step, rule, col, n_rows):
    if logs is None:
        return
    logs.append(
        {
            "step": step,
            "rule": rule,
            "col": col,
            "n_rows": int(n_rows),
        }
    )


def set_nan(data, hit, col, step, rule, logs):
    data.loc[hit, col] = np.nan
    log_step(logs, step, rule, col, hit.sum())


def check_cols(data):
    missing = []
    double = []
    for col in FEATURE_COLS:
        if col not in data.columns:
            missing.append(col)
        elif list(data.columns).count(col) > 1:
            double.append(col)
    if missing:
        raise ValueError(f"Kolom fitur hilang: {missing}")
    if double:
        raise ValueError(f"Kolom fitur dobel: {double}")


def to_num(data, col, logs):
    nan_before = data[col].isna().sum()
    data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")
    nan_after = data[col].isna().sum()
    log_step(logs, 1, "teks di kolom angka jadi NaN", col, nan_after - nan_before)


def to_num_cols(data, logs):
    for col in NUM_COLS:
        to_num(data, col, logs)


def clean_cat(data, col, logs):
    new_values = []
    n_changed = 0
    for cell in data[col]:
        if pd.isna(cell):
            new_values.append(np.nan)
            continue
        text = str(cell).strip()
        if text != str(cell):
            n_changed += 1
        new_values.append(text)

    data[col] = pd.Series(new_values, index=data.index, dtype=object)
    log_step(logs, 1, "spasi di kategori dibuang", col, n_changed)

    empty_text = data[col] == ""
    set_nan(data, empty_text, col, 1, "teks kosong jadi NaN", logs)


def clean_cats(data, logs):
    for col in CAT_COLS:
        clean_cat(data, col, logs)


def new_cat_to_nan(data, valid_cats, logs):
    for col in CAT_COLS:
        has_value = data[col].notna()
        not_in_train = ~data[col].isin(valid_cats[col])
        hit = has_value & not_in_train
        set_nan(data, hit, col, 1, "kategori baru (gak ada di train) jadi NaN", logs)


def learn_cats(data):
    valid_cats = {}
    for col in CAT_COLS:
        cats = data[col].dropna().unique().tolist()
        if len(cats) == 0:
            raise ValueError(f"Kolom kategori '{col}' kosong semua di train split.")
        valid_cats[col] = cats
    return valid_cats


def neg_to_nan(data, logs):
    for col in NON_NEG_COLS:
        hit = data[col] < 0
        set_nan(data, hit, col, 2, "angka negatif jadi NaN", logs)


def not_int(data, col):
    return data[col].notna() & (data[col] % 1 != 0)


def decimal_to_nan(data, logs):
    for col in INT_COLS:
        hit = not_int(data, col)
        set_nan(data, hit, col, 2, "angka desimal di kolom bulat jadi NaN", logs)


def bad_binary_to_nan(data, logs):
    for col in BINARY_COLS:
        has_value = data[col].notna()
        not_binary = ~data[col].isin([0, 1])
        hit = has_value & not_binary
        set_nan(data, hit, col, 2, "nilai biner selain 0/1 jadi NaN", logs)


def zero_guests_to_nan(data, logs):
    no_adults = data["num_adults"] == 0
    no_kids = data["num_children"].isna() | (data["num_children"] == 0)
    hit = no_adults & no_kids
    set_nan(data, hit, "num_adults", 2, "tamu total 0 -> num_adults jadi NaN", logs)


def bad_year_to_nan(data, logs):
    hit = data["arrival_year"] < MIN_YEAR
    set_nan(data, hit, "arrival_year", 2, "tahun < 1 jadi NaN", logs)


def bad_month_to_nan(data, logs):
    month = data["arrival_month"]
    hit = month.notna() & ((month < MIN_MONTH) | (month > MAX_MONTH))
    set_nan(data, hit, "arrival_month", 2, "bulan di luar 1-12 jadi NaN", logs)


def bad_day_to_nan(data, logs):
    hit = data["arrival_day"] < MIN_DAY
    set_nan(data, hit, "arrival_day", 2, "hari < 1 jadi NaN", logs)


def clip_day(data):
    has_year = data["arrival_year"].notna()
    has_month = data["arrival_month"].notna()
    has_day = data["arrival_day"].notna()
    risky_day = data["arrival_day"] > SAFE_DAY
    cand = has_year & has_month & has_day & risky_day

    years = data.loc[cand, "arrival_year"]
    months = data.loc[cand, "arrival_month"]
    days = data.loc[cand, "arrival_day"]

    new_days = []
    count = 0
    for year, month, day in zip(years, months, days):
        last_day = calendar.monthrange(int(year), int(month))[1]
        if day > last_day:
            new_days.append(float(last_day))
            count += 1
        else:
            new_days.append(day)

    data.loc[cand, "arrival_day"] = new_days
    return count


def fix_dates(data, logs):
    count = clip_day(data)
    log_step(logs, 2, "tanggal gak ada di kalender -> akhir bulan", "arrival_day", count)


def recheck_dates(data, logs):
    count = clip_day(data)
    log_step(logs, 6, "cek ulang tanggal setelah diisi -> akhir bulan", "arrival_day", count)
