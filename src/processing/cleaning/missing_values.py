import numpy as np

from .validation import log_step

MIN_GROUP_ROWS = 30

MODE_COLS = [
    "num_adults",
    "num_children",
    "special_requests",
    "needs_parking",
    "meal_package",
    "room_category",
    "booking_channel",
    "weekend_nights",
    "weekday_nights",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
]

MEDIAN_COLS = [
    "days_before_arrival",
]


def get_mode(data, col):
    modes = data[col].mode(dropna=True)
    if len(modes) == 0:
        raise ValueError(f"Kolom '{col}' kosong semua di train split; gak bisa diisi.")
    value = modes.iloc[0]
    if isinstance(value, str):
        return value
    return float(value)


def get_median(data, col):
    med = data[col].median()
    if np.isnan(med):
        raise ValueError(f"Kolom '{col}' kosong semua di train split; gak bisa diisi.")
    return float(med)


def room_month_price_med(data):
    has_price = data["nightly_rate_idr"].notna()
    has_room = data["room_category"].notna()
    has_month = data["arrival_month"].notna()
    complete = data.loc[has_price & has_room & has_month]

    groups = complete.groupby(["room_category", "arrival_month"])["nightly_rate_idr"]
    n_rows = groups.count()
    med = groups.median()

    result = {}
    for (room, month), count in n_rows.items():
        if count >= MIN_GROUP_ROWS:
            result[(room, month)] = float(med[(room, month)])
    return result


def room_price_med(data):
    has_price = data["nightly_rate_idr"].notna()
    has_room = data["room_category"].notna()
    complete = data.loc[has_price & has_room]

    med = complete.groupby("room_category")["nightly_rate_idr"].median()

    result = {}
    for room, value in med.items():
        result[room] = float(value)
    return result


def month_price_med(data):
    has_price = data["nightly_rate_idr"].notna()
    has_month = data["arrival_month"].notna()
    complete = data.loc[has_price & has_month]

    med = complete.groupby("arrival_month")["nightly_rate_idr"].median()

    result = {}
    for month, value in med.items():
        result[month] = float(value)
    return result


def fill_col(data, col, value, how, logs):
    is_nan = data[col].isna()
    data.loc[is_nan, col] = value
    log_step(logs, 5, f"NaN diisi {how}", col, is_nan.sum())


class MissingValueImputer:
    def __init__(self):
        self.fill_values_ = None
        self.room_month_med_ = None
        self.room_med_ = None
        self.month_med_ = None
        self.global_med_ = None

    def fit(self, data):
        fill_values = {}
        for col in MODE_COLS:
            fill_values[col] = get_mode(data, col)
        for col in MEDIAN_COLS:
            fill_values[col] = float(np.round(get_median(data, col)))

        self.fill_values_ = fill_values
        self.room_month_med_ = room_month_price_med(data)
        self.room_med_ = room_price_med(data)
        self.month_med_ = month_price_med(data)
        self.global_med_ = get_median(data, "nightly_rate_idr")
        return self

    def fill_price_room_month(self, data, logs):
        count = 0
        for (room, month), med in self.room_month_med_.items():
            no_price = data["nightly_rate_idr"].isna()
            same_room = data["room_category"] == room
            same_month = data["arrival_month"] == month
            hit = no_price & same_room & same_month
            data.loc[hit, "nightly_rate_idr"] = med
            count += hit.sum()
        log_step(logs, 5, "NaN diisi median kamar di bulan x", "nightly_rate_idr", count)

    def fill_price_room(self, data, logs):
        count = 0
        for room, med in self.room_med_.items():
            no_price = data["nightly_rate_idr"].isna()
            same_room = data["room_category"] == room
            hit = no_price & same_room
            data.loc[hit, "nightly_rate_idr"] = med
            count += hit.sum()
        log_step(logs, 5, "NaN diisi median kamar", "nightly_rate_idr", count)

    def fill_price_month(self, data, logs):
        count = 0
        for month, med in self.month_med_.items():
            no_price = data["nightly_rate_idr"].isna()
            same_month = data["arrival_month"] == month
            hit = no_price & same_month
            data.loc[hit, "nightly_rate_idr"] = med
            count += hit.sum()
        log_step(logs, 5, "NaN diisi median semua kamar di bulan x", "nightly_rate_idr", count)

    def fill_price_global(self, data, logs):
        hit = data["nightly_rate_idr"].isna()
        data.loc[hit, "nightly_rate_idr"] = self.global_med_
        log_step(logs, 5, "NaN diisi median global", "nightly_rate_idr", hit.sum())

    def fill_price(self, data, logs):
        self.fill_price_room_month(data, logs)
        self.fill_price_room(data, logs)
        self.fill_price_month(data, logs)
        self.fill_price_global(data, logs)

    def transform(self, data, logs=None):
        if self.fill_values_ is None:
            raise RuntimeError("MissingValueImputer belum di-fit.")

        data = data.copy()

        self.fill_price(data, logs)

        for col in MODE_COLS:
            fill_col(data, col, self.fill_values_[col], "modus", logs)
        for col in MEDIAN_COLS:
            fill_col(data, col, self.fill_values_[col], "median", logs)
        return data
