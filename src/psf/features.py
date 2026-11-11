"""Leakage-safe features: slot from hour, lagged smoothed history rates (Algorithm 1), holiday calendar."""
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAY_CSV = Path(__file__).resolve().parents[2] / "data" / "holidays_2025.csv"


def slot_from_hour(hour):
    """Morning (<12), Noon (12-14), Afternoon (15-17), Evening (>=18) as indices 0..3."""
    h = np.asarray(hour)
    return np.select([h < 12, h < 15, h < 18], [0, 1, 2], default=3)


def lagged_history(df, key, lag=1, alpha=20.0, day_col="day", y_col="y", prefix=None):
    """Smoothed failure rate and attempt count of entity `key`, using outcomes up to day d - lag only.

    rate = (F + alpha * g) / (n + alpha), with F, n the failures and attempts of the entity on days <= d - lag
    and g the global failure rate on days <= d - lag (Eq. 4). Where no earlier day exists, g falls back to the
    rate of the first day, as in Algorithm 1.
    Returns a DataFrame aligned to df.index with columns `<prefix>_rate` and `<prefix>_n`.
    """
    prefix = prefix or key
    d = df[[key, day_col, y_col]].copy()
    d["_q"] = d[day_col] - lag

    ent = d.groupby([key, day_col])[y_col].agg(F="sum", n="count").reset_index().sort_values(day_col)
    ent[["F", "n"]] = ent.groupby(key)[["F", "n"]].cumsum()

    glob = d.groupby(day_col)[y_col].agg(F="sum", n="count").sort_index().cumsum()
    first_day_rate = d.loc[d[day_col] == d[day_col].min(), y_col].mean()
    glob_rate = (glob["F"] / glob["n"]).rename("g").reset_index()

    left = d.reset_index().sort_values("_q")
    out = pd.merge_asof(left, ent.rename(columns={day_col: "_q"}), on="_q", by=key, direction="backward")
    out = pd.merge_asof(out.sort_values("_q"), glob_rate.rename(columns={day_col: "_q"}), on="_q", direction="backward")
    out["F"] = out["F"].fillna(0.0)
    out["n"] = out["n"].fillna(0.0)
    out["g"] = out["g"].fillna(first_day_rate)
    out[f"{prefix}_rate"] = (out["F"] + alpha * out["g"]) / (out["n"] + alpha)
    out[f"{prefix}_n"] = out["n"]
    out = out.set_index("index").sort_index()
    return out[[f"{prefix}_rate", f"{prefix}_n"]].reindex(df.index)


def load_holidays(path=HOLIDAY_CSV):
    h = pd.read_csv(path, parse_dates=["date"])
    return h


def calendar_features(dates, holidays=None, lebaran=("2025-03-28", "2025-04-08"), eid="2025-03-31"):
    """Six calendar indicators for each unique date (Sundays count as off days, Saturdays as working days)."""
    holidays = load_holidays() if holidays is None else holidays
    off = set(pd.to_datetime(holidays["date"]))
    u = pd.Series(pd.to_datetime(np.unique(pd.to_datetime(dates))))

    def is_off(ts):
        return (ts in off) or (ts.dayofweek == 6)

    rows = []
    for d in u:
        k = 1
        while is_off(d + pd.Timedelta(days=k)):
            k += 1
        rows.append(
            {
                "day": d,
                "cal_is_holiday": float(d in off),
                "cal_next_day_off": float(is_off(d + pd.Timedelta(days=1))),
                "cal_prev_day_off": float(is_off(d - pd.Timedelta(days=1))),
                "cal_days_to_next_workday": float(k),
                "cal_lebaran_window": float(pd.Timestamp(lebaran[0]) <= d <= pd.Timestamp(lebaran[1])),
                "cal_days_from_lebaran": float(np.clip((d - pd.Timestamp(eid)).days, -20, 30)),
            }
        )
    return pd.DataFrame(rows)
