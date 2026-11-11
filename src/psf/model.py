"""Slot-aware risk model: one classifier takes the slot as a feature; the risk matrix substitutes the slot."""
import numpy as np
import pandas as pd

from . import SLOTS
from .features import lagged_history

CAT = ["service", "loc_type", "region", "slot"]
NUM = [
    "weight_kg", "sla_days", "lead_days", "cod", "addr_len", "weekday", "zone",
    "recipient_id_rate", "recipient_id_n", "sender_id_rate", "sender_id_n", "zone_rate", "zone_n",
    "cal_is_holiday", "cal_next_day_off", "cal_prev_day_off", "cal_days_to_next_workday",
    "cal_lebaran_window", "cal_days_from_lebaran",
]
FEATURES = CAT + NUM


def add_history(df, lag=1, alpha=20.0):
    """Add lagged history rates/counts for recipient, sender and destination zone (outcomes up to d - lag)."""
    out = df.copy()
    for key in ("recipient_id", "sender_id", "zone"):
        h = lagged_history(out, key, lag=lag, alpha=alpha)
        out[h.columns] = h
    return out


def temporal_split(df, test_days=21):
    cut = df["day"].max() - test_days + 1
    return df[df["day"] < cut].copy(), df[df["day"] >= cut].copy()


class RiskModel:
    """CatBoost when installed (native categoricals), otherwise scikit-learn gradient boosting."""

    def __init__(self, backend="auto", iterations=300, seed=0):
        self.iterations, self.seed = iterations, seed
        if backend == "auto":
            try:
                import catboost  # noqa: F401
                backend = "catboost"
            except ImportError:
                backend = "sklearn"
        self.backend = backend

    def _x(self, df, slot=None):
        x = df[[c for c in FEATURES if c != "slot"]].copy()
        x["slot"] = np.asarray(df["slot_hist"] if slot is None else slot).astype(int).astype(str)
        return x[FEATURES]

    def fit(self, df, y):
        x = self._x(df)
        if self.backend == "catboost":
            from catboost import CatBoostClassifier

            self.m = CatBoostClassifier(iterations=self.iterations, learning_rate=0.06, depth=6,
                                        random_seed=self.seed, verbose=False, thread_count=1,
                                        allow_writing_files=False)
            self.m.fit(x, y, cat_features=CAT)
        else:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import OrdinalEncoder

            ct = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
                                   remainder="passthrough")
            self.m = make_pipeline(ct, HistGradientBoostingClassifier(max_iter=self.iterations, learning_rate=0.06,
                                                                      random_state=self.seed))
            self.m.fit(x, y)
        return self

    def predict(self, df, slot=None):
        return self.m.predict_proba(self._x(df, slot))[:, 1]

    def risk_matrix(self, df):
        """P[i, s] = P(fail | x_i, slot = s) for the four slots."""
        return np.column_stack([self.predict(df, np.full(len(df), s)) for s in range(len(SLOTS))])
