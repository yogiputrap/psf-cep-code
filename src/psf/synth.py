"""Synthetic shipments with a planted slot-aware failure mechanism.

No operator record is used. Only a few published aggregates from the paper set the scale: the study window,
the regional split, the historical slot shares and the slot-wise failure rates. Every other number below is an
arbitrary modelling choice, so results on this data illustrate the pipeline and are not the paper's results.

Because the data-generating process is known, the dataset also carries the true failure probability of each
shipment in each slot (columns p_true_0..p_true_3), which allows the model-based counterfactual to be checked
against the synthetic truth.
"""
import numpy as np
import pandas as pd

from . import features

SLOT_RATES = np.array([0.0935, 0.1242, 0.2349, 0.5924])  # published failure rate by slot (Morning..Evening)
HIST_SLOT_SHARE = np.array([0.196, 0.380, 0.317, 0.107])  # historical slot shares (arbitrary, close to the paper)
REGIONS = np.array(["RA", "RB", "RC"])
REGION_P = np.array([0.9733, 0.0063, 0.0204])
TARGET_RATE = 0.194


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic(n=30000, seed=0, start="2025-03-15", end="2025-06-25"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    w = np.where(days.dayofweek == 6, 0.15, 1.0) * rng.uniform(0.7, 1.3, len(days))
    day = rng.choice(len(days), size=n, p=w / w.sum())
    date = days[day]

    n_rec, n_snd, n_zone = int(n * 0.45), 600, 60
    recipient = rng.integers(0, n_rec, n)
    sender = rng.integers(0, n_snd, n)
    zone = rng.integers(0, n_zone, n)
    rec_z = rng.normal(0, 1.0, n_rec)[recipient]
    snd_z = rng.normal(0, 0.5, n_snd)[sender]
    zone_z = rng.normal(0, 0.5, n_zone)[zone]

    df = pd.DataFrame(
        {
            "ship_id": np.arange(n),
            "attempt_date": date,
            "region": rng.choice(REGIONS, n, p=REGION_P),
            "service": rng.choice(["S1", "S2", "S3"], n, p=[0.594, 0.208, 0.198]),
            "loc_type": rng.choice(["OT1", "OT2", "OT3"], n, p=[0.40, 0.30, 0.30]),
            "weight_kg": rng.gamma(2.0, 0.6, n).round(2),
            "sla_days": rng.integers(1, 6, n),
            "lead_days": rng.integers(0, 6, n),
            "cod": (rng.random(n) < 0.0638).astype(int),
            "addr_len": rng.integers(15, 120, n),
            "recipient_id": recipient,
            "sender_id": sender,
            "zone": zone,
        }
    )
    df["day"] = (df["attempt_date"] - days[0]).dt.days
    df["weekday"] = df["attempt_date"].dt.dayofweek
    hist_slot = rng.choice(4, size=n, p=HIST_SLOT_SHARE)
    df["slot_hist"] = hist_slot

    cal = features.calendar_features(df["attempt_date"]).rename(columns={"day": "attempt_date"})
    df = df.merge(cal, on="attempt_date", how="left")

    base = (
        0.9 * rec_z
        + 0.6 * snd_z
        + zone_z
        + 0.5 * df["cal_is_holiday"].to_numpy()
        + 0.3 * (df["weekday"].to_numpy() >= 5)
        + 0.4 * df["cod"].to_numpy()
        - 0.1 * df["lead_days"].to_numpy()
        + 0.004 * (df["addr_len"].to_numpy() - 60)
    )
    slot_int = _logit(SLOT_RATES)

    def mean_rate(shift):
        return _sigmoid(slot_int[hist_slot] + base + shift).mean()

    lo, hi = -4.0, 4.0
    for _ in range(60):  # bisection on a global intercept so the observed rate matches TARGET_RATE
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_rate(mid) < TARGET_RATE else (lo, mid)
    shift = (lo + hi) / 2

    for s in range(4):
        df[f"p_true_{s}"] = _sigmoid(slot_int[s] + base + shift)
    p_obs = df[[f"p_true_{s}" for s in range(4)]].to_numpy()[np.arange(n), hist_slot]
    df["y"] = (rng.random(n) < p_obs).astype(int)
    return df.sort_values(["day", "ship_id"]).reset_index(drop=True)
