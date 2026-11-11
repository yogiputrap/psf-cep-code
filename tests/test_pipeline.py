import numpy as np
import pandas as pd

from psf import features, metrics, model, scheduling, synth


def _small(n=6000, seed=1):
    return synth.make_synthetic(n, seed)


def test_slot_boundaries():
    assert features.slot_from_hour([8, 11, 12, 14, 15, 17, 18, 22]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_history_is_leakage_safe():
    """Changing outcomes on day d or later must not change the history features of day d (lag 1)."""
    df = _small()
    base = features.lagged_history(df, "recipient_id", lag=1)
    d0 = df["day"].max() - 5
    alt = df.copy()
    alt.loc[alt["day"] >= d0, "y"] = 1 - alt.loc[alt["day"] >= d0, "y"]
    pert = features.lagged_history(alt, "recipient_id", lag=1)
    early = df["day"] <= d0  # day d0 sees outcomes only up to d0 - 1, which were not perturbed
    np.testing.assert_allclose(base.loc[early].to_numpy(), pert.loc[early].to_numpy())
    assert not np.allclose(base.loc[~early].to_numpy(), pert.loc[~early].to_numpy())


def test_history_matches_naive_definition():
    df = _small(2000, 2)
    h = features.lagged_history(df, "sender_id", lag=2, alpha=20.0)
    for i in [10, 500, 1500, 1999]:
        r = df.loc[i]
        past = df[df["day"] <= r["day"] - 2]
        same = past[past["sender_id"] == r["sender_id"]]
        g = past["y"].mean() if len(past) else df.loc[df["day"] == df["day"].min(), "y"].mean()
        want = (same["y"].sum() + 20 * g) / (len(same) + 20)
        assert abs(h.loc[i, "sender_id_rate"] - want) < 1e-9
        assert h.loc[i, "sender_id_n"] == len(same)


def test_holiday_calendar_matches_decree():
    """15 March - 25 June 2025 holds 10 national holiday dates and 8 joint-leave days (SKB 1017/2/2 of 2024)."""
    h = features.load_holidays()
    w = h[(h["date"] >= "2025-03-15") & (h["date"] <= "2025-06-25")]
    assert w["type"].value_counts().to_dict() == {"national": 10, "joint_leave": 8}
    assert (h["date"] == "2025-06-27").any()  # 1 Muharam: listed, but after the last attempt day
    assert (h[h["date"] == "2025-08-18"]["source"] == "SKB-2025-amendment").all()


def test_capacity_formula():
    # C_r = 30 -> K = floor(30 * 50 * (0.4, 0.3, 0.3)) = (600, 450, 450); a pool of 1 gives (20, 15, 15)
    assert scheduling.capacities(30).tolist() == [600, 450, 450]
    assert scheduling.capacities(1).tolist() == [20, 15, 15]
    assert scheduling.capacities(1, availability=0.5).tolist() == [max(1, 10), max(1, 7), max(1, 7)]


def test_milp_feasible_and_ordered():
    df = model.add_history(_small(8000, 3))
    train, test = model.temporal_split(df, 21)
    test = test.reset_index(drop=True)
    P = test[[f"p_true_{s}" for s in range(4)]].to_numpy()  # use the true matrix: no model noise
    pool = scheduling.courier_pool(test)
    one = scheduling.schedule(test, P, "one_slot")
    unc = scheduling.schedule(test, P, "unconstrained")
    ne = scheduling.schedule(test, P, "not_earlier")
    for a in (one, unc, ne):
        assert scheduling.evaluate(test, P, a, pool=pool)["over_capacity"] == 0
    mask1 = scheduling.allowed_mask(test["slot_hist"].to_numpy(), "one_slot")
    assert mask1[np.arange(len(test)), one].all()
    assert (ne >= test["slot_hist"].to_numpy()).all()
    f = lambda a: scheduling.evaluate(test, P, a, pool=pool)["expected_fail_rate"]  # noqa: E731
    assert f(unc) <= f(one) + 1e-9  # unconstrained feasible set contains the one-slot set
    assert f(unc) <= f(ne) + 1e-9  # not-earlier is also a subset of the unconstrained set
    greedy = scheduling.schedule(test, P, "greedy")
    assert f(greedy) <= f(unc) + 1e-9  # greedy ignores capacity: lower bound


def test_block_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(21), 200)
    y = rng.integers(0, 2, len(days))
    p = np.clip(0.3 * y + rng.random(len(days)) * 0.7, 0, 1)
    auc, (lo, hi) = metrics.block_bootstrap_auc(y, p, days, B=200)
    assert lo <= auc <= hi
    d, (dl, dh) = metrics.paired_auc_difference(y, p, p, days, B=50)
    assert d == 0 and dl == 0 and dh == 0


def test_workload_balance_bounds():
    assert metrics.workload_balance_index([10, 10, 10], [10, 10, 10]) == 100.0
    assert metrics.workload_balance_index([30, 0, 0], [10, 10, 10]) == 0.0
