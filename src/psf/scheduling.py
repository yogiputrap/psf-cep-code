"""Capacitated slot allocation (Eq. 5-6): one MILP per (region, day), solved with HiGHS through SciPy."""
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

KAPPA = 50  # parcels per courier per day (design assumption)
PI = (0.4, 0.3, 0.3)  # Morning, Noon, Afternoon capacity shares (design assumption)
PARCELS_PER_COURIER_POOL = 35  # divisor used to derive the virtual pool size
EVENING = 3


def courier_pool(df, region_col="region", day_col="day"):
    """Virtual courier pool C_r = ceil(median daily volume of region r / 35)."""
    vol = df.groupby([region_col, day_col]).size().groupby(level=0).median()
    return {r: max(1, math.ceil(v / PARCELS_PER_COURIER_POOL)) for r, v in vol.items()}


def capacities(pool_size, availability=1.0):
    """K_{r,s} = max(1, floor(C_r * availability * kappa * pi_s)) for the three operational slots."""
    return np.array([max(1, math.floor(pool_size * availability * KAPPA * p)) for p in PI], dtype=float)


def allowed_mask(hist_slot, variant):
    """Boolean n x 4 matrix of permitted slots. Evening is always permitted."""
    s = np.arange(4)[None, :]
    h = np.asarray(hist_slot)[:, None]
    if variant == "unconstrained":
        m = np.ones((len(h), 4), bool)
    elif variant == "one_slot":
        m = np.abs(s - h) <= 1
    elif variant == "not_earlier":
        m = s >= h
    else:
        raise ValueError(variant)
    m[:, EVENING] = True
    return m


def solve_day(P, hist_slot, cap, variant="one_slot"):
    """Minimise sum P[i,s] x[i,s] s.t. each shipment gets one slot and operational slots respect cap."""
    n = P.shape[0]
    nv = 4 * n
    rows = np.repeat(np.arange(n), 4)
    cols = np.arange(nv)
    assign = coo_matrix((np.ones(nv), (rows, cols)), shape=(n, nv)).tocsr()
    cap_rows = np.tile(np.arange(3), n)
    cap_cols = np.concatenate([[4 * i + s for s in range(3)] for i in range(n)])
    capm = coo_matrix((np.ones(3 * n), (cap_rows, cap_cols)), shape=(3, nv)).tocsr()
    ub = allowed_mask(hist_slot, variant).astype(float).ravel()
    res = milp(
        c=P.ravel(),
        constraints=[LinearConstraint(assign, 1, 1), LinearConstraint(capm, 0, cap)],
        integrality=np.ones(nv),
        bounds=Bounds(np.zeros(nv), ub),
    )
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return res.x.reshape(n, 4).argmax(axis=1)


def schedule(df, P, variant="one_slot", availability=1.0, pool=None):
    """Assign slots for every (region, day) in df. Returns the array of assigned slots aligned to df rows.

    variant: historical | greedy | one_slot | unconstrained | not_earlier
    """
    assigned = df["slot_hist"].to_numpy().copy()
    if variant == "historical":
        return assigned
    if variant == "greedy":  # minimum-risk operational slot, no capacity (infeasible lower bound)
        return P[:, :3].argmin(axis=1)
    pool = courier_pool(df) if pool is None else pool
    for (r, _), idx in df.groupby(["region", "day"]).indices.items():
        cap = capacities(pool[r], availability)
        assigned[idx] = solve_day(P[idx], df["slot_hist"].to_numpy()[idx], cap, variant)
    return assigned


def evaluate(df, P, assigned, availability=1.0, pool=None):
    """Expected failure rate under the scoring matrix P, capacity violations and Evening share."""
    n = len(df)
    exp_fail = P[np.arange(n), assigned].mean()
    pool = courier_pool(df) if pool is None else pool
    over = 0.0
    for (r, _), idx in df.groupby(["region", "day"]).indices.items():
        load = np.bincount(assigned[idx], minlength=4)[:3]
        over += np.maximum(0.0, load - capacities(pool[r], availability)).sum()
    return {"expected_fail_rate": float(exp_fail), "over_capacity": float(over),
            "evening_share": float((assigned == EVENING).mean())}
