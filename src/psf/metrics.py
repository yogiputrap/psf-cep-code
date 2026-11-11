"""ROC-AUC with day-level block bootstrap (Eq. 11), paired differences, calibration error, workload balance."""
import numpy as np
from sklearn.metrics import roc_auc_score


def _day_index(days):
    days = np.asarray(days)
    ud = np.unique(days)
    return ud, {d: np.where(days == d)[0] for d in ud}


def block_bootstrap_auc(y, p, days, B=1000, seed=0):
    """AUC and 95% CI from resampling whole days with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    ud, idx = _day_index(days)
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        pick = rs.choice(len(ud), len(ud), replace=True)
        take = np.concatenate([idx[ud[i]] for i in pick])
        if y[take].min() != y[take].max():
            vals.append(roc_auc_score(y[take], p[take]))
    return float(roc_auc_score(y, p)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_auc_difference(y, p_a, p_b, days, B=1000, seed=0):
    """AUC_A - AUC_B on identical day resamples; 'significant' if the 95% CI excludes zero."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    ud, idx = _day_index(days)
    rs = np.random.RandomState(seed)
    diffs = []
    for _ in range(B):
        pick = rs.choice(len(ud), len(ud), replace=True)
        take = np.concatenate([idx[ud[i]] for i in pick])
        if y[take].min() != y[take].max():
            diffs.append(roc_auc_score(y[take], p_a[take]) - roc_auc_score(y[take], p_b[take]))
    d0 = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return float(d0), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def expected_calibration_error(y, p, bins=10):
    """Mean absolute gap between predicted and observed rate over equal-count deciles."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(p[c].mean() - y[c].mean()) for c in chunks))


def workload_balance_index(loads, caps):
    """Jain's index on utilisation load/capacity, rescaled to 0-100% (Eq. 10)."""
    u = np.asarray(loads, float) / np.asarray(caps, float)
    k = len(u)
    j = u.sum() ** 2 / (k * (u**2).sum())
    return float((j - 1 / k) / (1 - 1 / k) * 100)
