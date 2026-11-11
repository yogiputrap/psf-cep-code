"""End-to-end demo on synthetic data: features -> temporal hold-out -> risk matrix -> capacitated MILP.

Usage: python scripts/run_demo.py [--n 30000] [--seed 0] [--out results/demo]
The numbers it prints describe the synthetic data only; they are not the paper's results.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psf import metrics, model, scheduling, synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    df = model.add_history(synth.make_synthetic(args.n, args.seed), lag=1)
    train, test = model.temporal_split(df, test_days=21)
    test = test.reset_index(drop=True)
    print(f"synthetic shipments: {len(df)}  train {len(train)}  test {len(test)}  "
          f"failure rate {df['y'].mean():.3f} (test {test['y'].mean():.3f})")

    rm = model.RiskModel(seed=args.seed).fit(train, train["y"])
    p = rm.predict(test)
    auc, ci = metrics.block_bootstrap_auc(test["y"], p, test["day"], B=args.boot)
    print(f"ROC-AUC {auc:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f}), ECE {metrics.expected_calibration_error(test['y'], p):.4f}"
          f"  [backend: {rm.backend}]")

    P = rm.risk_matrix(test)
    truth = test[[f"p_true_{s}" for s in range(4)]].to_numpy()
    base = None
    rows = []
    for variant, avail in [("historical", 1.0), ("greedy", 1.0), ("one_slot", 1.0), ("unconstrained", 1.0),
                           ("not_earlier", 1.0), ("one_slot", 0.7), ("one_slot", 0.5)]:
        a = scheduling.schedule(test, P, variant, avail)
        m = scheduling.evaluate(test, P, a, avail)
        t = scheduling.evaluate(test, truth, a, avail)  # synthetic truth, available only for synthetic data
        hist_avail = scheduling.evaluate(test, P, test["slot_hist"].to_numpy(), avail)
        rows.append({"variant": variant, "availability": avail, "model_fail": m["expected_fail_rate"],
                     "model_reduction_pct": 100 * (1 - m["expected_fail_rate"] / hist_avail["expected_fail_rate"]),
                     "true_fail": t["expected_fail_rate"], "over_capacity": m["over_capacity"],
                     "evening_share": m["evening_share"]})
        if variant == "historical":
            base = m["expected_fail_rate"]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(res.round(4).to_string(index=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "demo_scheduling.csv", index=False)
    print(f"wrote {out / 'demo_scheduling.csv'}")


if __name__ == "__main__":
    main()
