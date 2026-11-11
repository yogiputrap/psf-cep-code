# Predictive scheduling framework: reference implementation (synthetic data only)

Code accompanying the manuscript *Hidden Failures and Upstream Limits: Slot-Aware Risk Prediction and Scheduling for First-Attempt Delivery in Courier, Express and Parcel Logistics*.

## Data policy

The shipment data of the case study are confidential operational records of a national courier company in Indonesia and contain
personal data (names, phone numbers, addresses). **They are not in this repository and cannot be shared.**
Everything here runs on one of three kinds of public or artificial input:

| Input | Where | What it is |
|---|---|---|
| Synthetic shipments | `src/psf/synth.py` (generated on the fly) | Artificial rows with a planted slot-aware failure mechanism. Only a few published aggregates set its scale (study window, regional split, slot-wise failure rates). It is not a masked copy of real data. |
| Holiday calendar | `data/holidays_2025.csv` | Dates from the 2025 joint decree No. 1017/2/2 of 2024 and its amendment No. 933/1/3 of 2025 (public documents). |
| Aggregate results | `results/aggregate/*.csv` | Numbers transcribed from the paper's tables. No row-level data. |

`.gitignore` blocks `data/raw/`, `data/private/`, parquet, pickle and `.env` files so that company data cannot be
committed by accident. Please keep it that way.

## What is implemented

| Paper section | Module |
|---|---|
| III-A target and slots | `psf/features.py::slot_from_hour`; label `y` in `psf/synth.py` |
| III-C slot-aware risk model, risk matrix | `psf/model.py::RiskModel` (CatBoost, or scikit-learn fallback) |
| III-D leakage-safe history (Eq. 4, Algorithm 1), calendar | `psf/features.py::lagged_history`, `calendar_features` |
| III-E capacitated allocation (Eq. 5-6, Algorithm 2): historical, greedy, one-slot, unconstrained, not-earlier | `psf/scheduling.py` (HiGHS through SciPy `milp`) |
| III-F metrics: day-level block bootstrap, paired differences, ECE, WBI | `psf/metrics.py` |

**Not included** (these depend on the company's raw fields): the PostgreSQL star-schema warehouse and its ETL,
the address and phone quality indicators, the predictive R-ESR, the TF-IDF and native CatBoost text features, the
ten-architecture comparison, and the Streamlit prototype. The paper's numbers cannot be reproduced from this
repository; the pipeline, the leakage controls and the optimisation model can be inspected and run.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions used for the paper (CPU only)
pytest -q                                # 8 tests: leakage safety, capacity formula, MILP feasibility, calendar counts
python scripts/run_demo.py               # synthetic end-to-end run, about one minute
```

`scripts/run_demo.py` trains on the first days, tests on the last 21 days, builds the risk matrix and runs the
scheduling variants and courier-availability levels. It reports the expected failure rate under the model and
under the synthetic ground truth, which is possible only because the data are synthetic. Results are written to
`results/demo/` (git-ignored). The demo's numbers describe the synthetic data only.

## Tests worth reading

* `test_history_is_leakage_safe`: flipping outcomes on day *d* and later does not change any history feature of day *d* (lag 1).
* `test_history_matches_naive_definition`: the vectorised history equals a direct implementation of Eq. 4.
* `test_holiday_calendar_matches_decree`: 10 national holiday dates and 8 joint-leave days fall between 15 March and 25 June 2025.
* `test_milp_feasible_and_ordered`: no capacity violations; unconstrained is no worse than one-slot; greedy is a lower bound.

## Licence and citation

No licence has been chosen yet. Add one (and the final citation) before making the repository public.
