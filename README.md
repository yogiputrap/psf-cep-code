# PSF: Predictive Scheduling Framework for First-Attempt Delivery in CEP Logistics

Reference implementation of a three-layer framework that turns shipment-level first-attempt failure risk into capacitated delivery-slot plans for a courier, express and parcel (CEP) company that holds no courier-level data. It accompanies the manuscript

> *Hidden Failures and Upstream Limits: Slot-Aware Risk Prediction and Scheduling for First-Attempt Delivery in Courier, Express and Parcel Logistics.*

| | |
|---|---|
| **Language** | Python 3.11 or newer, CPU only |
| **Core libraries** | CatBoost (with a scikit-learn fallback), SciPy HiGHS for the MILP, pandas, NumPy |
| **Runs on** | Synthetic data generated on the fly; no confidential data needed |
| **Companion dataset** | Anonymised shipments, private repository, access on request: https://github.com/yogiputrap/shipments_anon.parquet |

---

## 1. The idea in one page

A conventional CEP company schedules couriers statically and ignores the failure risk of each time slot. In partner-based operations it also has no run sheets or courier records. The framework works without them:

```
 Layer 1  Failure label            first-attempt failure = explicit failure
                                   + implicit carry-over (delivery proof > 1 h after the first attempt)
              |
 Layer 2  Slot-aware risk model    CatBoost on 95 name-free features, leakage-safe history,
                                   risk matrix: one failure probability per shipment and slot
              |
 Layer 3  Capacitated slot plan    MILP: move each shipment by at most one slot so that the expected
                                   failures fit the capacity of a virtual courier pool
```

Main results in the paper (real data, strict 21-day temporal hold-out):

| Result | Value |
|---|---|
| ROC-AUC, final model (95 features) | 0.854 (95% CI 0.842 to 0.866) |
| ROC-AUC, slot-only rule / logistic regression on 13 features | 0.710 / 0.757 |
| Mean ROC-AUC, walk-forward backtest (3 origins, horizons 1 to 14 days) | 0.839 |
| Expected failure rate, one-slot MILP | 20.4% down to 13.9% (32.2% relative) |
| Same, if shipments cannot be moved earlier | the gain turns into a 3.8% deterioration |
| Same, with only half of the couriers | the gain falls to 1.2% |

These numbers need the confidential data and are **not** reproduced by this repository (see Section 2).

## 2. Data policy: what you can and cannot run here

The original shipment records are confidential and contain personal data (names, phone numbers, addresses). They are not in this repository and cannot be shared.

| Input | Where | What it is |
|---|---|---|
| Synthetic shipments | `src/psf/synth.py` (generated on the fly) | Artificial rows with a planted slot-aware failure mechanism. Only a few published aggregates set its scale (study window, regional split, slot-wise failure rates). It is not a masked copy of real data. |
| Holiday calendar | `data/holidays_2025.csv` | Dates from the 2025 joint decree No. 1017/2/2 of 2024 and its amendment No. 933/1/3 of 2025 (public documents) |
| Aggregate results | `results/aggregate/*.csv` | Numbers transcribed from the paper's tables; no row-level data |
| Anonymised dataset | separate private repository | 94,360 shipments with all 95 features, access on request |

With the anonymised dataset, a CatBoost model on the paper's feature lists reproduces the reported ROC-AUC within sampling noise (0.8542 against 0.8539 on the original data); see that repository's README.

`.gitignore` blocks `data/raw/`, `data/private/`, parquet, pickle and `.env` files so that company data cannot be committed by accident. Please keep it that way.

## 3. Repository layout

```
src/psf/
  features.py     slots, leakage-safe lagged history, calendar features
  model.py        slot-aware risk model, temporal split, risk matrix
  scheduling.py   capacitated slot allocation (MILP and baselines)
  metrics.py      ROC-AUC with block bootstrap, paired differences, ECE, WBI
  synth.py        synthetic shipment generator
scripts/run_demo.py        end-to-end run on synthetic data
tests/test_pipeline.py     8 tests
data/holidays_2025.csv     public holiday calendar
results/aggregate/         result tables of the paper (real data, aggregated)
```

## 4. What is implemented

| Paper section | Module |
|---|---|
| III-A target and slots | `psf/features.py::slot_from_hour`; label `y` in `psf/synth.py` |
| III-C slot-aware risk model, risk matrix | `psf/model.py::RiskModel` (CatBoost, or scikit-learn fallback) |
| III-D leakage-safe history (Eq. 4, Algorithm 1), calendar | `psf/features.py::lagged_history`, `calendar_features` |
| III-E capacitated allocation (Eq. 5-6, Algorithm 2): historical, greedy, one-slot, unconstrained, not-earlier | `psf/scheduling.py` (HiGHS through SciPy `milp`) |
| III-F metrics: day-level block bootstrap, paired differences, ECE, WBI | `psf/metrics.py` |

**Not included**, because these depend on the company's raw fields: the PostgreSQL star-schema warehouse and its ETL, the address and phone quality indicators, the predictive R-ESR, the TF-IDF and native CatBoost text features, the ten-architecture comparison and the Streamlit prototype. The pipeline, the leakage controls and the optimisation model can be inspected and run.

## 5. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions used for the paper (CPU only)
pytest -q                                # 8 tests: leakage safety, capacity formula, MILP feasibility, calendar counts
python scripts/run_demo.py               # synthetic end-to-end run, about one minute
```

`run_demo.py` accepts `--n` (number of synthetic shipments, default 30000), `--seed`, `--out` (default `results/demo`) and `--boot` (bootstrap resamples, default 300).

The demo trains on the first days and tests on the last 21 days, builds the risk matrix and runs the scheduling variants and courier-availability levels. It reports the expected failure rate under the model and under the synthetic ground truth, which is possible only because the data are synthetic. Results go to `results/demo/` (git-ignored). The demo's numbers describe the synthetic data only.

## 6. Design choices worth knowing

* **Leakage safety first.** History features for day *d* use only outcomes before day *d* (lag 1). Invalid evaluation setups, such as in-sample scoring or random splits, inflate ROC-AUC without any change of model, so evaluation here splits by time only.
* **Name-free features.** The final model does not use personal names; name-derived indicators were tested separately (`results/aggregate/name_derived_indicator_ablation.csv`).
* **Capacity is an assumption.** The capacity of a slot is `K = C * kappa * pi`, with a virtual courier pool derived from daily volume, because no courier data exist. The sensitivity to `kappa` and `pi` is in `results/aggregate/capacity_sensitivity_kappa_pi.csv`.
* **Model-based, not field-tested.** Scheduling gains are model-based counterfactual estimates, not measured outcomes.

## 7. Tests worth reading

* `test_history_is_leakage_safe`: flipping outcomes on day *d* and later does not change any history feature of day *d* (lag 1).
* `test_history_matches_naive_definition`: the vectorised history equals a direct implementation of Eq. 4.
* `test_holiday_calendar_matches_decree`: 10 national holiday dates and 8 joint-leave days fall between 15 March and 25 June 2025.
* `test_milp_feasible_and_ordered`: no capacity violations; unconstrained is no worse than one-slot; greedy is a lower bound.

## 8. Aggregate results

`results/aggregate/` holds the paper's tables as CSV files: baselines, feature ladder and group ablation (3 seeds), redundancy check, name-derived indicators, paired AUC differences, ten-architecture comparison, label composition, walk-forward backtest, scheduling results and capacity sensitivity. They contain no shipment-level or personal data. They are the only results of the real data in this repository; the synthetic demo produces different numbers.

## 9. Limitations

One company, one region and 103 days; no external benchmark and no blind test after June 2025; part of the failure label (12.34% of shipments) is inferred from delivery-proof timing; no courier data, so courier effects are not measured. See the limitations section of the paper.

## 10. Licence and citation

No licence has been chosen yet. Add one (and the final citation) before submission.

```bibtex
@article{psf_cep_2025,
  title   = {Hidden Failures and Upstream Limits: Slot-Aware Risk Prediction and Scheduling for First-Attempt Delivery in Courier, Express and Parcel Logistics},
  author  = {Hidayat and Pratama},
  journal = {IEEE Access},
  note    = {Under preparation; complete the entry before submission}
}
```
