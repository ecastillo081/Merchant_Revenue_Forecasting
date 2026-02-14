# PROJECT_CONTEXT.md — Full System Narrative

*Structured repo audit: Principal Engineer + Finance Systems Architect perspective.*

---

## 1. Project Overview

**What problem does this system solve?**  
The system forecasts **monthly merchant revenue** for a panel of merchants and **compares multiple forecasting methods** so that FP&A can choose a method that reduces forecast error. It answers: “Which forecasting approach gives the lowest error for our merchants, and by how much?”

**Who is the intended user?**  
Internal finance/FP&A (and possibly analytics) users who need to set budgets, plan resources, and improve revenue forecast accuracy. The README frames the outcome as “tighter planning” and “improves budgets and inventory/cash planning.”

**What decision(s) does this system support?**  
- **Model selection:** Which forecasting method to adopt (e.g., Holt-Winters, SARIMA, Linear Regression vs. Seasonal Naive).  
- **Planning quality:** Using ~8–10% MAPE instead of ~12–13% reduces error in revenue expectations used for budgets and resource allocation.

**Revenue-generating, infrastructure, research, or internal tooling?**  
**Internal tooling / research.** It does not serve revenue directly; it improves the quality of internal forecasts that drive planning. The dataset is synthetic (“designed to mimic realistic FP&A challenges”), so the repo functions as a **methodology and comparison framework** rather than a production forecast pipeline.

**How does this repo fit into the broader portfolio?**  
Cannot infer from repo. There is no reference to other systems, APIs, or a central data platform. It appears **standalone**: one raw Excel input, local Python scripts, file-based outputs. README mentions “Next (V2): Incorporate promotions, marketing spend and macro index as external regressors…” suggesting this is v1 of a planned evolution.

---

## 2. Design Intent

**Philosophy:**  
- **Correctness-first** in the sense of rigorous evaluation: fixed train/test split (last 12 months), multiple error metrics (MAPE, sMAPE, MAE, RMSE, MASE), and portfolio-level aggregation (leaderboard, boxplots).  
- **Experimentation:** Many model families (naive, seasonal naive, SMA, WMA, SES, Holt, Holt-Winters, SARIMA, Linear Regression) are run in parallel so that the “best” can be chosen by a single metric (MAPE).  
- **Reproducibility of comparison** rather than production hardening: no automated tests, no versioned data, no API.

**Tradeoffs made:**  
- **Simplicity over robustness:** Paths are relative (`../data/raw/`, `../data/transformed/`, `../figures/`), so correct behavior depends on the current working directory (e.g., running from `source/` or project root).  
- **Single-metric “best model”:** `chosen_metric = "MAPE"` is hardcoded in `source/all_merchants.py` and `source/single_merchant_forecast_metrics.py`; no configuration layer.  
- **In-memory, file-out:** All compute is in-memory; outputs are Excel and PNG. No database, no idempotency keys, no append vs. replace policy documented.

**Where the system is opinionated:**  
- **Horizon:** Forecast horizon `H = 12` is fixed in `functions/forecast_methods.py` (12-month-ahead evaluation).  
- **Train/test split:** Always “last H months” as test; no expanding vs. rolling window options in code.  
- **SARIMA order:** Fixed `(1,1,1)` and seasonal `(1,1,1,12)` in both `all_merchants.py` and `single_merchant_forecast.py` (no auto-arima or grid search).  
- **Minimum history:** Merchants with fewer than `max(24, H+6)` months are skipped (`evaluate_one_merchant` returns `None`).

---

## 3. Architecture Overview

**Runtime environment:**  
Local only. Python scripts are intended to be run on a developer or analyst machine. No cloud, serverless, or containerization is present. No `Dockerfile`, no `.github/workflows`, no scheduler references.

**Data storage:**  
- **Raw:** Single file, `data/raw/merchant_monthly_revenue.xlsx`.  
- **Transformed outputs:** `data/transformed/leaderboard.xlsx`, `data/transformed/best_model_by_merchant.xlsx`, and optionally `data/transformed/{merchant_id}_forecast.xlsx`, `data/transformed/{merchant_id}_metrics.xlsx`.  
- **Artifacts:** `figures/` (e.g., `leaderboard_MAPE.png`, `boxplot_MAPE.png`, `M001_actual_vs_forecast.png`).  
- **Metadata:** `data/raw/merchant_monthly_data_dictionary.json` documents column semantics.  
No database, no data warehouse, no external APIs in code.

**Compute layers:**  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib` (see `requirements.txt`).  
- **Forecast logic:** `functions/forecast_methods.py` (baselines + linear regression); `statsmodels` for SES, Holt, Holt-Winters, SARIMAX.  
- **Metrics:** `functions/metrics.py` (MAPE, sMAPE, MAE, RMSE, MASE).  
- **Orchestration:** No job runner; entry points are scripts that assume a specific CWD.

**Frontend / backend:**  
No frontend. No backend service. This is a **batch script + Excel/PNG output** model. README and figures are the primary “interface.”

**CI/CD:**  
None. No GitHub Actions, no Jenkins, no pipeline definitions. `.gitignore` includes `.venv`, `README.md`, `data/generator`, `.idea`.

**Text diagram of system flow:**

```
User (analyst)
    │
    ├─► [Run] source/all_merchants.py  (from project root or source/)
    │       │
    │       ├─ reads: data/raw/merchant_monthly_revenue.xlsx
    │       ├─ uses: functions/forecast_methods.py, functions/metrics.py
    │       └─ writes: data/transformed/leaderboard.xlsx
    │                  data/transformed/best_model_by_merchant.xlsx
    │
    ├─► [Run] plots/forecasting_leaderboard.py
    │       │
    │       ├─ imports source.all_merchants (side effect: may re-run forecasts if run fresh)
    │       ├─ reads: data/transformed/leaderboard.xlsx
    │       └─ writes: figures/leaderboard_MAPE.png, figures/boxplot_MAPE.png
    │
    └─► [Run] source/single_merchant_forecast.py (then metrics, then plots/single_merchant_graphs.py)
            │
            ├─ reads: data/raw/merchant_monthly_revenue.xlsx
            ├─ hardcoded merchant_id = "M001"
            └─ writes: data/transformed/M001_forecast.xlsx, M001_metrics.xlsx, figures/M001_actual_vs_forecast.png
```

---

## 4. Data Model & Flow

**Raw data source:**  
- **Single table:** `merchant_monthly_revenue.xlsx` (schema described in `data/raw/merchant_monthly_data_dictionary.json`).  
- **Documented columns:** `date`, `merchant_id`, `region`, `vertical`, `revenue`, `orders`, `avg_order_value`, `marketing_spend`, `promo_month`, `seasonal_index`, `macro_index`, `mktg_rev_ratio`, `cac_per_order`.  
- **Scope (from README):** 50 merchants × 60 months (Jan 2020 – Dec 2024).  
- **Provenance:** README states “synthetic data was generated”; `.gitignore` references `data/generator` but no generator code is present in the repo — **cannot infer generation pipeline or idempotency**.

**Transform layers:**  
- **No staging/intermediate/marts in a dbt sense.** The only “transform” is in-code:  
  - `pd.to_datetime(…).dt.to_period("M").dt.to_timestamp()` and `set_index("date").asfreq("MS")` to get monthly series.  
  - For Linear Regression: creation of `rev_lag1`, `rev_lag12`, `month`, and month dummies; `fillna(method="bfill").fillna(0)` (pandas deprecated `method="bfill"` in favor of `bfill()` — potential future breakage).  
- **Outputs are “marts” only in the sense of purpose:** `leaderboard` (all merchants × models × metrics), `best_model_by_merchant`, and per-merchant forecast/metrics Excel files.

**Ledger or accounting logic:**  
None. This is forecasting and evaluation only; no double-entry, no reconciliation, no audit trail of money movement.

**Replace vs. append:**  
All writes **overwrite** the target files (`to_excel(…)` with no mode or append logic). No versioning, no “as-of” dates in filenames.

**Idempotency:**  
Not designed for. Re-running `all_merchants.py` overwrites `leaderboard.xlsx` and `best_model_by_merchant.xlsx`. No idempotency keys or incremental logic.

**Validation checks:**  
- **Minimum history:** In `evaluate_one_merchant`, a merchant is skipped if `len(y) < max(24, H + 6)`.  
- **Forecast length:** `forecasts_to_df` raises `ValueError` if a model’s forecast length does not match `y_test_index`.  
- **No schema validation** on read (no check that required columns exist or that `revenue` is non-negative).  
- **MASE:** In `functions/metrics.py`, division by zero is avoided by using `1e-8` when the seasonal denominator is zero or NaN.

**dbt:**  
Not present. No model layers, no dbt tests.

---

## 5. Core Business Logic

**What calculations matter?**  
- **Revenue forecasts** for the next 12 months (aligned with “last 12 months” test set) using:  
  - Naive, Seasonal Naive, SMA(3), WMA(1,2,3);  
  - SES, Holt (additive trend), Holt-Winters (additive trend, multiplicative seasonal, 12-month period);  
  - SARIMA(1,1,1)(1,1,1,12);  
  - Linear Regression with `marketing_spend`, `promo_month`, `macro_index`, `rev_lag1`, `rev_lag12`, and month dummies.  
- **Error metrics:** MAPE (%), sMAPE (%), MAE, RMSE, MASE (scale-free vs. naive seasonal).  
- **Best model:** Per merchant and globally, the model with **minimum MAPE** (or whatever `chosen_metric` is set to).

**What metrics are computed?**  
- **Per (merchant, model):** MAPE, sMAPE, MAE, RMSE, MASE.  
- **Portfolio:** Mean MAPE (and other metrics) across merchants per model (leaderboard), and distribution (boxplot).  
- **Decision metric:** “Best” model per merchant stored in `best_model_by_merchant.xlsx`.

**Assumptions embedded:**  
- **Horizon:** 12-month-ahead is the relevant planning horizon.  
- **Train/test:** Last 12 months are “future” for evaluation; no mention of holdout for true out-of-time validation.  
- **Linearity and stationarity:** SARIMA and Holt-Winters assume (or tolerate) certain time-series properties; Linear Regression assumes linear relationship of revenue to lags and regressors.  
- **Seasonality:** 12-month cycle is fixed (Holt-Winters `seasonal_periods=12`, SARIMA seasonal order 12, seasonal naive repeat of last 12 months).  
- **No missing months:** `asfreq("MS")` fills missing months with NaN; downstream logic (e.g., dropping or filling) is partial (e.g., Linear Regression uses bfill then 0).

**What decisions depend on this logic?**  
- Choosing a default or recommended forecasting method for the organization.  
- Interpreting “~30% relative improvement” (README) as justification to move from Seasonal Naive to Holt-Winters/SARIMA/Linear Regression for budget and planning.

---

## 6. Security & Risk Controls

**Secrets handling:**  
None in repo. No API keys, no credentials, no `.env` or vault references. All inputs are local files.

**Environment separation:**  
No dev/staging/prod. Single environment (local).

**Privilege boundaries:**  
Not applicable; no multi-user or role-based access. Whoever runs the scripts has full read/write to the paths used.

**Data retention assumptions:**  
Not specified. Outputs are overwritten on each run; there is no retention or archival policy in code or docs.

**Where this system could break:**  
- **Missing or moved file:** `merchant_monthly_revenue.xlsx` missing or wrong path → runtime error.  
- **Wrong working directory:** Scripts use `../data/`, `../figures/` → failures or writing to wrong place if run from a different CWD.  
- **Missing dependency:** `read_excel` requires an engine (e.g. `openpyxl` for `.xlsx`); `requirements.txt` does not list `openpyxl` → possible failure on a clean install.  
- **Pandas deprecation:** `fillna(method="bfill")` in `functions/forecast_methods.py` is deprecated → may break in future pandas versions.  
- **Merchant set change:** If raw data had more/fewer than 50 merchants or different IDs, behavior is data-driven except for `single_merchant_forecast.py`, which hardcodes `m_id = "M001"`.

**Known failure modes:**  
- **Silent skip:** In `all_merchants.main()`, if `evaluate_one_merchant` raises, the merchant is skipped with `print(f"Skip {m_id}: {e}")` and processing continues; no aggregation of failures or alerting.  
- **Import-time execution:** `plots/forecasting_leaderboard.py` imports `source.all_merchants`; `single_merchant_graphs.py` imports `single_merchant_forecast` and `single_merchant_forecast_metrics`. So “just plotting” can re-execute forecasting and overwrite outputs depending on how the interpreter resolves modules.

---

## 7. Operational Workflows

**How data gets refreshed:**  
- **Raw data:** Not automated. Replacement of `data/raw/merchant_monthly_revenue.xlsx` is assumed to be manual or done by an external process (e.g., a generator in `data/generator`, which is gitignored — cannot infer).  
- **Transformed data:** By re-running `source/all_merchants.py` (and optionally single-merchant scripts). No scheduler, no cron, no pipeline.

**Manual steps required:**  
1. Ensure `data/raw/merchant_monthly_revenue.xlsx` exists and is up to date.  
2. Run from project root (or adjust CWD): e.g. `python source/all_merchants.py`.  
3. To regenerate leaderboard figures: run from project root e.g. `python plots/forecasting_leaderboard.py` (and be aware of import side effects).  
4. For single-merchant view: run `source/single_merchant_forecast.py`, then `source/single_merchant_forecast_metrics.py`, then `plots/single_merchant_graphs.py` (or rely on import chain); change `m_id` in `single_merchant_forecast.py` for another merchant.

**Scripts and their roles:**  
- **`source/all_merchants.py`:** Loads raw Excel, loops over all merchants, runs all models, computes metrics, writes `leaderboard.xlsx` and `best_model_by_merchant.xlsx`. Only entry point with `if __name__ == "__main__"` guard.  
- **`source/single_merchant_forecast.py`:** Loads raw Excel, filters to one merchant (M001), runs models (no Linear Regression in the exported `preds` in the current snippet for the dataframe, but `all_merchants` includes it), builds `forecasts_df`, writes `data/transformed/{m_id}_forecast.xlsx`. Runs on import.  
- **`source/single_merchant_forecast_metrics.py`:** Imports from `single_merchant_forecast`, computes metrics table, writes `{m_id}_metrics.xlsx`, defines `best_model`.  
- **`plots/forecasting_leaderboard.py`:** Reads `leaderboard.xlsx`, produces bar chart and boxplot, saves to `figures/`.  
- **`plots/single_merchant_graphs.py`:** Imports from single-merchant forecast and metrics, plots actual vs. forecasts, saves to `figures/{m_id}_actual_vs_forecast.png`.

**Scheduling:**  
None. No cron, no CI job, no workflow engine.

**One-time or sandbox utilities:**  
None identified. The repo is the “utility”; no separate one-off scripts or sandbox configs.

---

## 8. Testing & Validation Philosophy

**What is validated?**  
- **Consistency of forecast length** vs. test period in `forecasts_to_df` (raises if lengths differ).  
- **Minimum history** per merchant (skip if insufficient data).  
- **Numerical robustness** in metrics: MAPE/sMAPE use `np.clip(…, 1e-8, None)` to avoid division by zero; MASE falls back to `1e-8` when seasonal denominator is zero or NaN.

**What is not validated?**  
- No unit tests (no `pytest`, no `tests/` directory).  
- No integration tests (no “run pipeline on fixture data and assert outputs”).  
- No schema or data quality checks on input (column presence, types, non-negative revenue, no duplicate (merchant_id, date)).  
- No regression tests on metric values or model rankings.  
- No checks that Excel outputs are readable or that figures render.

**Where silent failures could happen?**  
- **Merchant skip:** Exceptions in `evaluate_one_merchant` cause that merchant to be skipped with only a print; leaderboard is partial without explicit warning.  
- **Missing or empty series:** If a model returns `None` or wrong length, `forecasts_to_df` may skip (continue) or raise; if all models for a merchant fail, no row is added to leaderboard.  
- **Plots:** If `leaderboard.xlsx` is missing or empty when running `forecasting_leaderboard.py`, the script will fail; no graceful handling.

**Financial invariants:**  
None enforced. No reconciliation to a ledger, no check that forecast totals or metrics satisfy accounting or consistency rules.

---

## 9. Strengths of the System

- **Clear separation of concerns:** Forecasting primitives live in `functions/forecast_methods.py`, metrics in `functions/metrics.py`; both are reusable and avoid duplication between `all_merchants` and `single_merchant_forecast`.  
- **Rigorous evaluation design:** Fixed horizon (H=12), explicit train/test split, and five metrics (MAPE, sMAPE, MAE, RMSE, MASE) give a disciplined comparison framework.  
- **Portfolio view:** Aggregation across merchants (leaderboard, boxplot) supports “which method works best on average and how variable is performance?” rather than one-off charts.  
- **Data dictionary:** `merchant_monthly_data_dictionary.json` documents column semantics, improving interpretability and onboarding.  
- **Standard methods:** Use of statsmodels and sklearn aligns with common practice and makes the methodology easy to compare to literature or other teams.  
- **Single source of horizon and metric:** `H` and the set of models are centralized in one place for the batch pipeline (`all_merchants`), reducing drift between single-merchant and all-merchant runs (except for the hardcoded M001 and the fact that single_merchant_forecast does not include Linear Regression in its exported `preds` for the dataframe in the current code).

---

## 10. Fragility & Technical Debt

- **Path and CWD dependence:** All file paths are relative (`../data/`, `../figures/`). Scripts assume a specific working directory; no `Path(__file__).resolve().parent` or config-based root.  
- **Tight coupling via imports:** Plot scripts import from source modules that execute on import (e.g., load data, run models). Running “just the plot” can re-run forecasts and overwrite outputs; order of execution is implicit.  
- **Hardcoded choices:** `m_id = "M001"` in `single_merchant_forecast.py`; `chosen_metric = "MAPE"` in two files; SARIMA orders fixed in two places.  
- **Deprecated pandas:** `fillna(method="bfill")` in `functions/forecast_methods.py` should be replaced with `bfill()` for future pandas compatibility.  
- **Incomplete requirements:** `openpyxl` (or equivalent) is needed for `read_excel` on `.xlsx` but is not in `requirements.txt` — risk of failure on clean install.  
- **No tests:** Any refactor or dependency upgrade can break behavior with no automated signal.  
- **Partial single-merchant parity:** `single_merchant_forecast.py` does not add Linear Regression to the `preds` dict that is passed to `forecasts_to_df`, so the single-merchant Excel and plots do not include that model (while `all_merchants` does).  
- **Error handling:** Exceptions in the merchant loop only result in a print; no structured logging, no exit code, no report of how many merchants failed.

---

## 11. If This Repo Disappeared Tomorrow

- **Capability lost:**  
  - Reproducible comparison of multiple forecasting methods (naive through SARIMA and Linear Regression) on a common merchant panel.  
  - Generation of a MAPE-based (or configurable-metric) leaderboard and best-model-per-merchant table.  
  - Single-merchant forecast and actual-vs-forecast charts for at least one merchant (M001).  
  - A single place that encodes H=12, the train/test split, and the chosen metric set.

- **Decisions that would degrade:**  
  - Choice of “which forecasting method to use for merchant revenue” would no longer be backed by this exact evaluation.  
  - Any process that depended on `leaderboard.xlsx` or `best_model_by_merchant.xlsx` (e.g., a human or downstream script reading these files) would lose updated inputs unless reimplemented.

- **Systems that would break:**  
  - Cannot infer from repo. No other repositories or services reference this codebase in the audited files. Any dependency would be external (e.g., a manual process that runs these scripts or consumes the Excel/figures).

---

## 12. One-Paragraph Executive Summary

This repository is an **internal merchant revenue forecasting and method-comparison tool** for FP&A. It takes a single Excel file of monthly merchant revenue (and related drivers), runs nine forecasting methods—from naive baselines to Holt-Winters, SARIMA, and Linear Regression—and evaluates them on the last 12 months using MAPE, sMAPE, MAE, RMSE, and MASE. Outputs are Excel leaderboards, best-model-per-merchant tables, and PNG figures. The design prioritizes **correctness of comparison and experimentation** over production hardening: no tests, no CI/CD, and paths and some parameters are hardcoded. It is **standalone and local** (no cloud or APIs). If the repo were lost, the organization would lose a reproducible, multi-model evaluation framework that currently supports the decision to move from ~12–13% to ~8–10% MAPE for budget and planning.
