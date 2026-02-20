# Retail Demand Forecasting Pipeline

> Enterprise-grade, SKU-level daily demand forecasting using a three-stage ML pipeline — built on the UCI Online Retail dataset, tracked with MLflow, and containerised with Docker.

---

## What This Does

Most retail forecasting fails because it treats a slow-moving, spiky gift item the same as a high-volume commodity. This pipeline doesn't.

It solves three problems that generic regression models ignore:

**The zero problem.** Roughly 70% of SKU-days have zero sales. This is not zero demand — it is a mix of genuine low demand, weekends, and stockouts. A naive model predicts near-zero for everything and looks accurate on RMSE while being useless operationally.

**The identity problem.** A SKU with a coefficient of variation of 3.0 needs a completely different forecast strategy than one with a CV of 0.3, even if their 7-day rolling mean looks identical.

**The scale problem.** A SKU that sells 200 units a day and a SKU that sells 2 units a day should not be trained with the same loss function. High-volume errors are operationally critical; low-volume errors are noise.

The pipeline addresses all three with a staged architecture that routes each SKU-day to the right specialist model.

---

## Architecture

```
Raw Transactions (541k rows)
         │
         ▼
┌──────────────────────┐
│    Data Ingestion    │  Remove cancellations, returns, bad prices
└──────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Feature Engineering │  3 layers: Temporal · SKU Identity · OOS Proxy
└──────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Three-Stage Pipeline                     │
│                                                             │
│   ┌─────────────────────────────────────┐                   │
│   │  Stage 1: Zero / Nonzero Classifier │  LightGBM         │
│   └─────────────────────────────────────┘                   │
│          │ ZERO                │ NONZERO                    │
│       predict 0          ┌─────────────────────────┐        │
│                          │ Stage 2: Low/High Router│        │
│                          └─────────────────────────┘        │
│                          │ LOW (1–5)   │ HIGH (>5)          │
│                     ┌────────┐    ┌──────────────┐          │
│                     │  3b    │    │     3a       │          │
│                     │LightGBM│    │XGBoost +     │          │
│                     │L1 loss │    │LightGBM      │          │
│                     │[0,5]   │    │log-target    │          │
│                     └────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│   MLflow Tracking    │  Params · Metrics · Artifacts per run
└──────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Business Dashboard  │  6 normalised charts + self-contained HTML
└──────────────────────┘
```

## Feature Engineering

Three layers give each model a complete picture of what it is forecasting.

### Layer 1 — Temporal History

| Feature | Description |
|---|---|
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | Sales 1, 7, 14, 28 days ago (leak-safe via shift) |
| `rolling_mean_3`, `_7`, `_28` | Rolling average over 3 / 7 / 28 days |
| `ema_7` | Exponentially weighted mean, span 7 |
| `same_dow_mean` | Avg sales on the same day-of-week over the last 4 weeks |
| `recent_momentum` | `rolling_mean_3 / rolling_mean_28` — demand accelerating or fading? |
| `zero_streak` | Consecutive zero-sale days immediately before this row |
| `days_since_sale` | Days elapsed since the last nonzero sale |

### Layer 2 — SKU Identity

Without these, the model is a generalist. A commodity item and a luxury one-off look identical based on lags alone.

| Feature | Description |
|---|---|
| `sku_avg_volume` | Historical mean daily sales on active days |
| `sku_median_vol` | Median — robust to spike outliers |
| `sku_cv` | Coefficient of variation (`std / mean`): how erratic is this SKU? |
| `sku_zero_propensity` | Fraction of all calendar days with zero sales |
| `sku_vol_tier` | Ordinal bucket: 0 = low, 1 = mid, 2 = high |

All identity stats respect `train_cutoff` — no future data leaks into static features.

### Layer 3 — Stockout Proxy

In retail, a zero often means "out of stock", not "no demand". A model that learns from OOS zeros as genuine demand signal will systematically underforecast the recovery period.

```
potential_oos = 1  when ALL of the following are true:
  (a) zero_streak is between 1–5 days
      (short run — a prolonged gap is more likely genuine low demand)
  (b) rolling 28-day sales before the zeros > 50% of the SKU's historical average
      (the SKU was actually selling before it went dark)
  (c) price has not changed in the last 7 days
      (a price drop signals a deliberate destock, not an OOS event)
```

---

## Business Metrics

All evaluation uses retail-standard normalised metrics.

| Metric | Formula | Why It Matters |
|---|---|---|
| **WAPE** | `Σ\|actual − pred\| / Σ actual` | Primary KPI. Handles zeros, comparable across SKUs of any scale |
| **Revenue WAPE** | Error weighted by unit price | Translates forecast error into £ impact |
| **MAE** | `mean(\|actual − pred\|)` | Interpretable in units |
| **Median AE** | `median(\|actual − pred\|)` | Robust to spike SKUs |
| **SKU-level WAPE** | Per-SKU WAPE, filtered to SKUs with ≥10 test-period sales | Shows the distribution of model quality across the catalogue |

The **baseline** is the nonzero training mean — not the all-rows mean, which is deflated to ~3 units by the 70% zero-filled date spine and creates a misleadingly easy baseline to beat.

---

## Business Dashboard

Six charts are generated automatically after every run, each answering a real operational question:

| Chart | Question It Answers |
|---|---|
| Forecast Accuracy by Volume Tier | Which product segments can we actually trust the forecast for? |
| Monthly Bias Analysis | Are we systematically over- or under-ordering in specific months? |
| Revenue at Risk (£) | What is the financial cost of our forecast errors, by SKU? |
| Stockout Proxy Validation | Is the OOS detector finding real stockouts or just weekend dips? |
| Normalised Feature Importance | What signals actually drive the forecast, in business language? |
| SKU Portfolio Segmentation | Where in the catalogue should we focus improvement effort? |

---

## MLflow Tracking

Every pipeline run logs the following automatically.

**Parameters:** dataset row count, number of SKUs, train/val/test split dates, all hyperparameters for each of the four model stages.

**Metrics:**

| Metric Key | Description |
|---|---|
| `val_mae`, `test_mae` | Validation and test MAE |
| `val_wape`, `test_wape` | Validation and test WAPE |
| `zero_gate_threshold` | Tuned Stage 1 threshold |
| `s1_val_accuracy` | Stage 1 classifier accuracy |
| `s3a_val_mae`, `s3b_val_mae` | Per-stage validation MAE (back-transformed) |
| `mae_improvement_pct` | % MAE improvement over nonzero-mean baseline |
| `sku_wape_median` | Median per-SKU WAPE on test set |
| `pred_zero_rate` vs `actual_zero_rate` | Zero gate health diagnostic |

**Artifacts:** `pipeline.pkl`, `feature_importance.csv`, all 6 chart PNGs, `pitch_dashboard.html`, `test_predictions.csv`

---

## Project Structure

```
demand_forecasting/
├── src/
│   ├── data_ingestion.py       # Load, clean, parse dates (mixed format support)
│   ├── feature_engineering.py  # 3-layer vectorised feature matrix (no Python loops)
│   ├── model_training.py       # Three-stage pipeline + MLflow tracking
│   ├── evaluation.py           # WAPE / MAE / RMSE / bucket / SKU-level metrics
│   ├── visualisation.py        # 6 business charts + self-contained HTML dashboard
│   └── run_pipeline.py         # End-to-end entry point
├── data/                       # Place Online Retail.csv here  ← gitignored
├── models/                     # pipeline.pkl written here     ← gitignored
├── outputs/                    # test_predictions.csv          ← gitignored
├── reports/                    # PNGs + pitch_dashboard.html   ← gitignored
├── mlflow_data/                # MLflow experiment store       ← gitignored
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

## Setup & Running

### Prerequisites

- The UCI Online Retail dataset — download from [archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- Place the file at `data/Online Retail.csv`

---

### Option 1 — Docker (recommended)

No local Python environment required. The pipeline and MLflow UI each run in their own container.

```bash
# 1. Clone
git clone https://github.com/yourname/demand-forecasting.git
cd demand-forecasting

# 2. Add data
cp "/path/to/Online Retail.csv" data/

# 3. Build image and run the full pipeline
docker compose up --build

# 4. Once the pipeline finishes, start the MLflow UI
docker compose up mlflow
```

Open **http://localhost:5000** to view experiment runs.
Open `reports/pitch_dashboard.html` in your browser for the business dashboard.

All outputs (`models/`, `outputs/`, `reports/`, `mlflow_data/`) are written to your local machine via volume mounts — the containers do not need to stay running to access them.

---

### Option 2 — Local Python

```bash
# 1. Clone and enter
git clone https://github.com/yourname/demand-forecasting.git
cd demand-forecasting

# 2. Install
pip install -r requirements.txt

# 3. Add data
cp "/path/to/Online Retail.csv" data/

# 4. Run the pipeline
python src/run_pipeline.py
```

**Open the MLflow UI** (in a second terminal, after the pipeline has run at least once):

```bash
mlflow ui --backend-store-uri ./mlflow_data
```

Then open **http://127.0.0.1:5000**.

> **Mac users:** Port 5000 conflicts with AirPlay Receiver. Use `--port 5001` and go to `http://127.0.0.1:5001` instead. Or disable AirPlay in System Settings → General → AirDrop & Handoff.

**Open the business dashboard:**

```bash
open reports/pitch_dashboard.html       # macOS
start reports/pitch_dashboard.html      # Windows
xdg-open reports/pitch_dashboard.html   # Linux
```

---

## Output Files

| Path | Description |
|---|---|
| `outputs/test_predictions.csv` | StockCode, date, actual sales, predicted, absolute error |
| `reports/pitch_dashboard.html` | Self-contained HTML business dashboard (no server required) |
| `reports/01_forecast_accuracy_by_tier.png` | WAPE by SKU volume tier |
| `reports/02_bias_analysis.png` | Monthly over/under-forecast bias |
| `reports/03_revenue_at_risk.png` | £ error by SKU + revenue WAPE distribution |
| `reports/04_oos_detection.png` | Stockout proxy validation |
| `reports/05_feature_importance.png` | Normalised feature contribution by group |
| `reports/06_sku_segmentation.png` | Portfolio map: intermittency vs volume |
| `models/pipeline.pkl` | Serialised trained pipeline (all four stages) |
| `models/feature_importance.csv` | XGBoost feature importances, normalised to 100% |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `lightgbm` | ≥ 4.0 | Stages 1, 2, 3b |
| `xgboost` | ≥ 2.0 | Stage 3a ensemble member |
| `pandas` | ≥ 2.0 | Feature engineering |
| `numpy` | ≥ 1.24 | Vectorised computations |
| `scikit-learn` | ≥ 1.3 | Metrics |
| `mlflow` | ≥ 2.10 | Experiment tracking |
| `matplotlib` | ≥ 3.7 | Business charts |
| `joblib` | ≥ 1.3 | Model serialisation |

---

## Known Limitations

**No inventory feed.** The stockout proxy is a heuristic based on price stability and pre-gap demand history, not actual stock levels. A production system would join on real inventory data.

**Single-step forecast.** This forecasts one day ahead using lagged features. Multi-step horizons (7-day, 30-day) would require either recursive prediction or direct multi-output models.

**Static SKU identity.** `sku_cv`, `sku_vol_tier`, etc. are computed once per run. SKUs that change behaviour over time — seasonal launches, product lifecycle phases — would benefit from rolling identity windows.

**No external signals.** Promotions, planned price changes, and marketing events are not in this dataset. In a production environment these would be among the highest-value features available.

---

## License

MIT — use freely, attribution appreciated.
