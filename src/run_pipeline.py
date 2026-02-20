"""
run_pipeline.py
───────────────
End-to-end entry point. Run from the project root:

    python src/run_pipeline.py

MLflow tracks to a local file store by default so it works with or
without Docker. To view results afterwards:

    mlflow ui --backend-store-uri ./mlflow_data
    # open http://localhost:5000
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_ingestion      import load_data
from feature_engineering import build_features, FEATURES
from model_training      import train, predict
from evaluation          import compute_metrics, bucket_metrics, sku_wape, print_summary
from visualization       import build_pitch_dashboard

OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ── MLflow: always use local file store unless overridden by env var ──────────
# "http://mlflow:5000" only works inside Docker; locally it hangs on DNS.
# The env var lets docker-compose override it without touching this file.
_default_uri = f"file:///{(ROOT / 'mlflow_data').as_posix()}"
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", _default_uri)


def _mlflow_log(run_id, metrics: dict, params: dict, artifacts: list):
    """Log to MLflow without blocking the main pipeline if it fails."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(run_id=run_id):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            for path in artifacts:
                if Path(path).exists():
                    mlflow.log_artifact(path)
    except Exception as e:
        print(f"[mlflow] Logging skipped: {e}")


def main():
    print("=" * 60)
    print("  DEMAND FORECASTING PIPELINE")
    print("=" * 60)
    print(f"[mlflow] Tracking URI: {MLFLOW_URI}")

    # ── Create MLflow run ID up front (non-blocking) ──────────────────────────
    run_id = None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("demand_forecasting")
        run = mlflow.start_run(run_name="three_stage_pipeline")
        run_id = run.info.run_id
        mlflow.end_run()   # close immediately — we log at the end
        print(f"[mlflow] Run ID: {run_id}")
    except Exception as e:
        print(f"[mlflow] Could not initialise run ({e}). Continuing without tracking.")

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    df = load_data()

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("\n[2/5] Building features …")
    data = build_features(df).sort_values("date")

    # ── 3. Split ──────────────────────────────────────────────────────────────
    print("\n[3/5] Splitting …")
    test_cutoff = data["date"].max() - pd.Timedelta(days=30)
    val_cutoff  = test_cutoff        - pd.Timedelta(days=30)

    train_df = data[data["date"] <= val_cutoff]
    val_df   = data[(data["date"] > val_cutoff) & (data["date"] <= test_cutoff)]
    test_df  = data[data["date"] > test_cutoff]

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name:5s}: {len(split):>8,} rows | "
              f"zero-rate: {(split['sales']==0).mean():.2%} | "
              f"{split['date'].min().date()} → {split['date'].max().date()}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[4/5] Training …")
    models = train(train_df, val_df, save=True)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating …")
    X_test = test_df[FEATURES]
    y_test = test_df["sales"]
    preds  = predict(X_test, models)

    pred_zero_rate   = (preds == 0).mean()
    actual_zero_rate = (y_test == 0).mean()
    print(f"\n  Predicted zero-rate : {pred_zero_rate:.2%}")
    print(f"  Actual zero-rate    : {actual_zero_rate:.2%}")
    if pred_zero_rate > actual_zero_rate + 0.10:
        print("  ⚠  Zero gate is over-predicting zeros")

    nonzero_train_mean = train_df.loc[train_df["sales"] > 0, "sales"].mean()
    baseline_pred      = np.full(len(y_test), nonzero_train_mean)
    print(f"  Baseline predicts   : {nonzero_train_mean:.2f} units")

    baseline_metrics = compute_metrics(y_test, baseline_pred, "Baseline (nonzero train mean)")
    model_metrics    = compute_metrics(y_test, preds,         "Three-Stage Pipeline")

    print_summary(baseline_metrics)
    print_summary(model_metrics)

    pct = (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100
    print(f"\n  MAE improvement: {pct:.1f}%")

    nz       = y_test > 0
    nz_model = compute_metrics(y_test[nz], preds[nz], "Model — nonzero days only")
    print_summary(nz_model)

    print("\nMAE by sales volume bucket:")
    print(bucket_metrics(y_test, preds).to_string())

    sw = sku_wape(test_df, preds, min_sales=10)
    print(f"\nSKU-level WAPE — {len(sw):,} SKUs")
    print(f"  Median : {sw.median()*100:.1f}%")
    print(f"  Mean   : {sw.mean()*100:.1f}%")
    print(f"  > 100% : {(sw > 1).sum()}")

    # ── Save predictions ───────────────────────────────────────────────────────
    out = test_df[["StockCode", "date", "sales"]].copy()
    out["predicted"] = preds
    out["abs_error"] = np.abs(out["sales"] - out["predicted"])
    csv_path = OUTPUTS_DIR / "test_predictions.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n  Predictions → {csv_path}")

    # ── Visualisations ─────────────────────────────────────────────────────────
    print("\n[viz] Building business dashboard …")
    build_pitch_dashboard(
        train_df       = train_df,
        test_df        = test_df,
        preds          = preds,
        baseline_preds = baseline_pred,
        models         = models,
        features       = FEATURES,
    )

    # ── Log everything to MLflow at the very end (non-blocking) ───────────────
    if run_id:
        artifacts = [str(csv_path)] + \
                    [str(p) for p in REPORTS_DIR.glob("*.png")] + \
                    [str(REPORTS_DIR / "pitch_dashboard.html")]

        _mlflow_log(
            run_id    = run_id,
            params    = {
                "n_skus":       int(data["StockCode"].nunique()),
                "total_rows":   len(data),
                "test_cutoff":  str(test_cutoff.date()),
                "val_cutoff":   str(val_cutoff.date()),
            },
            metrics   = {
                "mae":                  round(float(model_metrics["mae"]),    4),
                "rmse":                 round(float(model_metrics["rmse"]),   4),
                "wape":                 round(float(model_metrics["wape"]),   4),
                "median_ae":            round(float(model_metrics["med_ae"]), 4),
                "baseline_mae":         round(float(baseline_metrics["mae"]), 4),
                "mae_improvement_pct":  round(float(pct), 2),
                "pred_zero_rate":       round(float(pred_zero_rate), 4),
                "actual_zero_rate":     round(float(actual_zero_rate), 4),
                "sku_wape_median":      round(float(sw.median()), 4),
                "sku_wape_mean":        round(float(sw.mean()), 4),
                "mae_nonzero":          round(float(nz_model["mae"]), 4),
                "wape_nonzero":         round(float(nz_model["wape"]), 4),
            },
            artifacts = artifacts,
        )
        print(f"[mlflow] Logged → run {run_id}")
        print(f"         View:  mlflow ui --backend-store-uri {ROOT / 'mlflow_data'}")

    print("\nDone. Open reports/pitch_dashboard.html in your browser.")


if __name__ == "__main__":
    main()