"""
evaluation.py  –  All demand-forecasting metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

BINS   = [-1, 0, 5, 15, 30, 60, float("inf")]
LABELS = ["0 (zero)", "1–5", "6–15", "16–30", "31–60", "60+"]


def compute_metrics(y_true, y_pred, label="") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    wape   = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-9))
    nz     = y_true > 0
    mape   = float(mean_absolute_percentage_error(y_true[nz], y_pred[nz])) if nz.sum() else np.nan
    return dict(label=label, mae=mae, rmse=rmse, med_ae=med_ae, mape=mape, wape=wape)


def bucket_metrics(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    df["abs_err"] = np.abs(df["actual"] - df["pred"])
    df["bucket"]  = pd.cut(df["actual"], bins=BINS, labels=LABELS)
    agg = df.groupby("bucket", observed=True).agg(
        count=("actual", "count"), mean_sales=("actual", "mean"), mae=("abs_err", "mean")
    )
    agg["rel_mae_%"] = (agg["mae"] / agg["mean_sales"].clip(lower=0.01) * 100).round(1)
    agg["rows_%"]    = (agg["count"] / len(df) * 100).round(1)
    return agg


def sku_wape(test_df, y_pred, min_sales=10.0) -> pd.Series:
    df = test_df[["StockCode", "sales"]].copy()
    df["pred"]    = y_pred
    df["abs_err"] = np.abs(df["sales"] - df["pred"])
    totals = df.groupby("StockCode")["sales"].sum()
    active = totals[totals >= min_sales].index
    return (
        df[df["StockCode"].isin(active)]
        .groupby("StockCode")
        .apply(lambda x: x["abs_err"].sum() / x["sales"].sum(), include_groups=False)
        .sort_values(ascending=False)
    )


def print_summary(metrics: dict):
    label = metrics.get("label", "")
    print(f"\n{'═'*54}")
    if label:
        print(f"  {label}")
        print(f"{'─'*54}")
    print(f"  MAE      : {metrics['mae']:.4f}")
    print(f"  RMSE     : {metrics['rmse']:.4f}")
    print(f"  Median AE: {metrics['med_ae']:.4f}")
    print(f"  MAPE     : {metrics['mape']*100:.2f}%  (nonzero actuals)")
    print(f"  WAPE     : {metrics['wape']*100:.2f}%  ← primary retail KPI")
    print(f"{'═'*54}")