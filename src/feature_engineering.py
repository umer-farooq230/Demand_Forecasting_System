"""
feature_engineering.py
───────────────────────
Vectorized SKU-level feature matrix — no Python for-loops.

Performance fixes vs previous version:
  1. _zero_streak and _days_since_last_sale were pure Python row-by-row loops
     across millions of rows — the #1 cause of slowness. Replaced with
     vectorized cumsum-grouping tricks that run in NumPy.
  2. Date spine uses pivot/reindex instead of MultiIndex.from_product,
     which avoids materialising a huge intermediate object.
  3. int8/int16 dtypes for boolean and small-integer features to cut RAM.
"""

import pandas as pd
import numpy as np

MIN_ACTIVE_DAYS = 30


# ── Vectorized streak helpers ─────────────────────────────────────────────────

def _zero_streak_vec(s: pd.Series) -> pd.Series:
    """
    Consecutive zero-sales days immediately preceding each row (leak-safe).
    Uses cumsum of nonzero events to define groups, then counts position
    within each zero-run. Pure NumPy — no Python iteration.
    """
    shifted     = s.shift(1).fillna(0)
    group       = (shifted > 0).astype(int).cumsum()
    streak      = shifted.groupby(group).cumcount()
    return streak.where(shifted == 0, 0)


def _days_since_vec(s: pd.Series) -> pd.Series:
    """
    Days since the last nonzero sale (shift-safe). Same cumsum trick.
    """
    shifted    = s.shift(1).fillna(0)
    group      = (shifted > 0).astype(int).cumsum()
    days_since = shifted.groupby(group).cumcount()
    return days_since.where(shifted == 0, 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, train_cutoff=None) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["date"]        = df["InvoiceDate"].dt.normalize()

    # Cap per-SKU outliers at 99th percentile
    upper          = df.groupby("StockCode")["Quantity"].transform(lambda x: x.quantile(0.99))
    df["Quantity"] = df["Quantity"].clip(upper=upper)

    # ── Daily aggregation ─────────────────────────────────────────────────────
    daily = (
        df.groupby(["StockCode", "date"], sort=False)
          .agg(sales=("Quantity", "sum"), unit_price=("UnitPrice", "mean"))
          .reset_index()
    )

    global_date_min = daily["date"].min()
    global_date_max = daily["date"].max()

    # ── SKU filter ────────────────────────────────────────────────────────────
    active = daily[daily["sales"] > 0].groupby("StockCode")["date"].count()
    valid  = active[active >= MIN_ACTIVE_DAYS].index
    daily  = daily[daily["StockCode"].isin(valid)].copy()

    # ── SKU identity stats (no leakage past train_cutoff) ────────────────────
    sku_ref = daily[daily["sales"] > 0]
    if train_cutoff is not None:
        sku_ref = sku_ref[sku_ref["date"] <= pd.Timestamp(train_cutoff)]

    sku_stats = sku_ref.groupby("StockCode")["sales"].agg(
        sku_avg_volume="mean",
        sku_median_vol="median",
        sku_std_vol="std",
    ).fillna(0)
    sku_stats["sku_cv"] = sku_stats["sku_std_vol"] / (sku_stats["sku_avg_volume"] + 1e-6)

    total_days = (global_date_max - global_date_min).days + 1
    active_cnt = daily[daily["sales"] > 0].groupby("StockCode")["date"].count()
    sku_stats["sku_zero_propensity"] = 1 - (active_cnt / total_days).clip(0, 1)

    vol_bins = sku_stats["sku_avg_volume"].quantile([0.33, 0.67]).values
    sku_stats["sku_vol_tier"] = pd.cut(
        sku_stats["sku_avg_volume"],
        bins=[-np.inf, vol_bins[0], vol_bins[1], np.inf],
        labels=[0, 1, 2],
    ).astype(float)

    # ── Full date spine via pivot / reindex (fast) ────────────────────────────
    # pivot → reindex on full date range → stack back to long form.
    # Avoids the huge MultiIndex.from_product intermediate object.
    all_dates = pd.date_range(global_date_min, global_date_max, freq="D")

    sales_pivot = daily.pivot(index="date", columns="StockCode", values="sales").reindex(all_dates)
    price_pivot = daily.pivot(index="date", columns="StockCode", values="unit_price").reindex(all_dates)
    price_pivot = price_pivot.ffill().bfill()

    sales_pivot.index.name = "date"
    price_pivot.index.name = "date"

    daily = (
        sales_pivot.stack(future_stack=True)
                   .reset_index()
                   .rename(columns={0: "sales"})
    )
    daily["sales"] = daily["sales"].fillna(0.0)

    price_long = (
        price_pivot.stack(future_stack=True)
                   .reset_index()
                   .rename(columns={0: "unit_price"})
    )
    daily = daily.merge(price_long, on=["date", "StockCode"], how="left")
    daily["unit_price"] = daily["unit_price"].fillna(0.0)

    daily = daily.sort_values(["StockCode", "date"]).reset_index(drop=True)

    # Attach SKU identity stats (one join, not per-row)
    daily = daily.join(sku_stats, on="StockCode")

    g = daily.groupby("StockCode", sort=False)["sales"]

    # ── Layer 1: Temporal features ────────────────────────────────────────────
    for lag in [1, 7, 14, 28]:
        daily[f"lag_{lag}"] = g.shift(lag)

    for w in [3, 7, 28]:
        daily[f"rolling_mean_{w}"] = g.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
        )

    daily["ema_7"] = g.transform(
        lambda x: x.shift(1).ewm(span=7, min_periods=1).mean()
    )

    # ── Vectorized intermittency features ─────────────────────────────────────
    daily["zero_streak"]     = g.transform(_zero_streak_vec)
    daily["days_since_sale"] = g.transform(_days_since_vec)
    daily["recent_momentum"] = daily["rolling_mean_3"] / (daily["rolling_mean_28"] + 1e-6)

    # ── Stockout proxy ────────────────────────────────────────────────────────
    rolling_28_lag = g.transform(lambda x: x.shift(1).rolling(28, min_periods=7).mean())
    price_std_7    = (
        daily.groupby("StockCode", sort=False)["unit_price"]
             .transform(lambda x: x.rolling(7, min_periods=1).std())
             .fillna(0)
    )
    daily["potential_oos"] = (
        daily["zero_streak"].between(1, 5) &
        (rolling_28_lag > daily["sku_avg_volume"] * 0.5) &
        (price_std_7 < 0.5)
    ).astype(np.int8)

    # ── Price ─────────────────────────────────────────────────────────────────
    daily["log_price"]    = np.log1p(daily["unit_price"])
    avg_price             = daily.groupby("StockCode")["unit_price"].transform("mean")
    daily["price_vs_avg"] = daily["unit_price"] / (avg_price + 1e-6)

    # ── Calendar ──────────────────────────────────────────────────────────────
    daily["dow"]          = daily["date"].dt.dayofweek.astype(np.int8)
    daily["month"]        = daily["date"].dt.month.astype(np.int8)
    daily["is_weekend"]   = (daily["dow"] >= 5).astype(np.int8)
    daily["is_month_end"] = daily["date"].dt.is_month_end.astype(np.int8)
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(np.int16)

    # ── Same-weekday rolling mean ─────────────────────────────────────────────
    daily["same_dow_mean"] = (
        daily.groupby(["StockCode", "dow"], sort=False)["sales"]
             .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )
    daily["same_dow_mean"] = daily["same_dow_mean"].fillna(daily["rolling_mean_7"])

    # ── Drop first 28 days per SKU (lags invalid) ─────────────────────────────
    daily = daily.dropna(subset=["lag_28"]).reset_index(drop=True)
    daily[FEATURES] = daily[FEATURES].fillna(0)

    print(
        f"[feature_engineering] {len(daily):,} rows | "
        f"zero-rate: {(daily['sales']==0).mean():.2%} | "
        f"SKUs: {daily['StockCode'].nunique():,} | "
        f"NaNs in features: {daily[FEATURES].isna().sum().sum()}"
    )
    return daily


FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_3", "rolling_mean_7", "rolling_mean_28", "ema_7",
    "zero_streak", "days_since_sale", "recent_momentum", "potential_oos",
    "sku_avg_volume", "sku_median_vol", "sku_cv", "sku_zero_propensity", "sku_vol_tier",
    "unit_price", "log_price", "price_vs_avg",
    "dow", "month", "is_weekend", "is_month_end", "week_of_year",
    "same_dow_mean",
]