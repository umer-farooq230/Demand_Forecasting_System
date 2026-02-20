"""
data_ingestion.py
─────────────────
Loads and performs initial cleaning of the Online Retail CSV.
Logs data quality stats to MLflow if an active run exists.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = "Online Retail.csv"


def load_data(filename: str = RAW_FILE, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    filepath = Path(data_dir) / filename

    df = pd.read_csv(
        filepath,
        encoding="ISO-8859-1",
        dtype={"CustomerID": str, "InvoiceDate": str},
    )
    original_rows = len(df)

    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=True)
    df = df.dropna(subset=["StockCode", "InvoiceDate"])
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    cleaned_rows = len(df)
    removed_rows = original_rows - cleaned_rows

    print(
        f"[data_ingestion] {original_rows:,} rows → {cleaned_rows:,} clean "
        f"({removed_rows:,} removed) | "
        f"date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}"
    )

    # ── MLflow (no-op if no active run) ──────────────────────────────────────
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_params({
                "data_raw_rows":          original_rows,
                "data_cleaned_rows":      cleaned_rows,
                "data_removed_rows":      removed_rows,
                "data_date_min":          str(df["InvoiceDate"].min().date()),
                "data_date_max":          str(df["InvoiceDate"].max().date()),
                "data_n_skus_raw":        int(df["StockCode"].nunique()),
                "data_n_countries":       int(df["Country"].nunique()),
                "data_cancellation_rate": round(removed_rows / original_rows, 4),
            })
    except Exception:
        pass

    return df