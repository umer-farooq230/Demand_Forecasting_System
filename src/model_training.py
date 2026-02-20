"""
model_training.py
─────────────────
Three-stage demand forecasting pipeline with MLflow experiment tracking.

  Stage 1  – Zero / nonzero classifier       (LightGBM)
  Stage 2  – Low / high classifier            (LightGBM, conditioned on nonzero)
  Stage 3a – High-volume ensemble             (XGBoost + LightGBM on log-target)
  Stage 3b – Low-volume L1 regressor          (LightGBM)

MLflow logs: all hyperparameters, zero-gate threshold, and model artifacts.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import FEATURES

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

LOW_THRESHOLD = 5


def _lgb_callbacks(patience: int = 50):
    return [early_stopping(patience, verbose=False)]


def _tune_zero_threshold(
    prob_nz: np.ndarray,
    y_val: pd.Series,
    X_val: pd.DataFrame,
    models_partial: dict,
) -> float:
    """
    Sweep thresholds on real pipeline predictions and return the one that
    minimises MAE on the validation set.
    """
    prob_hi  = models_partial["lh_clf"].predict_proba(X_val)[:, 1]
    hi_log   = (models_partial["xgb_hi"].predict(X_val) +
                models_partial["lgb_hi"].predict(X_val)) / 2
    hi_p     = np.expm1(hi_log).clip(0)
    lo_p     = models_partial["lgb_lo"].predict(X_val).clip(0, LOW_THRESHOLD)
    nz_preds = prob_hi * hi_p + (1 - prob_hi) * lo_p

    best_thresh, best_mae = 0.5, float("inf")
    for t in np.arange(0.10, 0.80, 0.025):
        gated = np.where(prob_nz >= t, nz_preds, 0.0)
        m = mean_absolute_error(y_val, gated)
        if m < best_mae:
            best_mae, best_thresh = m, t

    print(
        f"  → threshold={best_thresh:.3f} | "
        f"pred zero-rate={(prob_nz < best_thresh).mean():.2%} | "
        f"actual zero-rate={(y_val == 0).mean():.2%} | "
        f"val MAE={best_mae:.4f}"
    )
    return best_thresh


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, save: bool = True) -> dict:
    X_tr,  y_tr  = train_df[FEATURES], train_df["sales"]
    X_val, y_val = val_df[FEATURES],   val_df["sales"]

    # ── Hyperparameters (logged to MLflow) ────────────────────────────────────
    hparams = dict(
        low_threshold            = LOW_THRESHOLD,
        nz_clf_n_estimators      = 1000,
        nz_clf_lr                = 0.02,
        lh_clf_n_estimators      = 1000,
        lh_clf_lr                = 0.02,
        xgb_hi_n_estimators      = 2000,
        xgb_hi_lr                = 0.01,
        xgb_hi_max_depth         = 6,
        xgb_hi_subsample         = 0.8,
        xgb_hi_colsample_bytree  = 0.7,
        lgb_hi_n_estimators      = 2000,
        lgb_hi_num_leaves        = 63,
        lgb_hi_max_depth         = 7,
        lgb_hi_lr                = 0.01,
        lgb_lo_n_estimators      = 1000,
        lgb_lo_lr                = 0.02,
        lgb_lo_max_depth         = 5,
    )

    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_params(hparams)
            mlflow.log_params({
                "train_rows":      len(train_df),
                "val_rows":        len(val_df),
                "train_zero_rate": round((y_tr == 0).mean(), 4),
                "val_zero_rate":   round((y_val == 0).mean(), 4),
                "n_features":      len(FEATURES),
            })
    except Exception:
        pass

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print("[train] Stage 1: Zero/Nonzero classifier …")
    nz_clf = LGBMClassifier(
        n_estimators=hparams["nz_clf_n_estimators"],
        learning_rate=hparams["nz_clf_lr"],
        is_unbalance=True,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    nz_clf.fit(
        X_tr, (y_tr > 0).astype(int),
        eval_set=[(X_val, (y_val > 0).astype(int))],
        callbacks=_lgb_callbacks(),
    )
    prob_nz_val = nz_clf.predict_proba(X_val)[:, 1]

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("[train] Stage 2: Low/High classifier …")
    nz_tr, nz_val = y_tr > 0, y_val > 0
    lh_clf = LGBMClassifier(
        n_estimators=hparams["lh_clf_n_estimators"],
        learning_rate=hparams["lh_clf_lr"],
        is_unbalance=True,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lh_clf.fit(
        X_tr[nz_tr], (y_tr[nz_tr] > LOW_THRESHOLD).astype(int),
        eval_set=[(X_val[nz_val], (y_val[nz_val] > LOW_THRESHOLD).astype(int))],
        callbacks=_lgb_callbacks(),
    )

    # ── Stage 3a ──────────────────────────────────────────────────────────────
    print("[train] Stage 3a: High-volume regressor …")
    hi_tr, hi_val = y_tr > LOW_THRESHOLD, y_val > LOW_THRESHOLD
    y_tr_log      = np.log1p(y_tr[hi_tr])
    y_val_log     = np.log1p(y_val[hi_val])
    raw_weights   = np.log1p(train_df["sku_avg_volume"])
    weights_hi    = raw_weights[hi_tr].values

    xgb_hi = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=hparams["xgb_hi_n_estimators"],
        learning_rate=hparams["xgb_hi_lr"],
        max_depth=hparams["xgb_hi_max_depth"],
        subsample=hparams["xgb_hi_subsample"],
        colsample_bytree=hparams["xgb_hi_colsample_bytree"],
        min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=50, eval_metric="mae",
        random_state=42, n_jobs=-1,
    )
    xgb_hi.fit(
        X_tr[hi_tr], y_tr_log,
        eval_set=[(X_val[hi_val], y_val_log)],
        sample_weight=weights_hi,
        verbose=False,
    )

    lgb_hi = LGBMRegressor(
        objective="regression_l1",
        n_estimators=hparams["lgb_hi_n_estimators"],
        num_leaves=hparams["lgb_hi_num_leaves"],
        max_depth=hparams["lgb_hi_max_depth"],
        min_child_samples=30,
        learning_rate=hparams["lgb_hi_lr"],
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_hi.fit(
        X_tr[hi_tr], y_tr_log,
        eval_set=[(X_val[hi_val], y_val_log)],
        sample_weight=weights_hi,
        callbacks=_lgb_callbacks(),
    )

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    print("[train] Stage 3b: Low-volume regressor …")
    lo_tr  = (y_tr  > 0) & (y_tr  <= LOW_THRESHOLD)
    lo_val = (y_val > 0) & (y_val <= LOW_THRESHOLD)
    lgb_lo = LGBMRegressor(
        objective="regression_l1",
        n_estimators=hparams["lgb_lo_n_estimators"],
        learning_rate=hparams["lgb_lo_lr"],
        max_depth=hparams["lgb_lo_max_depth"],
        min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_lo.fit(
        X_tr[lo_tr], y_tr[lo_tr],
        eval_set=[(X_val[lo_val], y_val[lo_val])],
        callbacks=_lgb_callbacks(),
    )

    # ── Threshold tuning ──────────────────────────────────────────────────────
    print("[train] Tuning zero-gate threshold …")
    models_partial = dict(lh_clf=lh_clf, xgb_hi=xgb_hi, lgb_hi=lgb_hi, lgb_lo=lgb_lo)
    zero_thresh    = _tune_zero_threshold(prob_nz_val, y_val, X_val, models_partial)

    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_param("zero_gate_threshold", round(zero_thresh, 3))
    except Exception:
        pass

    models = dict(
        nz_clf=nz_clf, lh_clf=lh_clf,
        xgb_hi=xgb_hi, lgb_hi=lgb_hi, lgb_lo=lgb_lo,
        zero_thresh=zero_thresh,
    )

    if save:
        path = MODELS_DIR / "pipeline.pkl"
        joblib.dump(models, path)
        print(f"[train] Saved → {path}")

        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_artifact(str(path), artifact_path="model")
        except Exception:
            pass

    return models


def predict(X: pd.DataFrame, models: dict) -> np.ndarray:
    prob_nz = models["nz_clf"].predict_proba(X)[:, 1]
    prob_hi = models["lh_clf"].predict_proba(X)[:, 1]

    hi_log  = (models["xgb_hi"].predict(X) + models["lgb_hi"].predict(X)) / 2
    hi_pred = np.expm1(hi_log).clip(0)
    lo_pred = models["lgb_lo"].predict(X).clip(0, LOW_THRESHOLD)

    nonzero_demand = prob_hi * hi_pred + (1 - prob_hi) * lo_pred
    return np.where(prob_nz >= models["zero_thresh"], nonzero_demand, 0.0)