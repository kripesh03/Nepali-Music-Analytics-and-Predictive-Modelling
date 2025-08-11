# save as train_popularity_models.py
# Usage:
#   python train_popularity_models.py --csv ./data/musicdata_cleaned.csv
# Prints RMSE and R^2 (%) for each model in the terminal,
# saves the best model + scaler into ./models/, and writes a metrics CSV.

import argparse
import json
import os
import sys
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)
pd.set_option("display.width", 160)


def parse_args():
    ap = argparse.ArgumentParser(description="Train and compare ML models for song popularity prediction.")
    ap.add_argument("--csv", type=str, default="./data/musicdata_cleaned.csv", help="Path to CSV dataset.")
    ap.add_argument("--target", type=str, default="Pop.", help="Target column for popularity.")
    ap.add_argument(
        "--features",
        type=str,
        default="BPM,Energy,Dance,Loud,Valence,Length,Acoustic",
        help="Comma-separated list of feature columns.",
    )
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--models_dir", type=str, default="./models", help="Directory to save models/scaler/metrics.")
    return ap.parse_args()


def to_seconds_length(col: pd.Series) -> pd.Series:
    """
    Converts a 'Length' column that may be in formats like '3:45', '04:01', '241' (seconds), or '4:25'
    into total seconds (float). If it's already numeric, return as-is.
    """
    def parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        # If strictly numeric, treat as seconds
        if s.replace(".", "", 1).isdigit():
            try:
                return float(s)
            except ValueError:
                return np.nan
        # Try mm:ss or h:mm:ss with pandas to_timedelta
        try:
            # If input missing hours, prepend '0:' safely
            if s.count(":") == 1:
                s = "0:" + s
            td = pd.to_timedelta(s, errors="coerce")
            return td.total_seconds() if pd.notna(td) else np.nan
        except Exception:
            return np.nan

    return col.apply(parse_one)


def build_models(random_state: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

    # Optional: include XGBoost / LightGBM / CatBoost if available
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            n_estimators=800,
            random_seed=random_state,
            verbose=False,
        )
    except Exception:
        pass

    return models


def main():
    args = parse_args()
    csv_path = args.csv
    target_col = args.target
    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    test_size = args.test_size
    seed = args.seed
    outdir = args.models_dir

    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: CSV not found at {csv_path}")

    os.makedirs(outdir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns in CSV: {missing}")

    # Robust Length parsing if present
    if "Length" in feature_cols:
        df["Length"] = to_seconds_length(df["Length"])

    # Clean NaNs
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_col])
    # Let the pipeline impute feature NaNs

    # Feature matrix and target
    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Preprocess: impute + scale numeric features
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)], remainder="drop"
    )

    # Build candidate models
    models = build_models(seed)

    results = []
    best_name = None
    best_pipe = None
    best_r2 = -np.inf

    print("\n=== Training & Evaluating Models ===")
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        r2_pct = r2 * 100.0  # "prediction percentage" shown as R^2 %

        results.append(
            {
                "model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "R2_percent": r2_pct,
            }
        )

        print(f"{name:16s} | RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | R^2: {r2:7.4f} ({r2_pct:6.2f}%)")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    # Save best model pipeline (includes scaler/imputer)
    best_model_path = os.path.join(outdir, f"{best_name}_best_model.pkl")
    joblib.dump(best_pipe, best_model_path)

    # Save a readable metrics report
    metrics_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    metrics_csv = os.path.join(outdir, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Also save JSON summary
    summary_json = {
        "best_model": best_name,
        "best_R2": best_r2,
        "best_R2_percent": best_r2 * 100.0,
        "saved_model_path": best_model_path,
        "metrics_csv": metrics_csv,
        "features": feature_cols,
        "target": target_col,
        "test_size": test_size,
        "random_seed": seed,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\n=== Summary ===")
    print(f"Best Model          : {best_name}")
    print(f"Best R^2            : {best_r2:.4f} ({best_r2*100:.2f}%)")
    print(f"Saved Model Path    : {best_model_path}")
    print(f"Metrics CSV         : {metrics_csv}")
    print(f"Note: The printed 'percentage' is R^2 × 100 for each model.\n")


if __name__ == "__main__":
    main()
