"""Train the Mumbai/BMC price model from the cleaned BMC-only dataset."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "mumbai_bmc_listings.csv"
OUT = ROOT / "models" / "bmc_model.joblib"
NUM_COLS = ["sqft", "bhk", "is_new", "age_years"]
CAT_COLS = ["node", "bmc_ward"]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=400, max_depth=7,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[NUM_COLS + CAT_COLS], df["price"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(build_pipeline(), X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    pipe = build_pipeline().fit(X, y)
    pred = pipe.predict(X)
    print(f"CV MAE: Rs {np.mean(scores) / 100000:.2f}L (+/- {np.std(scores) / 100000:.2f}L)")
    print(f"Train R2: {r2_score(y, pred):.4f} | Train MAE: Rs {mean_absolute_error(y, pred) / 100000:.2f}L")
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": pipe, "num_cols": NUM_COLS, "cat_cols": CAT_COLS}, OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
