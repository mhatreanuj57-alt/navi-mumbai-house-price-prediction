import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = "data/navimumbai_listings.csv"
OUT = "models/nm_model.joblib"

num_cols = ["sqft", "bhk", "is_new", "amenities"]
cat_cols = ["node"]
target = "price"


def build_pipeline():
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=400, max_depth=6,
        min_samples_leaf=10, l2_regularization=0.1, random_state=42,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = pd.read_csv(DATA)
    X = df[num_cols + cat_cols]
    y = df[target]

    pipe = build_pipeline()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(pipe, X, y, cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
    print(f"CV MAE: Rs {np.mean(scores)/100000:.2f}L (+/- {np.std(scores)/100000:.2f}L)")

    pipe.fit(X, y)
    pred = pipe.predict(X)
    print(f"Train R2: {r2_score(y, pred):.4f}  |  Train MAE: Rs {mean_absolute_error(y, pred)/100000:.2f}L")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": pipe, "num_cols": num_cols, "cat_cols": cat_cols}, OUT)
    print(f"Saved -> {OUT} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
