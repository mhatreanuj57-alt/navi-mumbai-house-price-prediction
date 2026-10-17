from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).parent
REGIONS = ("Navi Mumbai", "Mumbai / BMC")
NODES = ("Airoli", "Belapur", "Dronagiri", "Ghansoli", "Kamothe", "Kharghar", "Nerul", "Panvel", "Sanpada", "Seawoods", "Taloja", "Ulwe", "Vashi")
CONFIGS = {"1RK": {"bhk": 1, "sqft": 300}, "1BHK": {"bhk": 1, "sqft": 450}, "2BHK": {"bhk": 2, "sqft": 720}, "3BHK": {"bhk": 3, "sqft": 1075}, "4BHK": {"bhk": 4, "sqft": 1520}}
NM_COORDS = {"Kharghar": (19.0968, 73.0741), "Ulwe": (18.9636, 73.0575), "Kamothe": (19.1166, 73.0892), "Panvel": (18.9894, 73.1175), "Taloja": (19.0797, 73.1213), "Seawoods": (19.0174, 73.0228), "Belapur": (19.0059, 73.0329), "Ghansoli": (19.1176, 73.0063), "Sanpada": (19.0645, 73.0094), "Airoli": (19.1556, 72.9987), "Dronagiri": (18.9044, 72.9976), "Nerul": (19.0348, 73.0197), "Vashi": (19.0771, 72.9986)}


def fmt_inr(price: float) -> str:
    return f"Rs {price / 1e7:.2f} Cr" if price >= 1e7 else f"Rs {price / 1e5:.1f} Lakh"


@lru_cache(maxsize=2)
def get_bundle(region: str):
    return joblib.load(ROOT / "models" / ("nm_model.joblib" if region == "Navi Mumbai" else "bmc_model.joblib"))


@lru_cache(maxsize=2)
def get_location_data(region: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / ("navimumbai_listings.csv" if region == "Navi Mumbai" else "mumbai_bmc_listings.csv"))


class Listing(BaseModel):
    region: Literal["Navi Mumbai", "Mumbai / BMC"] = "Navi Mumbai"
    node: str
    sqft: float = Field(gt=50, le=10_000)
    bhk: int = Field(ge=1, le=7)
    is_new: bool = True
    amenities: int | None = Field(default=None, ge=0, le=14)

    @field_validator("node")
    @classmethod
    def node_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("node cannot be empty")
        return value.strip()


class Prediction(BaseModel):
    region: str
    node: str
    bmc_ward: str | None = None
    sqft: float
    bhk: int
    is_new: bool
    predicted_price: float
    price_formatted: str
    rate_per_sqft: float


app = FastAPI(title="Mumbai Home Price API", description="Navi Mumbai and BMC locality price predictions", version="2.0.0")


@app.get("/")
def root():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": all(get_bundle(region) is not None for region in REGIONS)}


@app.get("/localities")
def localities(region: Literal["Navi Mumbai", "Mumbai / BMC"]):
    df = get_location_data(region)
    if region == "Navi Mumbai":
        locations = [{"node": node, "latitude": NM_COORDS[node][0], "longitude": NM_COORDS[node][1]} for node in sorted(NM_COORDS)]
    else:
        locations = (df.groupby(["node", "bmc_ward"], as_index=False)[["latitude", "longitude"]].median().sort_values("node").to_dict("records"))
    return {"region": region, "localities": locations}


@app.get("/insights")
def insights(region: Literal["Navi Mumbai", "Mumbai / BMC"] = "Navi Mumbai"):
    df = get_location_data(region)
    details = {
        "Navi Mumbai": {"cv_mae_lakh": 16.8, "r2": None, "model": "HistGradientBoostingRegressor", "note": "5-fold cross-validation MAE from the original Navi Mumbai training run."},
        "Mumbai / BMC": {"cv_mae_lakh": 58.26, "r2": 0.8724, "model": "HistGradientBoostingRegressor", "note": "5-fold cross-validation MAE; R² is the training fit metric from the BMC training run."},
    }[region]
    return {"region": region, "listings": len(df), "localities": int(df["node"].nunique()), **details}


def get_diagnostics(region: str, node: str | None = None, sqft: float = 720, bhk: int = 2, is_new: bool = True):
    """Input-responsive diagnostics from closest historical comparables."""
    df, bundle = get_location_data(region), get_bundle(region)
    if node and node in set(df["node"]):
        candidates = df[df["node"] == node].copy()
        # Keep the historical records most comparable to this exact request.
        candidates["_distance"] = (
            (candidates["sqft"] - sqft).abs() / 100
            + (candidates["bhk"] - bhk).abs() * 10
            + (candidates["is_new"] - int(is_new)).abs() * 2
        )
        df = candidates.nsmallest(min(350, len(candidates)), "_distance").drop(columns="_distance")
    X, y = df[bundle["num_cols"] + bundle["cat_cols"]], df["price"].to_numpy()
    predicted = bundle["model"].predict(X)
    edges = np.quantile(y, [0, 0.25, 0.5, 0.75, 1.0])
    actual_band = np.digitize(y, edges[1:-1], right=True)
    predicted_band = np.digitize(predicted, edges[1:-1], right=True)
    matrix = [[int(np.sum((actual_band == row) & (predicted_band == col))) for col in range(4)] for row in range(4)]
    residual_lakh = (y - predicted) / 100_000
    hist, hist_edges = np.histogram(residual_lakh, bins=20)
    sample = np.linspace(0, len(y) - 1, min(350, len(y)), dtype=int)
    return {
        "comparable_listings": len(df),
        "mae_lakh": round(float(np.mean(np.abs(y - predicted)) / 100_000), 2),
        "r2": round(float(1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
        "band_accuracy": round(float(np.mean(actual_band == predicted_band) * 100), 1),
        "bands": ["Budget", "Mid-range", "Premium", "Luxury"],
        "confusion_matrix": matrix,
        "scatter": {"actual_lakh": (y[sample] / 100_000).round(2).tolist(), "predicted_lakh": (predicted[sample] / 100_000).round(2).tolist()},
        "error_histogram": {"edges_lakh": hist_edges.round(2).tolist(), "counts": hist.tolist()},
    }


@app.get("/diagnostics")
def diagnostics(
    region: Literal["Navi Mumbai", "Mumbai / BMC"] = "Navi Mumbai",
    node: str | None = None,
    sqft: float = 720,
    bhk: int = 2,
    is_new: bool = True,
):
    return {"region": region, "node": node, "method": "live diagnostic using the closest historical comparables for the current selection", **get_diagnostics(region, node, sqft, bhk, is_new)}


def estimate(region: str, node: str, sqft: float, bhk: int, is_new: bool, amenities: int | None = None):
    df = get_location_data(region)
    if node not in set(df["node"]):
        raise HTTPException(status_code=422, detail=f"Unknown locality for {region}: {node}")
    local = df[df["node"] == node]
    if region == "Navi Mumbai":
        row, ward = {"sqft": sqft, "bhk": bhk, "is_new": int(is_new), "amenities": amenities if amenities is not None else int(local["amenities"].median()), "node": node}, None
    else:
        ward = str(local["bmc_ward"].mode().iloc[0])
        resale_age = int(local.loc[local["is_new"] == 0, "age_years"].median()) if (local["is_new"] == 0).any() else 5
        row = {"sqft": sqft, "bhk": bhk, "is_new": int(is_new), "age_years": 0 if is_new else resale_age, "node": node, "bmc_ward": ward}
    return float(get_bundle(region)["model"].predict(pd.DataFrame([row]))[0]), ward


@app.post("/predict", response_model=Prediction)
def predict(listing: Listing):
    price, ward = estimate(listing.region, listing.node, listing.sqft, listing.bhk, listing.is_new, listing.amenities)
    return Prediction(region=listing.region, node=listing.node, bmc_ward=ward, sqft=listing.sqft, bhk=listing.bhk, is_new=listing.is_new, predicted_price=round(price, 2), price_formatted=fmt_inr(price), rate_per_sqft=round(price / listing.sqft, 2))


@app.get("/ladder/{node}")
def ladder(node: str, region: Literal["Navi Mumbai", "Mumbai / BMC"] = "Navi Mumbai", is_new: bool = True):
    out = []
    for name, cfg in CONFIGS.items():
        price, _ = estimate(region, node, cfg["sqft"], cfg["bhk"], is_new)
        out.append({"config": name, "typical_sqft": cfg["sqft"], "predicted_price": round(price, 2), "price_formatted": fmt_inr(price), "rate_per_sqft": round(price / cfg["sqft"], 2)})
    return {"region": region, "node": node, "is_new": is_new, "ladder": out}
