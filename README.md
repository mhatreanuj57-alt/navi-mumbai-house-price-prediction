# Navi Mumbai / BMC House Price Prediction

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Author: Anuj Mhatre](https://img.shields.io/badge/Author-Anuj%20Mhatre-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-teal?logo=fastapi&logoColor=white)

> **Live Demo:** Updated Render URL will be added after direct deployment.

---

## Why We Built This

Home buyers and sellers in Navi Mumbai lack reliable, data-driven price estimates. Real estate agents often rely on gut feeling, and online portals show wildly inconsistent valuations for the same locality. We wanted to build a transparent, ML-powered tool that provides instant, explainable price predictions across 13 Navi Mumbai localities — trained on **1,400+ real listings** with validated cross-validation metrics.

---

## What It Does

- **Two-region finder** — choose **Navi Mumbai** or **Mumbai / BMC**, then choose a locality from the relevant dropdown before predicting
- **BMC localities** — 36 supported Mumbai localities trained separately on 22,895 cleaned BMC listings; the result identifies its BMC ward group

- **Instant Price Prediction** — Select locality, configuration (1RK–4BHK), carpet area, and condition (new/resale) to get an ML-powered price estimate with per-sqft rate
- **Price Ladder** — Compare all 5 configurations side-by-side for any locality with an interactive bar chart
- **Interactive Map** — Visual locality coordinates on a Streamlit map for geographic context
- **Model Insights Tab** — 5-fold cross-validation metrics: MAE (~₹16.8L), R² score, band classification accuracy, confusion matrix, predicted-vs-actual scatter, and error distribution histogram
- **REST API** — FastAPI backend with `/predict`, `/ladder/{node}`, and `/health` endpoints for programmatic access
- **13 Localities** — Airoli, Belapur, Dronagiri, Ghansoli, Kamothe, Kharghar, Nerul, Panvel, Sanpada, Seawoods, Taloja, Ulwe, Vashi

---

## How It Was Built

| Component | Technology |
|-----------|-----------|
| **ML Model** | HistGradientBoostingRegressor (scikit-learn) |
| **Data** | 1,400+ real Navi Mumbai property listings |
| **Preprocessing** | Custom location normalization, amenity scoring, outlier filtering |
| **Backend** | FastAPI with Pydantic validation |
| **Frontend** | Streamlit with custom CSS (Inter font, gradient cards) |
| **Containerization** | Docker (Python 3.12-slim) |
| **Deployment** | Render (Streamlit app) |

**Architecture:** The model pipeline chains `ColumnTransformer` (passthrough numeric + OneHotEncoder for locality) into `HistGradientBoostingRegressor`. The trained model is serialized as `nm_model.joblib` and loaded by both the Streamlit app and FastAPI server.

### Mumbai / BMC data

The BMC model is trained separately from the Navi Mumbai model on the public [Mumbai House Price Data (70k Entries)](https://www.kaggle.com/datasets/kevinnadar22/mumbai-house-price-data-70k-entries) dataset (2024 Makaan listings, Apache-2.0). The cleaning pipeline retains only BMC localities with at least 50 usable listings, producing **22,895 listings across 36 localities**. The UI exposes the locality first, then labels each prediction with its broad BMC ward group.

---

## Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/anu-mhatre-1812/navi-mumbai-house-price-prediction.git
cd navi-mumbai-house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Recreate the Mumbai/BMC clean dataset and model after obtaining the source CSV
python prepare_mumbai_bmc.py
python train_mumbai_bmc.py

# Run the Streamlit app
streamlit run streamlit_ui.py

# Run the FastAPI server (in a separate terminal)
uvicorn price_api:app --reload --port 8000
```

The Streamlit app opens at `http://localhost:8501` and the API docs at `http://localhost:8000/docs`.

### Docker

```bash
# Build the Docker image (serves the FastAPI backend)
docker build -t navi-mumbai-api .

# Run the container
docker run -p 8000:8000 navi-mumbai-api
```

API available at `http://localhost:8000`.

---

## Project Structure

```
navi-mumbai-house-price-prediction/
├── streamlit_ui.py            # Streamlit web interface (Price Finder + Model Insights)
├── price_api.py              # FastAPI REST API (predict, ladder, health)
├── train_navimumbai.py       # Model training with 5-fold CV
├── prepare_navimumbai.py     # Data preprocessing & cleaning pipeline
├── data/
│   └── navimumbai_listings.csv   # Cleaned dataset (1,400+ listings)
├── models/
│   └── nm_model.joblib       # Serialized trained model
├── tests/
│   └── test_api.py           # 8 API tests
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Dev/test dependencies
├── Dockerfile                # Container config (FastAPI on port 8000)
├── render.yaml               # Render deployment config
├── Makefile                  # Dev shortcuts
└── .streamlit/               # Streamlit config
```

---

## API Reference

### `POST /predict`

Predict house price for a listing.

```json
{
  "node": "Kharghar",
  "sqft": 720,
  "bhk": 2,
  "is_new": true,
  "amenities": null
}
```

**Response:**
```json
{
  "node": "Kharghar",
  "sqft": 720,
  "bhk": 2,
  "is_new": true,
  "predicted_price": 6500000.0,
  "price_formatted": "Rs 65.0 Lakh",
  "rate_per_sqft": 9027.78
}
```

### `GET /ladder/{node}`

Get price ladder (all configurations) for a locality.

```
GET /ladder/Kharghar?is_new=true
```

**Response:**
```json
{
  "node": "Kharghar",
  "is_new": true,
  "ladder": [
    {"config": "1RK", "typical_sqft": 300, "predicted_price": 3200000, "price_formatted": "Rs 32.0 Lakh", "rate_per_sqft": 10666.67},
    ...
  ]
}
```

### `GET /health`

Health check endpoint.

```json
{"status": "ok", "model_loaded": true}
```

---

## Deployment

### Render (Recommended)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Runtime:** Python 3.13.14
   - **Build Command:** `bash build.sh`
   - **Start Command:** `python streamlit_ui.py`
5. Deploy the latest commit on `main`. For an existing manually created service, update these settings in the Render dashboard; committing `render.yaml` does not update that service automatically.
6. If a previous build used different dependencies, choose **Clear build cache & deploy**. The build script installs dependencies using the same Python interpreter and checks imports before deployment.

### Docker on Any Cloud

```bash
docker build -t navi-mumbai-api .
docker run -d -p 8000:8000 --name price-api navi-mumbai-api
```

---

## Test Results

```
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict_kharghar_2bhk PASSED
tests/test_api.py::test_predict_all_nodes PASSED
tests/test_api.py::test_resale_cheaper_than_new PASSED
tests/test_api.py::test_unknown_node_rejected PASSED
tests/test_api.py::test_invalid_sqft_rejected PASSED
tests/test_api.py::test_ladder_ulwe PASSED
tests/test_api.py::test_ladder_unknown_node_rejected PASSED

============================== 8 passed ==============================
```

Tests cover: health check, prediction accuracy for Kharghar 2BHK, all 13 nodes, resale-vs-new pricing, input validation (unknown node, invalid sqft), price ladder completeness, and unknown node rejection.

---

## Author

**Anuj Mhatre**
- GitHub: [anu-mhatre-1812](https://github.com/anu-mhatre-1812)
- Email: anujmhatre125@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for details.
