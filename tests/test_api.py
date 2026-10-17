import fastapi.testclient
import pytest

from price_api import NODES

client = fastapi.testclient.TestClient(__import__("price_api").app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_kharghar_2bhk():
    r = client.post("/predict", json={"node": "Kharghar", "sqft": 720, "bhk": 2})
    assert r.status_code == 200
    body = r.json()
    assert 3_000_000 < body["predicted_price"] < 12_000_000
    assert body["rate_per_sqft"] > 0


def test_predict_bmc_locality():
    r = client.post("/predict", json={"region": "Mumbai / BMC", "node": "Andheri", "sqft": 720, "bhk": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["region"] == "Mumbai / BMC"
    assert body["bmc_ward"] == "Western Suburbs"
    assert body["predicted_price"] > 0


def test_bmc_localities_and_ladder():
    locations = client.get("/localities", params={"region": "Mumbai / BMC"})
    assert locations.status_code == 200
    assert any(item["node"] == "Andheri" for item in locations.json()["localities"])
    ladder = client.get("/ladder/Andheri", params={"region": "Mumbai / BMC"})
    assert ladder.status_code == 200
    assert len(ladder.json()["ladder"]) == 5


def test_bmc_live_diagnostics():
    r = client.get("/diagnostics", params={"region": "Mumbai / BMC"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["confusion_matrix"]) == 4
    assert len(body["scatter"]["actual_lakh"]) > 100
    assert len(body["error_histogram"]["counts"]) == 20


def test_diagnostics_change_with_property_inputs():
    client_args = {"region": "Mumbai / BMC", "node": "Andheri", "bhk": 2, "is_new": True}
    compact = client.get("/diagnostics", params={**client_args, "sqft": 500}).json()
    large = client.get("/diagnostics", params={**client_args, "sqft": 1800}).json()
    assert compact["scatter"]["actual_lakh"] != large["scatter"]["actual_lakh"]


def test_predict_all_nodes():
    for node in NODES:
        r = client.post("/predict", json={"node": node, "sqft": 600, "bhk": 2})
        assert r.status_code == 200
        assert r.json()["predicted_price"] > 0


def test_resale_cheaper_than_new():
    new = client.post("/predict", json={"node": "Ulwe", "sqft": 700, "bhk": 2, "is_new": True}).json()
    old = client.post("/predict", json={"node": "Ulwe", "sqft": 700, "bhk": 2, "is_new": False}).json()
    assert old["predicted_price"] != new["predicted_price"]


def test_unknown_node_rejected():
    r = client.post("/predict", json={"node": "Mumbai Central", "sqft": 600, "bhk": 2})
    assert r.status_code == 422


def test_invalid_sqft_rejected():
    r = client.post("/predict", json={"node": "Vashi", "sqft": 10, "bhk": 2})
    assert r.status_code == 422


def test_ladder_ulwe():
    r = client.get("/ladder/Ulwe")
    assert r.status_code == 200
    ladder = r.json()["ladder"]
    assert len(ladder) == 5
    prices = [entry["predicted_price"] for entry in ladder]
    assert prices == sorted(prices)


def test_ladder_unknown_node_rejected():
    r = client.get("/ladder/Nowhere")
    assert r.status_code == 422
