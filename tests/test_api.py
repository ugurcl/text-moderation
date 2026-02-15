from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_seconds" in data
    assert data["model_loaded"] is True
    assert data["database_connected"] is True


def test_predict():
    r = client.post("/predict", json={"text": "Samsung Galaxy S24"})
    assert r.status_code == 200
    data = r.json()
    assert "label" in data
    assert "confidence" in data
    assert "allowed" in data


def test_predict_response_types():
    r = client.post("/predict", json={"text": "Nike Air Max 270"})
    data = r.json()
    assert isinstance(data["label"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["allowed"], bool)


def test_predict_empty_text():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_missing_text():
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_batch():
    r = client.post("/predict/batch", json={"texts": ["Samsung Galaxy", "test"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2


def test_predict_batch_empty():
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


def test_stats():
    client.post("/predict", json={"text": "test for stats"})
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "blocked" in data
    assert "allowed" in data
    assert "by_label" in data


def test_history():
    r = client.get("/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_history_limit():
    r = client.get("/history?limit=5")
    assert r.status_code == 200
    assert len(r.json()) <= 5


def test_explain():
    r = client.post("/predict/explain", json={"text": "Samsung Galaxy S24"})
    assert r.status_code == 200
    data = r.json()
    assert "label" in data
    assert "probabilities" in data
    assert "top_features" in data


def test_explain_empty_text():
    r = client.post("/predict/explain", json={"text": ""})
    assert r.status_code == 422
