import os
import pytest
import pandas as pd
import joblib
from fastapi.testclient import TestClient
from src.app import *


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"
    assert json_data["model_status"] == "chargé"
    assert json_data["data_status"] == "chargée"


def test_predict_valid_sku(client):
    payload = {"sku": "SKU1"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["sku"] == "SKU1"
    # Le DummyModel retourne toujours 100.0
    assert json_data["predicted_price"] == 100.0
    assert "timestamp" in json_data


def test_predict_invalid_sku(client):
    payload = {"sku": "SKU_UNKNOWN"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 404


def test_reload_model(client):
    response = client.post("/reload-model")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["message"] == "Modèle rechargé avec succès"
