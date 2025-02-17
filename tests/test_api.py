import os
import pytest
import pandas as pd
import joblib
from fastapi.testclient import TestClient
from src.api import app, load_latest_model, data, model


# Client de test pour FastAPI
@pytest.fixture
def client():
    return TestClient(app)


# Mock des fichiers nécessaires
@pytest.fixture
def mock_latest_model_file(tmp_path):
    """Crée un fichier latest_model.txt pointant vers un modèle factice"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_path = model_dir / "fake_model.pkl"
    joblib.dump("fake_model", model_path)  # Sauvegarde un faux modèle

    latest_model_txt = model_dir / "latest_model.txt"
    latest_model_txt.write_text(model_path.name)

    return latest_model_txt


@pytest.fixture
def mock_preprocessed_data(tmp_path):
    """Crée un fichier CSV prétraité fictif"""
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    df = pd.DataFrame(
        {
            "SKU": ["A123", "B456"],
            "PrixInitial": [100.0, 150.0],
            "AgeProduitEnJours": [30, 45],
            "QuantiteVendue": [10, 5],
            "UtiliteProduit": [0.8, 0.6],
            "ElasticitePrix": [0.5, 0.7],
            "Remise": [0.1, 0.2],
            "Qualite": [0.9, 1.0],
            "Timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )
    file_path = data_dir / "preprocessed_data.csv"
    df.to_csv(file_path, index=False)

    latest_preprocessed_txt = data_dir / "latest_preprocessed.txt"
    latest_preprocessed_txt.write_text(str(file_path))

    return latest_preprocessed_txt


def test_health_endpoint(client):
    """Vérifie que l'endpoint /health fonctionne"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_status_endpoint(client, mock_latest_model_file, mock_preprocessed_data):
    """Vérifie que /status retourne bien l'état du modèle et des données"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()

    assert "model_status" in data
    assert "data_status" in data
    assert data["model_status"] in ["chargé", "non chargé"]
    assert data["data_status"] in ["chargées", "non chargées"]


def test_predict_invalid_sku(client, mock_preprocessed_data):
    """Vérifie que /predict renvoie une erreur si le SKU est inconnu"""
    global data
    data = pd.read_csv(mock_preprocessed_data)  # Charger les données mockées
    response = client.post("/predict", json={"sku": "Z999"})  # SKU inconnu
    assert response.status_code == 404
    assert "SKU non trouvé" in response.json()["detail"]


def test_reload_model(client, mock_latest_model_file):
    """Vérifie que /reload-model recharge le modèle"""
    response = client.post("/reload-model")
    assert response.status_code == 200
    assert response.json() == {"message": "Modèle rechargé avec succès"}
