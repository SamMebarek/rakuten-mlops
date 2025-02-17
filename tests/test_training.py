import os
import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from src.training import (
    is_mlflow_active,
    get_latest_preprocessed_file,
    main,
    LATEST_PREPROCESSED_FILE,
    MODEL_DIR,
)


@pytest.fixture
def sample_training_data(tmp_path):
    """Fixture pour créer un fichier de données d'entraînement temporaire."""
    test_file = tmp_path / "test_preprocessed.csv"
    df = pd.DataFrame(
        {
            "SKU": ["A", "B"],
            "Prix": [10.5, 15.3],
            "PrixInitial": [9.5, 14.0],
            "Timestamp": ["2024-01-01", "2024-01-02"],
            "AgeProduitEnJours": [5, 10],
            "QuantiteVendue": [3, 6],
            "UtiliteProduit": [0.8, 0.9],
            "ElasticitePrix": [0.5, 0.6],
            "Remise": [0.1, 0.2],
            "Qualite": [1.0, 1.0],
            "Mois_sin": [0.5, 0.6],
            "Mois_cos": [0.8, 0.7],
            "Heure_sin": [0.3, 0.4],
            "Heure_cos": [0.9, 0.8],
        }
    )
    df.to_csv(test_file, index=False)
    return test_file


@pytest.fixture
def mock_latest_preprocessed_file(tmp_path, sample_training_data):
    """Fixture pour créer un fichier latest_preprocessed.txt pointant vers les données de test."""
    latest_file = tmp_path / "latest_preprocessed.txt"
    latest_file.write_text(str(sample_training_data))
    return latest_file


@pytest.fixture
def mock_model_directory(tmp_path):
    """Fixture pour un dossier temporaire de modèles."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# ---- Test de `is_mlflow_active()` ----
def test_is_mlflow_active():
    """Vérifie que `is_mlflow_active()` retourne un booléen et gère MLflow actif/inactif."""
    with patch("requests.get") as mock_get:
        # Simuler un serveur MLflow actif
        mock_get.return_value.status_code = 200
        assert is_mlflow_active() is True

        # Simuler un serveur MLflow inactif
        mock_get.side_effect = requests.exceptions.RequestException
        assert is_mlflow_active() is False


# ---- Test de `get_latest_preprocessed_file()` ----
def test_get_latest_preprocessed_file(mock_latest_preprocessed_file, monkeypatch):
    """Vérifie que `get_latest_preprocessed_file()` récupère bien le dernier fichier stocké."""
    monkeypatch.setattr(
        "src.training.LATEST_PREPROCESSED_FILE", str(mock_latest_preprocessed_file)
    )
    latest_file = get_latest_preprocessed_file()
    assert latest_file == str(mock_latest_preprocessed_file.read_text()).strip()


def test_get_latest_preprocessed_file_missing(tmp_path, monkeypatch):
    """Vérifie le comportement si `latest_preprocessed.txt` est absent."""
    missing_file = tmp_path / "missing_latest_preprocessed.txt"
    monkeypatch.setattr("src.training.LATEST_PREPROCESSED_FILE", str(missing_file))

    with pytest.raises(FileNotFoundError, match="Fichier de référence .* introuvable"):
        get_latest_preprocessed_file()


# ---- Test du chargement des données ----
def test_training_data_loading(sample_training_data):
    """Vérifie que les données d'entraînement se chargent correctement."""
    df = pd.read_csv(sample_training_data)
    assert not df.empty, "Le fichier CSV chargé ne doit pas être vide."
    assert "Prix" in df.columns, "La colonne 'Prix' doit être présente."
    assert "SKU" in df.columns, "La colonne 'SKU' doit être présente."


def test_training_data_loading_missing(tmp_path):
    """Vérifie que le chargement échoue si le fichier est absent."""
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        pd.read_csv(missing_file)


# ---- Test de l'exécution complète du pipeline ----
def test_pipeline_execution(
    mock_latest_preprocessed_file, mock_model_directory, monkeypatch
):
    """Vérifie que `main()` s'exécute sans erreur et produit des logs."""
    monkeypatch.setattr(
        "src.training.LATEST_PREPROCESSED_FILE", str(mock_latest_preprocessed_file)
    )
    monkeypatch.setattr("src.training.MODEL_DIR", str(mock_model_directory))

    with patch("src.training.logger.info") as mock_logger:
        try:
            main()
        except Exception as e:
            pytest.fail(
                f"L'exécution du pipeline a levé une exception inattendue : {e}"
            )

        # Vérifier qu'au moins un log a été écrit
        assert mock_logger.called, "Le pipeline doit générer des logs."
