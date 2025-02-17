import pytest
import os
import pandas as pd
import numpy as np
from src.preprocessing import (
    get_latest_file,
    drop_unused_columns,
    convert_types,
    add_time_features,
    preprocessing_pipeline,
    run_preprocessing,
)

# Chemin fictif pour les tests
TEST_LATEST_INGESTED = "tests/data/latest_ingested.txt"
TEST_LATEST_PREPROCESSED = "tests/data/latest_preprocessed.txt"


@pytest.fixture
def sample_dataframe():
    """Fixture : retourne un DataFrame de test avec les colonnes nécessaires."""
    return pd.DataFrame(
        {
            "SKU": ["A", "B"],
            "Timestamp": pd.to_datetime(["2024-01-01 12:00:00", "2024-06-01 18:00:00"]),
            "PrixInitial": [10.5, 15.3],
            "Prix": [12, 18],
            "AgeProduitEnJours": [5, 10],
            "QuantiteVendue": [2, 3],
            "UtiliteProduit": [0.5, 0.6],
            "ElasticitePrix": [0.7, 0.8],
            "Remise": [0.1, 0.2],
            "Qualite": [0.9, 1.0],
        }
    )


@pytest.fixture
def tmp_latest_ingested_file(tmp_path):
    """Fixture : crée un fichier latest_ingested.txt temporaire."""
    test_file = tmp_path / "latest_ingested.txt"
    test_file.write_text("Data/test_file.csv")
    return test_file


def test_get_latest_file(tmp_latest_ingested_file):
    """Vérifie la récupération du dernier fichier stocké."""
    assert get_latest_file(str(tmp_latest_ingested_file)) == "Data/test_file.csv"


def test_drop_unused_columns(sample_dataframe):
    """Vérifie que les colonnes inutiles sont bien supprimées."""
    df_cleaned = drop_unused_columns(sample_dataframe)
    assert "DateLancement" not in df_cleaned.columns
    assert "PrixPlancher" not in df_cleaned.columns
    assert "Prix" in df_cleaned.columns


def test_convert_types(sample_dataframe):
    """Vérifie la conversion correcte des types de colonnes."""
    df_converted = convert_types(sample_dataframe)

    # Vérifier que les types sont bien convertis
    assert df_converted["PrixInitial"].dtype == "float64"
    assert df_converted["Prix"].dtype == "float64"
    assert df_converted["AgeProduitEnJours"].dtype == "int64"
    assert df_converted["Timestamp"].dtype == "datetime64[ns]"


def test_add_time_features(sample_dataframe):
    """Vérifie la génération correcte des colonnes temporelles."""
    df_transformed = add_time_features(sample_dataframe)

    assert "Mois_sin" in df_transformed.columns
    assert "Mois_cos" in df_transformed.columns
    assert "Heure_sin" in df_transformed.columns
    assert "Heure_cos" in df_transformed.columns


def test_pipeline_preprocessing(sample_dataframe):
    """Teste l'exécution complète du pipeline."""
    df_transformed = preprocessing_pipeline.fit_transform(sample_dataframe)

    assert (
        df_transformed.shape[1] >= 10
    ), "Le DataFrame transformé doit contenir au moins 10 colonnes."
