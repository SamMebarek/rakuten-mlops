# tests/conftest.py

import pytest
import yaml
from pathlib import Path
from src.generation.generation import ParametresSynthese, GenerateurDonnees
from fastapi.testclient import TestClient

# tests/conftest.py
import sys

# Ajoute le répertoire racine du projet au sys.path.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """
    Crée un fichier de configuration minimal et valide pour la génération.
    """
    config_content = """
seed: 42
random_seed_range: 10001
n_skus: 5
n_periodes: 3
categories:
  Electronique:
    prix_moyen: 100
    prix_ecart_type: 20
    plancher_pct: 0.5
    delta: 0.01
    gamma: 0.001
    demande_lambda: 2
    remise_prob: 0.05
beta_prix_mean: 0.05
beta_prix_std: 0.005
beta_qualite_mean: 0.05
beta_qualite_std: 0.005
beta_promo_mean: 0.03
beta_promo_std: 0.002
erreur_std: 1
remise_valeur: 0.1
prix_minimum: 0.01
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def params(minimal_config: Path):
    """
    Retourne une instance de ParametresSynthese basée sur le fichier de configuration minimal.
    """
    return ParametresSynthese(config_path=str(minimal_config))


@pytest.fixture
def generator(params):
    """
    Retourne une instance de GenerateurDonnees basée sur les paramètres fournis.
    """
    return GenerateurDonnees(params)


import pytest
import pandas as pd
from pathlib import Path

from src.preprocessing.preprocessing import (
    drop_unused_columns,
    convert_types,
    add_time_features,
    preprocessing_pipeline,
    run_preprocessing,
)


@pytest.fixture
def sample_dataframe():
    """
    Retourne un DataFrame de test comportant les colonnes attendues pour la conversion,
    incluant quelques colonnes inutiles devant être supprimées.
    """
    df = pd.DataFrame(
        {
            "SKU": ["A", "B"],
            "Timestamp": ["2024-01-01 12:00:00", "2024-06-01 18:00:00"],
            "PrixInitial": [10.5, 15.3],
            "Prix": [12, 18],
            "AgeProduitEnJours": [5.0, 10.0],  # À convertir en int64
            "QuantiteVendue": [2.0, 3.0],  # À convertir en int64
            "UtiliteProduit": [0.5, 0.6],
            "ElasticitePrix": [0.7, 0.8],
            "Remise": [0.1, 0.2],
            "Qualite": [0.9, 1.0],
            # Colonnes à supprimer
            "DateLancement": ["2024-01-01", "2024-01-01"],
            "PrixPlancher": [5, 7],
            "PlancherPourcentage": [0.4, 0.4],
            "ErreurAleatoire": [0.0, 0.0],
            "Annee": [2024, 2024],
            "Mois": [1, 6],
            "Jour": [1, 1],
            "Heure": [12, 18],
            "Promotion": [0, 1],
            "Categorie": ["Cat1", "Cat1"],
        }
    )
    return df


@pytest.fixture
def temp_preprocessing_config(tmp_path: Path):
    """
    Crée un fichier de configuration temporaire pour la section preprocessing.
    On utilise des guillemets simples pour les chemins afin d'éviter les problèmes d'échappement.
    """
    # Créer un CSV d'ingestion temporaire
    ingestion_csv = tmp_path / "ingested_data.csv"
    df_ingestion = pd.DataFrame(
        {
            "SKU": ["A", "B"],
            "Timestamp": ["2024-01-01 12:00:00", "2024-06-01 18:00:00"],
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
    ingestion_csv.write_text(df_ingestion.to_csv(index=False))
    output_csv = tmp_path / "preprocessed_data.csv"
    config_content = f"""
preprocessing:
  input: '{ingestion_csv}'
  output: '{output_csv}'
"""
    config_file = tmp_path / "test_preprocessing_config.yaml"
    config_file.write_text(config_content)
    return config_file, output_csv


@pytest.fixture
def training_config(tmp_path: Path) -> Path:
    """
    Crée un fichier de configuration temporaire pour la section training.
    La configuration contient les paramètres de training et de modèle.
    Le placeholder {input_csv} sera remplacé dans le test.
    """
    config_content = """
train:
  input: "{input_csv}"
  test_size: 0.5
  random_state: 42
  param_dist:
    n_estimators_min: 50
    n_estimators_max: 100
    learning_rate_min: 0.01
    learning_rate_max: 0.1
    max_depth_min: 3
    max_depth_max: 7
    subsample_min: 0.6
    subsample_max: 1.0
    colsample_bytree_min: 0.6
    colsample_bytree_max: 1.0
    gamma_min: 0.0
    gamma_max: 0.3

model_config:
  mlflow_tracking_uri: "https://dagshub.com/xxx/rakuten-mlops.mlflow"
  mlflow_experiment_name: "DynamicPricing"
  mlflow_model_name: "BestModel"
"""
    config_file = tmp_path / "training_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def training_csv(tmp_path: Path) -> Path:
    """
    Crée un fichier CSV temporaire avec des données d'entraînement minimales.
    Pour que la cross-validation avec cv=3 fonctionne, on crée 6 échantillons.
    """
    data = {
        "SKU": ["A", "B", "C", "D", "E", "F"],
        "Prix": [10.5, 15.3, 12.0, 14.5, 11.0, 16.0],
        "PrixInitial": [9.5, 14.0, 11.0, 13.5, 10.0, 15.0],
        "Timestamp": [
            "2024-01-01 12:00:00",
            "2024-01-02 12:00:00",
            "2024-01-03 12:00:00",
            "2024-01-04 12:00:00",
            "2024-01-05 12:00:00",
            "2024-01-06 12:00:00",
        ],
        "AgeProduitEnJours": [5, 10, 7, 8, 6, 9],
        "QuantiteVendue": [3, 6, 4, 5, 3, 7],
        "UtiliteProduit": [0.8, 0.9, 0.85, 0.87, 0.82, 0.91],
        "ElasticitePrix": [0.5, 0.6, 0.55, 0.58, 0.52, 0.61],
        "Remise": [0.1, 0.2, 0.15, 0.18, 0.12, 0.22],
        "Qualite": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Mois_sin": [0.5, 0.6, 0.55, 0.57, 0.52, 0.61],
        "Mois_cos": [0.8, 0.7, 0.75, 0.73, 0.78, 0.69],
        "Heure_sin": [0.3, 0.4, 0.35, 0.37, 0.32, 0.41],
        "Heure_cos": [0.9, 0.8, 0.85, 0.83, 0.88, 0.79],
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "training_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


# Importer le module de l'application pour pouvoir patcher ses variables globales
import src.app.app as app_mod
from src.app.app import app


# Modèle factice pour la prédiction
class DummyModel:
    def predict(self, df):
        import numpy as np

        # Retourne toujours 100.0
        return np.array([100.0])


@pytest.fixture
def client():
    # Créer un DataFrame factice pour les données avec au moins 3 enregistrements pour le SKU "SKU1"
    df_dummy = pd.DataFrame(
        {
            "SKU": ["SKU1", "SKU1", "SKU1"],
            "Timestamp": [
                "2024-01-01 12:00:00",
                "2024-01-02 12:00:00",
                "2024-01-03 12:00:00",
            ],
            "PrixInitial": [100, 100, 100],
            "AgeProduitEnJours": [5, 6, 7],
            "QuantiteVendue": [10, 10, 10],
            "UtiliteProduit": [0.5, 0.5, 0.5],
            "ElasticitePrix": [0.2, 0.2, 0.2],
            "Remise": [0.1, 0.1, 0.1],
            "Qualite": [0.9, 0.9, 0.9],
            "Mois_sin": [0.5, 0.5, 0.5],
            "Mois_cos": [0.8, 0.8, 0.8],
            "Heure_sin": [0.3, 0.3, 0.3],
            "Heure_cos": [0.9, 0.9, 0.9],
        }
    )
    # Patch les variables globales du module d'application
    app_mod.model = DummyModel()
    app_mod.data = df_dummy
    return TestClient(app)
