# src/app/app.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import os
import logging
import yaml
import s3fs
import mlflow.pyfunc
from datetime import datetime
from pydantic import BaseModel, Field

# Chargement des paramètres globaux
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Configuration
DVC_S3_ENDPOINT = params["data"]["dvc_endpoint"]
DVC_FILE_PATH = params["data"]["processed_file"]
MLFLOW_TRACKING_URI = params["model_config"]["mlflow_tracking_uri"]
MLFLOW_EXPERIMENT = params["model_config"]["mlflow_experiment_name"]
MLFLOW_MODEL_NAME = params["model_config"]["mlflow_model_name"]

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Initialisation de FastAPI
app = FastAPI()

# Connexion à S3 via s3fs
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": DVC_S3_ENDPOINT})


def load_data():
    """Charge les données prétraitées depuis DVC S3"""
    try:
        with fs.open(DVC_FILE_PATH, "rb") as f:
            df = pd.read_csv(f)
        logger.info(f"Données chargées avec succès depuis {DVC_FILE_PATH}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données depuis DVC : {e}")
        return None


def load_latest_model():
    """Charge le modèle depuis MLflow/DagsHub"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Modèle chargé avec succès depuis MLflow : {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle depuis MLflow : {e}")
        return None


# Chargement initial des données et du modèle
data = load_data()
model = load_latest_model()


# Définition des modèles Pydantic
class PredictionRequest(BaseModel):
    sku: str = Field(..., title="SKU", description="Identifiant du produit")


class PredictionResponse(BaseModel):
    sku: str
    timestamp: str
    predicted_price: float


@app.get("/health")
def health():
    """Vérifie l'état de l'API"""
    return {
        "status": "OK",
        "model_status": "chargé" if model else "non chargé",
        "data_status": "chargée" if data is not None else "non chargée",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédiction du prix pour un SKU donné"""
    global model, data

    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    if data is None:
        raise HTTPException(status_code=500, detail="Données non chargées.")

    # Vérification de la présence du SKU dans les données
    sku_data = data[data["SKU"] == request.sku]
    if sku_data.empty:
        logger.warning(f"SKU {request.sku} non trouvé dans les données.")
        raise HTTPException(status_code=404, detail="SKU non trouvé.")

    # Trier par timestamp et prendre les 3 dernières occurrences
    sku_data = sku_data.sort_values(by="Timestamp", ascending=False).head(3)
    if sku_data.shape[0] < 3:
        raise HTTPException(
            status_code=400, detail="Données insuffisantes pour la prédiction."
        )

    # Calcul de la moyenne des features numériques
    numeric_features = [
        "PrixInitial",
        "AgeProduitEnJours",
        "QuantiteVendue",
        "UtiliteProduit",
        "ElasticitePrix",
        "Remise",
        "Qualite",
    ]
    sku_data_mean = sku_data[numeric_features].mean()

    # Ajout des features temporelles actuelles
    timestamp_now = datetime.now()
    new_values = pd.DataFrame(
        {
            "Mois_sin": [np.sin(2 * np.pi * timestamp_now.month / 12)],
            "Mois_cos": [np.cos(2 * np.pi * timestamp_now.month / 12)],
            "Heure_sin": [np.sin(2 * np.pi * timestamp_now.hour / 24)],
            "Heure_cos": [np.cos(2 * np.pi * timestamp_now.hour / 24)],
        }
    )

    # Fusionner les features
    feature_values = pd.concat([sku_data_mean.to_frame().T, new_values], axis=1)

    # Vérification stricte de l'ordre des colonnes
    expected_features = numeric_features + [
        "Mois_sin",
        "Mois_cos",
        "Heure_sin",
        "Heure_cos",
    ]
    if list(feature_values.columns) != expected_features:
        logger.error(
            f"Mismatch des features. Attendu: {expected_features}, Obtenu: {list(feature_values.columns)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Les features ne correspondent pas à celles attendues.",
        )

    # Faire la prédiction
    feature_values = feature_values.values.reshape(1, -1)
    predicted_price = float(model.predict(feature_values)[0])

    return PredictionResponse(
        sku=request.sku,
        timestamp=timestamp_now.strftime("%Y-%m-%d %H:%M:%S"),
        predicted_price=round(predicted_price, 2),
    )


@app.post("/reload-model")
def reload_model():
    """Recharge le modèle depuis MLflow"""
    global model
    model = load_latest_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Aucun modèle trouvé.")
    return {"message": "Modèle rechargé avec succès"}
