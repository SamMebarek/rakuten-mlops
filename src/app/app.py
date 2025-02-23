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
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Remplacement des variables d'environnement
for section in config:
    if isinstance(config[section], dict):
        for key, value in config[section].items():
            if isinstance(value, str) and "${" in value:
                env_var = value.strip("${}").strip()
                if env_var in os.environ:
                    config[section][key] = os.getenv(env_var)

# Configuration MLflow
MLFLOW_TRACKING_URI = config["model_config"]["mlflow_tracking_uri"]
MLFLOW_EXPERIMENT = config["model_config"]["mlflow_experiment_name"]
MLFLOW_MODEL_NAME = config["model_config"]["mlflow_model_name"]

# Vérification et forçage de MLflow
if not MLFLOW_TRACKING_URI.startswith("http"):
    raise ValueError(f"🚨 ERREUR : MLFLOW_TRACKING_URI invalide")

print(f"MLflow va utiliser l'URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

# Vérification que l'URI est bien prise en compte
assert (
    mlflow.get_tracking_uri() == MLFLOW_TRACKING_URI
), "MLflow n'a pas pris l'URI en compte"

# Configuration des données
DVC_BUCKET_PATH = config["data"]["dvc_path"]
DVC_RELATIVE_PATH = config["data"][
    "processed_file"
]  # ex: "data/processed/preprocessed_data.csv"
DVC_S3_ENDPOINT = config["data"]["dvc_endpoint"]

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Initialisation de FastAPI
app = FastAPI()

# Récupération des credentials AWS depuis les variables d'environnement
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not aws_access_key_id or not aws_secret_access_key:
    logger.error(
        "Les variables d'environnement AWS_ACCESS_KEY_ID ou AWS_SECRET_ACCESS_KEY ne sont pas définies."
    )

# Connexion à S3 via s3fs avec les credentials
fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url": DVC_S3_ENDPOINT,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }
)


def load_data():
    """
    Charge les données prétraitées.
    Si le fichier existe localement (selon DVC_RELATIVE_PATH), il est chargé depuis le système de fichiers local.
    Sinon, le fichier est recherché sur S3 en combinant DVC_BUCKET_PATH et DVC_RELATIVE_PATH.
    """
    if os.path.exists(DVC_RELATIVE_PATH):
        try:
            df = pd.read_csv(DVC_RELATIVE_PATH)
            logger.info(f"Données chargées localement depuis {DVC_RELATIVE_PATH}")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement local des données : {e}")

    try:
        full_path = f"{DVC_BUCKET_PATH.rstrip('/')}/{DVC_RELATIVE_PATH.lstrip('/')}"
        with fs.open(full_path, "rb") as f:
            df = pd.read_csv(f)
        logger.info(f"Données chargées avec succès depuis {full_path}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données depuis DVC : {e}")
        return None


def load_latest_model():
    """Charge le modèle depuis MLflow/DagsHub"""
    try:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Modèle chargé avec succès depuis MLflow")
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

    sku_data = data[data["SKU"] == request.sku]
    if sku_data.empty:
        logger.warning(f"SKU {request.sku} non trouvé dans les données.")
        raise HTTPException(status_code=404, detail="SKU non trouvé.")

    sku_data = sku_data.sort_values(by="Timestamp", ascending=False).head(3)
    if sku_data.shape[0] < 3:
        raise HTTPException(
            status_code=400, detail="Données insuffisantes pour la prédiction."
        )

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

    timestamp_now = datetime.now()
    new_values = pd.DataFrame(
        {
            "Mois_sin": [np.sin(2 * np.pi * timestamp_now.month / 12)],
            "Mois_cos": [np.cos(2 * np.pi * timestamp_now.month / 12)],
            "Heure_sin": [np.sin(2 * np.pi * timestamp_now.hour / 24)],
            "Heure_cos": [np.cos(2 * np.pi * timestamp_now.hour / 24)],
        }
    )

    feature_values = pd.concat([sku_data_mean.to_frame().T, new_values], axis=1)
    for col in ["AgeProduitEnJours", "QuantiteVendue"]:
        if col in feature_values.columns:
            feature_values[col] = feature_values[col].round(0).astype("int64")

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
            detail="Les features ne correspondent pas à celles attendues.",
        )

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
