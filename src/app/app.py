# src/app/app.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from pydantic import BaseModel, Field

# Configuration
DATA_DIR = "Data"
LATEST_PREPROCESSED_FILE = os.path.join(DATA_DIR, "latest_preprocessed.txt")
MODEL_DIR = "models"
LATEST_MODEL_FILE = os.path.join(MODEL_DIR, "latest_model.txt")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Initialisation de l'API
app = FastAPI()


# Chargement du dernier fichier prétraité
def get_latest_preprocessed_file():
    try:
        if not os.path.exists(LATEST_PREPROCESSED_FILE):
            raise FileNotFoundError("Fichier latest_preprocessed.txt introuvable.")
        with open(LATEST_PREPROCESSED_FILE, "r") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du fichier prétraité : {e}")
        return None


def load_data():
    try:
        latest_data_path = get_latest_preprocessed_file()
        if latest_data_path and os.path.exists(latest_data_path):
            return pd.read_csv(latest_data_path)
        else:
            logger.error("Fichier prétraité introuvable ou corrompu.")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données prétraitées : {e}")
        return None


data = load_data()


# Chargement dynamique du modèle avec logs de diagnostic
def load_latest_model():
    try:
        if not os.path.exists(LATEST_MODEL_FILE):
            raise FileNotFoundError("latest_model.txt introuvable.")

        with open(LATEST_MODEL_FILE, "r") as f:
            model_folder = f.read().strip()

        model_path = os.path.join(MODEL_DIR, model_folder, "model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle introuvable à l'emplacement : {model_path}"
            )

        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise ValueError("Le modèle chargé n'est pas valide.")

        logger.info(f"Modèle chargé avec succès depuis : {model_path}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        return None


model = load_latest_model()


# Définition du modèle de requête
class PredictionRequest(BaseModel):
    sku: str = Field(..., title="SKU", description="Identifiant du produit")


# Définition du modèle de réponse
class PredictionResponse(BaseModel):
    sku: str
    timestamp: str
    predicted_price: float


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/status")
def status():
    model_status = "chargé" if model else "non chargé"
    return {
        "model_status": model_status,
        "last_model": (
            open(LATEST_MODEL_FILE).read().strip()
            if os.path.exists(LATEST_MODEL_FILE)
            else "Inconnu"
        ),
        "data_status": "chargées" if data is not None else "non chargées",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
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

    # **Calculer la moyenne des features numériques (excluant SKU, Prix, Timestamp)**
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

    # **Récupérer dynamiquement le timestamp actuel**
    timestamp_now = datetime.now()
    new_values = pd.DataFrame(
        {
            "Mois_sin": [np.sin(2 * np.pi * timestamp_now.month / 12)],
            "Mois_cos": [np.cos(2 * np.pi * timestamp_now.month / 12)],
            "Heure_sin": [np.sin(2 * np.pi * timestamp_now.hour / 24)],
            "Heure_cos": [np.cos(2 * np.pi * timestamp_now.hour / 24)],
        }
    )

    # **Fusionner la moyenne des features avec les nouvelles features temporelles**
    feature_values = pd.concat([sku_data_mean.to_frame().T, new_values], axis=1)

    # **Vérification stricte du bon ordre des colonnes**
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
            detail=f"Les features ne correspondent pas à celles attendues. Attendu: {expected_features}, Obtenu: {list(feature_values.columns)}",
        )

    # **Transformer en tableau pour le modèle**
    feature_values = feature_values.values.reshape(1, -1)

    # **Faire la prédiction et convertir le résultat en float natif**
    predicted_price = float(model.predict(feature_values)[0])

    return PredictionResponse(
        sku=request.sku,
        timestamp=timestamp_now.strftime("%Y-%m-%d %H:%M:%S"),
        predicted_price=round(predicted_price, 2),
    )


@app.post("/reload-model")
def reload_model():
    global model
    model = load_latest_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Aucun modèle trouvé.")
    return {"message": "Modèle rechargé avec succès"}
