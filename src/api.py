from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pydantic import BaseModel, Field
from datetime import datetime

# Configuration
DATA_PATH = "Data/preprocessed_data.csv"
MODEL_DIR = "models"
LATEST_MODEL_FILE = os.path.join(MODEL_DIR, "latest_model.txt")

# Initialisation de l'API
app = FastAPI()


# Chargement des données prétraitées
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


data = load_data()


# Chargement dynamique du modèle avec logs de diagnostic
def load_latest_model():
    if not os.path.exists(LATEST_MODEL_FILE):
        print("[ERREUR] latest_model.txt introuvable.")
        return None

    with open(LATEST_MODEL_FILE, "r") as f:
        model_folder = f.read().strip()

    model_path = os.path.join(MODEL_DIR, model_folder, "model.pkl")

    if not os.path.exists(model_path):
        print(f"[ERREUR] Modèle introuvable à l'emplacement : {model_path}")
        return None

    print(f"[INFO] Chargement du modèle depuis : {model_path}")
    return joblib.load(model_path)


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
