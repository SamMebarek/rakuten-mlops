from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Configuration
DATA_PATH = "Data/preprocessed_data.csv"
MODEL_PATH = "models/xgb_model/model.pkl"

# Initialisation de l'API
app = FastAPI()

# Chargement initial du modèle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# Chargement des données prétraitées
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


data = load_data()
data["2"] = pd.to_datetime(data["2"], errors="coerce")


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict(sku: str):
    global model, data
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    if data is None:
        raise HTTPException(status_code=500, detail="Données non chargées.")

    # Vérification de la présence du SKU
    sku_data = data[data.iloc[:, 3] == sku]  # Colonne SKU est la 4e colonne (index 3)
    if sku_data.empty:
        raise HTTPException(status_code=404, detail="SKU non trouvé.")

    # Sélection des trois dernières valeurs basées sur le timestamp (colonne index 5)
    sku_data.iloc[:, 5] = pd.to_datetime(sku_data.iloc[:, 5])  # Conversion en datetime
    sku_data = sku_data.sort_values(by=sku_data.columns[5], ascending=False).head(3)

    if sku_data.shape[0] < 3:
        raise HTTPException(
            status_code=400, detail="Données insuffisantes pour la prédiction."
        )

    # Suppression des colonnes non numériques (SKU, Timestamp, Prix)
    numeric_columns = data.columns.difference(
        ["0", "2", "Prix"]
    )  # Exclure SKU (index 3), Timestamp (index 5), Prix
    feature_values = sku_data[numeric_columns].mean().values.reshape(1, -1)

    # Vérification du bon nombre de features
    if feature_values.shape[1] != model.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail=f"Feature shape mismatch, expected: {model.n_features_in_}, got {feature_values.shape[1]}",
        )

    # Prédiction
    predicted_price = model.predict(feature_values)[0]

    return {
        "sku": sku,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_price": round(predicted_price, 2),
    }


@app.post("/reload-model")
def reload_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Modèle non trouvé.")

    model = joblib.load(MODEL_PATH)
    return {"message": "Modèle rechargé avec succès"}


# Exécution de l'API avec uvicorn si ce script est lancé directement
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
