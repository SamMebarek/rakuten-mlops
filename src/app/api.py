# # src/app/api.PythonFinalizationError

# from fastapi import FastAPI, HTTPException
# import pandas as pd
# import numpy as np
# import mlflow.pyfunc
# import os
# import logging
# import dvc.api
# from datetime import datetime
# from pydantic import BaseModel, Field

# # Configuration du logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("api")

# # Initialisation de l'API
# app = FastAPI()

# # Chemins des fichiers DVC
# DVC_DATA_PATH = "data/processed/preprocessed_data.csv"
# DVC_REPO = "https://dagshub.com/<user>/<repo>.git"  # Remplacer par votre repo

# # Configuration MLflow
# MLFLOW_TRACKING_URI = "https://dagshub.com/SamMebarek/rakuten-mlops.mlflow"
# MLFLOW_MODEL_NAME = "DynamicPricingModel"

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# def load_data():
#     """Charge les données preprocessées depuis DVC."""
#     try:
#         with dvc.api.open(DVC_DATA_PATH, repo=DVC_REPO, mode="r") as f:
#             df = pd.read_csv(f)
#         logger.info("Données prétraitées chargées avec succès depuis DVC.")
#         return df
#     except Exception as e:
#         logger.error(f"Erreur lors du chargement des données depuis DVC : {e}")
#         return None


# def load_latest_model():
#     """Charge le modèle depuis MLflow."""
#     try:
#         model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")
#         logger.info("Modèle chargé avec succès depuis MLflow.")
#         return model
#     except Exception as e:
#         logger.error(f"Erreur lors du chargement du modèle depuis MLflow : {e}")
#         return None


# data = load_data()
# model = load_latest_model()


# # Définition du modèle de requête
# class PredictionRequest(BaseModel):
#     sku: str = Field(..., title="SKU", description="Identifiant du produit")


# # Définition du modèle de réponse
# class PredictionResponse(BaseModel):
#     sku: str
#     timestamp: str
#     predicted_price: float


# @app.get("/health")
# def health():
#     """Endpoint de vérification de l'état de l'API."""
#     return {"status": "OK"}


# @app.get("/status")
# def status():
#     """Endpoint pour obtenir le statut du modèle et des données."""
#     model_status = "chargé" if model else "non chargé"
#     data_status = "chargées" if data is not None else "non chargées"
#     return {"model_status": model_status, "data_status": data_status}


# @app.post("/predict", response_model=PredictionResponse)
# def predict(request: PredictionRequest):
#     """Effectue une prédiction de prix sur un SKU donné."""
#     global model, data
#     if model is None:
#         raise HTTPException(status_code=500, detail="Modèle non chargé.")
#     if data is None:
#         raise HTTPException(status_code=500, detail="Données non chargées.")

#     sku_data = data[data["SKU"] == request.sku]
#     if sku_data.empty:
#         logger.warning(f"SKU {request.sku} non trouvé dans les données.")
#         raise HTTPException(status_code=404, detail="SKU non trouvé.")

#     sku_data = sku_data.sort_values(by="Timestamp", ascending=False).head(3)
#     if sku_data.shape[0] < 3:
#         raise HTTPException(
#             status_code=400, detail="Données insuffisantes pour la prédiction."
#         )

#     # **Vérification du bon ordre des colonnes**
#     numeric_features = [
#         "PrixInitial",
#         "AgeProduitEnJours",
#         "QuantiteVendue",
#         "UtiliteProduit",
#         "ElasticitePrix",
#         "Remise",
#         "Qualite",
#     ]

#     # **Fusionner la moyenne des features avec les nouvelles features temporelles**

#     sku_data_mean = sku_data[numeric_features].mean()

#     timestamp_now = datetime.now()
#     new_values = pd.DataFrame(
#         {
#             "Mois_sin": [np.sin(2 * np.pi * timestamp_now.month / 12)],
#             "Mois_cos": [np.cos(2 * np.pi * timestamp_now.month / 12)],
#             "Heure_sin": [np.sin(2 * np.pi * timestamp_now.hour / 24)],
#             "Heure_cos": [np.cos(2 * np.pi * timestamp_now.hour / 24)],
#         }
#     )

#     feature_values = pd.concat([sku_data_mean.to_frame().T, new_values], axis=1)
#     expected_features = numeric_features + [
#         "Mois_sin",
#         "Mois_cos",
#         "Heure_sin",
#         "Heure_cos",
#     ]

#     if list(feature_values.columns) != expected_features:
#         logger.error(
#             f"Mismatch des features. Attendu: {expected_features}, Obtenu: {list(feature_values.columns)}"
#         )
#         raise HTTPException(
#             status_code=500,
#             detail=f"Les features ne correspondent pas à celles attendues.",
#         )

#     feature_values = feature_values.values.reshape(1, -1)
#     predicted_price = float(model.predict(feature_values)[0])

#     return PredictionResponse(
#         sku=request.sku,
#         timestamp=timestamp_now.strftime("%Y-%m-%d %H:%M:%S"),
#         predicted_price=round(predicted_price, 2),
#     )


# @app.post("/reload-model")
# def reload_model():
#     """Recharge le modèle depuis MLflow."""
#     global model
#     model = load_latest_model()
#     if model is None:
#         raise HTTPException(status_code=500, detail="Aucun modèle trouvé.")
#     return {"message": "Modèle rechargé avec succès"}
