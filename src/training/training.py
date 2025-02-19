# src/training/training.py

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
import requests
import yaml

# Création du dossier logs si besoin
os.makedirs("logs", exist_ok=True)

# Configuration du logging pour le module training
logging.basicConfig(
    filename=os.path.join("logs", "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("training")


def is_mlflow_active(mlflow_uri: str, timeout: int = 3) -> bool:
    """
    Vérifie si le serveur MLflow spécifié est actif.
    """
    try:
        response = requests.get(
            f"{mlflow_uri}/api/2.0/mlflow/experiments/list", timeout=timeout
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    """
    Pipeline d'entraînement XGBoost avec versioning MLflow.

    """
    try:
        # Lecture des paramètres depuis params.yaml
        with open("params.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        train_params = config["train"]

        # Chemins et configuration MLflow
        data_path = train_params.get("input", "data/preprocessed_data.csv")
        mlflow_uri = train_params.get("mlflow_tracking_uri", "http://127.0.0.1:8080")
        experiment_name = train_params.get("experiment_name", "DynamicPricing")

        # b) Paramètres pour le split
        test_size = train_params.get("test_size", 0.2)
        random_state = train_params.get("random_state", 17)

        logger.info(f"Chemin du CSV prétraité : {data_path}")
        logger.info(f"MLflow URI : {mlflow_uri}, Exp Name : {experiment_name}")

        if not os.path.exists(data_path):
            logger.error(f"Fichier prétraité introuvable : {data_path}")
            return

        # 2. Chargement des données prétraitées
        df = pd.read_csv(data_path)
        logger.info(f"Données prétraitées chargées : shape={df.shape}")

        if "Prix" not in df.columns:
            logger.error("La colonne 'Prix' est manquante dans le DataFrame.")
            return

        # 3. Séparation X / y (on retire Prix, SKU, Timestamp)
        y = df["Prix"].values
        X = df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 4. Configuration de MLflow
        if is_mlflow_active(mlflow_uri):
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow activé et configuré.")
        else:
            logger.warning("MLflow non accessible. Le training se fera sans tracking.")

        # 5. Construction du param_dist depuis params.yaml
        dist_params = train_params.get("param_dist", {})
        # Valeurs par défaut si certaines clés manquent
        n_estimators_min = dist_params.get("n_estimators_min", 50)
        n_estimators_max = dist_params.get("n_estimators_max", 200)
        learning_rate_min = dist_params.get("learning_rate_min", 0.01)
        learning_rate_max = dist_params.get("learning_rate_max", 0.2)
        max_depth_min = dist_params.get("max_depth_min", 3)
        max_depth_max = dist_params.get("max_depth_max", 7)
        subsample_min = dist_params.get("subsample_min", 0.6)
        subsample_max = dist_params.get("subsample_max", 1.0)
        colsample_min = dist_params.get("colsample_bytree_min", 0.6)
        colsample_max = dist_params.get("colsample_bytree_max", 1.0)
        gamma_min = dist_params.get("gamma_min", 0.0)
        gamma_max = dist_params.get("gamma_max", 0.3)

        # Création du dictionnaire pour RandomizedSearchCV
        param_dist = {
            "n_estimators": randint(n_estimators_min, n_estimators_max),
            "learning_rate": uniform(
                learning_rate_min, learning_rate_max - learning_rate_min
            ),
            "max_depth": randint(max_depth_min, max_depth_max),
            "subsample": uniform(subsample_min, subsample_max - subsample_min),
            "colsample_bytree": uniform(colsample_min, colsample_max - colsample_min),
            "gamma": uniform(gamma_min, gamma_max - gamma_min),
        }

        model_xgb = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", random_state=random_state),
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=random_state,
        )

        # 6. Entraînement et log MLflow
        with (
            mlflow.start_run(run_name="XGBoost_RandSearch")
            if is_mlflow_active(mlflow_uri)
            else open(os.devnull, "w")
        ):
            logger.info("Démarrage de la recherche d'hyperparamètres XGB")
            model_xgb.fit(X_train, y_train)

            y_pred = model_xgb.predict(X_test)
            r2_xgb = r2_score(y_test, y_pred)

            best_params = model_xgb.best_params_
            logger.info(f"Meilleurs paramètres : {best_params}")
            logger.info(f"Métrique R² = {r2_xgb:.4f}")

            # Log des métriques et hyperparamètres dans MLflow
            mlflow.log_metric("r2_score", r2_xgb)
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)

            # Log du modèle dans MLflow (artifact_path = "xgb_model")
            mlflow.sklearn.log_model(
                model_xgb.best_estimator_,
                artifact_path="xgb_model",
                registered_model_name="MyXGBModel",  # Optionnel, pour le Model Registry
            )

            # MLflow log
            logger.info("Modèle loggé dans MLflow.")

            print("\nRésumé de l'entraînement :")
            print(f"R² du modèle : {r2_xgb:.4f}")
            print(f"Meilleurs hyperparamètres : {best_params}")

    except Exception as e:
        logger.error(f"Erreur dans le pipeline d'entraînement : {e}")
        raise


if __name__ == "__main__":
    main()
