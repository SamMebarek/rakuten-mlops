# src/training.py

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
import requests

# Configuration du logging
logging.basicConfig(
    filename="Logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("training")

# Définition des chemins
LATEST_PREPROCESSED_FILE = "Data/latest_preprocessed.txt"
MODEL_DIR = "models"
MLFLOW_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model")
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"


# Vérifier si le serveur MLflow est actif
def is_mlflow_active():
    try:
        response = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list", timeout=3
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_latest_preprocessed_file():
    """Récupère le dernier fichier prétraité stocké dans latest_preprocessed.txt"""
    if not os.path.exists(LATEST_PREPROCESSED_FILE):
        logger.error(f"Fichier de référence {LATEST_PREPROCESSED_FILE} introuvable.")
        raise FileNotFoundError(
            f"Fichier de référence {LATEST_PREPROCESSED_FILE} introuvable."
        )
    with open(LATEST_PREPROCESSED_FILE, "r") as f:
        return f.read().strip()


def main():
    """
    Pipeline d'entraînement avec XGBoost et tracking via MLflow
    """
    try:
        DATA_PATH = get_latest_preprocessed_file()
        logger.info(f"Données utilisées : {DATA_PATH}")

        if not os.path.exists(DATA_PATH):
            logger.error("Fichier prétraité introuvable: %s", DATA_PATH)
            return

        df = pd.read_csv(DATA_PATH)
        logger.info("Données prétraitées chargées : shape=%s", df.shape)

        if "Prix" not in df.columns:
            logger.error("La colonne 'Prix' est manquante dans le DataFrame prétraité.")
            return

        # Séparation X / y
        y = df["Prix"].values
        X = df.drop(columns=["Prix", "SKU", "Timestamp"])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=17
        )

        # Vérification de MLflow
        if is_mlflow_active():
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("DynamicPricing")
        else:
            logger.warning(
                "MLflow n'est pas accessible. L'entraînement sera exécuté sans tracking."
            )

        # Optimisation des hyperparamètres
        param_dist = {
            "n_estimators": randint(50, 200),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(3, 7),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0, 0.3),
        }

        model_xgb = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", random_state=17),
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=17,
        )

        with (
            mlflow.start_run(run_name="XGBoost_RandSearch")
            if is_mlflow_active()
            else open(os.devnull, "w")
        ):
            logger.info("Démarrage de la recherche d'hyperparamètres XGB")
            try:
                model_xgb.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement : {e}")
                return

            y_pred = model_xgb.predict(X_test)
            r2_xgb = r2_score(y_test, y_pred)

            logger.info("Meilleurs paramètres: %s", model_xgb.best_params_)
            logger.info("Métriques XGBoost => R²=%.4f", r2_xgb)

            # Sauvegarde du modèle
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"xgb_model_{timestamp}"
            model_path = os.path.join(MODEL_DIR, model_filename)

            input_example = X_test.iloc[0:1]
            os.makedirs(MODEL_DIR, exist_ok=True)

            mlflow.sklearn.log_model(
                model_xgb.best_estimator_,
                artifact_path="xgb_model",
                input_example=input_example,
            )
            mlflow.sklearn.save_model(model_xgb.best_estimator_, path=model_path)

            latest_model_path = os.path.join(MODEL_DIR, "latest_model.txt")
            with open(latest_model_path, "w") as f:
                f.write(model_filename)

            logger.info(f"Modèle sauvegardé localement dans {model_path}")

        print("\nRésumé de l'entraînement :")
        print(f"Modèle sauvegardé : {model_filename}")
        print(f"R² du modèle : {r2_xgb:.4f}")
        print(f"Meilleurs hyperparamètres : {model_xgb.best_params_}")

    except Exception as e:
        logger.error(f"Erreur dans le pipeline d'entraînement : {e}")
        raise


if __name__ == "__main__":
    main()
