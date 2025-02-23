import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Remplacement propre des variables d'environnement
for section in config:
    if isinstance(config[section], dict):
        for key, value in config[section].items():
            if isinstance(value, str) and "${" in value:
                env_var = value.strip("${}").strip()
                if env_var in os.environ:
                    config[section][key] = os.getenv(env_var)

# Vérification et forçage de MLflow
mlflow_tracking_uri = config["model_config"]["mlflow_tracking_uri"]
if not mlflow_tracking_uri.startswith("http"):
    raise ValueError(f"ERREUR : MLFLOW_TRACKING_URI invalide")

print(f"MLflow va utiliser L'URI)

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_registry_uri(mlflow_tracking_uri)

# Vérification que l'URI est bien prise en compte
assert (
    mlflow.get_tracking_uri() == mlflow_tracking_uri
), "MLflow n'a pas pris l'URI en compte"

mlflow.set_experiment(config["model_config"]["mlflow_experiment_name"])

# Configuration du logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("training")


def main():
    train_params = config["train"]
    model_config = config["model_config"]

    logger.info("Début de l'entraînement du modèle.")

    # Chargement des données prétraitées
    data_path = train_params["input"]
    if not os.path.exists(data_path):
        logger.error(f"Fichier prétraité introuvable : {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Données prétraitées chargées, shape = {df.shape}")

    if "Prix" not in df.columns:
        logger.error("La colonne 'Prix' est manquante dans le DataFrame.")
        return

    # Séparation des features et de la target
    y = df["Prix"].values
    X = (
        df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_params["test_size"],
        random_state=train_params["random_state"],
    )

    # Définition des hyperparamètres
    dist_params = train_params["param_dist"]
    param_dist = {
        "n_estimators": randint(
            dist_params["n_estimators_min"], dist_params["n_estimators_max"]
        ),
        "learning_rate": uniform(
            dist_params["learning_rate_min"],
            dist_params["learning_rate_max"] - dist_params["learning_rate_min"],
        ),
        "max_depth": randint(
            dist_params["max_depth_min"], dist_params["max_depth_max"]
        ),
        "subsample": uniform(
            dist_params["subsample_min"],
            dist_params["subsample_max"] - dist_params["subsample_min"],
        ),
        "colsample_bytree": uniform(
            dist_params["colsample_bytree_min"],
            dist_params["colsample_bytree_max"] - dist_params["colsample_bytree_min"],
        ),
        "gamma": uniform(
            dist_params["gamma_min"],
            dist_params["gamma_max"] - dist_params["gamma_min"],
        ),
    }

    # Initialisation du modèle avec RandomizedSearchCV
    model_xgb = RandomizedSearchCV(
        XGBRegressor(
            objective="reg:squarederror", random_state=train_params["random_state"]
        ),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=train_params["random_state"],
    )

    # Entraînement et logging dans MLflow
    with mlflow.start_run(run_name="XGBoost_RandSearch") as run:
        logger.info("Démarrage de la recherche d'hyperparamètres XGBoost.")
        model_xgb.fit(X_train, y_train)

        y_pred = model_xgb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        best_params = model_xgb.best_params_

        logger.info(f"Meilleurs paramètres : {best_params}")
        logger.info(f"Métrique R² : {r2:.4f}")

        mlflow.log_metric("r2_score", r2)
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_train, model_xgb.best_estimator_.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model_xgb.best_estimator_,
            artifact_path="xgb_model",
            registered_model_name=model_config["mlflow_model_name"],
            signature=signature,
            input_example=input_example,
        )

        print(f"Modèle enregistré avec succès dans MLflow. R² : {r2:.4f}")


if __name__ == "__main__":
    main()
