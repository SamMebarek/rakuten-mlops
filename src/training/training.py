# # src/training/training.py

# import logging
# import os
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import r2_score
# from xgboost import XGBRegressor
# from scipy.stats import randint, uniform
# import mlflow
# import mlflow.sklearn
# import requests
# import yaml
# from mlflow.tracking import MlflowClient
# from mlflow.models import infer_signature

# # Création du dossier logs si besoin
# os.makedirs("logs", exist_ok=True)

# # Configuration du logging
# logging.basicConfig(
#     filename=os.path.join("logs", "training.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger("training")


# # def is_mlflow_active(mlflow_uri: str, timeout: int = 3) -> bool:
# #     """Vérifie si le serveur MLflow est actif."""
# #     try:
# #         response = requests.get(
# #             f"{mlflow_uri}/api/2.0/mlflow/experiments/list", timeout=timeout
# #         )
# #         return response.status_code == 200
# #     except requests.exceptions.RequestException:
# #         return False

# from urllib.parse import urlparse
# import requests
# import logging

# logger = logging.getLogger("training")


# def is_mlflow_active(mlflow_uri: str, timeout: int = 3) -> bool:
#     """
#     Vérifie si le serveur MLflow est actif en utilisant les identifiants présents dans l'URI.
#     La fonction extrait le nom d'utilisateur et le token (mot de passe) si disponibles,
#     reconstruit l'URI sans ces informations pour la requête et utilise l'authentification HTTP Basic.
#     """
#     try:
#         # Extraction des informations d'authentification depuis l'URI
#         parsed_uri = urlparse(mlflow_uri)
#         auth = None
#         if parsed_uri.username and parsed_uri.password:
#             auth = (parsed_uri.username, parsed_uri.password)
#             # Recomposer l'URI sans les informations d'authentification
#             netloc = parsed_uri.hostname
#             if parsed_uri.port:
#                 netloc += f":{parsed_uri.port}"
#             sanitized_uri = f"{parsed_uri.scheme}://{netloc}{parsed_uri.path}"
#         else:
#             sanitized_uri = mlflow_uri

#         # Envoi de la requête avec authentification si disponible
#         response = requests.get(
#             f"{sanitized_uri}/api/2.0/mlflow/experiments/list",
#             timeout=timeout,
#             auth=auth,
#         )
#         logger.info(f"Vérification MLflow, code HTTP reçu : {response.status_code}")
#         return response.status_code == 200
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Erreur lors de la vérification de MLflow: {e}")
#         return False


# def main():
#     """Pipeline d'entraînement XGBoost avec versioning MLflow."""
#     try:
#         # Lecture des paramètres depuis params.yaml
#         with open("params.yaml", "r", encoding="utf-8") as f:
#             config = yaml.safe_load(f)
#         train_params = config["train"]
#         model_config = config["model_config"]

#         # Configuration MLflow
#         mlflow_uri = model_config["mlflow_tracking_uri"]
#         experiment_name = model_config["mlflow_experiment_name"]
#         model_name = model_config["mlflow_model_name"]

#         # Vérification et configuration de MLflow
#         if is_mlflow_active(mlflow_uri):
#             mlflow.set_tracking_uri(mlflow_uri)
#             mlflow.set_experiment(experiment_name)
#             logger.info("MLflow activé et configuré.")
#         else:
#             logger.warning("MLflow non accessible. Le training se fera sans tracking.")

#         # Chargement des données prétraitées
#         data_path = train_params["input"]
#         if not os.path.exists(data_path):
#             logger.error(f"Fichier prétraité introuvable : {data_path}")
#             return

#         df = pd.read_csv(data_path)
#         logger.info(f"Données prétraitées chargées : shape={df.shape}")

#         if "Prix" not in df.columns:
#             logger.error("La colonne 'Prix' est manquante dans le DataFrame.")
#             return

#         # Séparation X / y (on retire Prix, SKU, Timestamp)
#         y = df["Prix"].values
#         X = df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
#         X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X,
#             y,
#             test_size=train_params["test_size"],
#             random_state=train_params["random_state"],
#         )

#         # Définition des hyperparamètres pour RandomizedSearchCV
#         dist_params = train_params["param_dist"]
#         param_dist = {
#             "n_estimators": randint(
#                 dist_params["n_estimators_min"], dist_params["n_estimators_max"]
#             ),
#             "learning_rate": uniform(
#                 dist_params["learning_rate_min"],
#                 dist_params["learning_rate_max"] - dist_params["learning_rate_min"],
#             ),
#             "max_depth": randint(
#                 dist_params["max_depth_min"], dist_params["max_depth_max"]
#             ),
#             "subsample": uniform(
#                 dist_params["subsample_min"],
#                 dist_params["subsample_max"] - dist_params["subsample_min"],
#             ),
#             "colsample_bytree": uniform(
#                 dist_params["colsample_bytree_min"],
#                 dist_params["colsample_bytree_max"]
#                 - dist_params["colsample_bytree_min"],
#             ),
#             "gamma": uniform(
#                 dist_params["gamma_min"],
#                 dist_params["gamma_max"] - dist_params["gamma_min"],
#             ),
#         }

#         model_xgb = RandomizedSearchCV(
#             XGBRegressor(
#                 objective="reg:squarederror", random_state=train_params["random_state"]
#             ),
#             param_distributions=param_dist,
#             n_iter=10,
#             cv=3,
#             verbose=1,
#             n_jobs=-1,
#             random_state=train_params["random_state"],
#         )

#         # Entraînement et log MLflow
#         with mlflow.start_run(run_name="XGBoost_RandSearch") as run:
#             logger.info("Démarrage de la recherche d'hyperparamètres XGB")
#             model_xgb.fit(X_train, y_train)

#             y_pred = model_xgb.predict(X_test)
#             r2_xgb = r2_score(y_test, y_pred)

#             best_params = model_xgb.best_params_
#             logger.info(f"Meilleurs paramètres : {best_params}")
#             logger.info(f"Métrique R² = {r2_xgb:.4f}")

#             # Log des métriques et hyperparamètres dans MLflow
#             mlflow.log_metric("r2_score", r2_xgb)
#             for param_name, param_value in best_params.items():
#                 mlflow.log_param(param_name, param_value)

#             # # Log du modèle dans MLflow
#             # input_example = X_test.iloc[
#             #     :1
#             # ]  # Extrait un exemple pour faciliter l'inférence
#             # signature = infer_signature(
#             #     X_train, model_xgb.best_estimator_.predict(X_train)
#             # )

#             # model_path = "xgb_model"
#             # mlflow.sklearn.log_model(
#             #     sk_model=model_xgb.best_estimator_,
#             #     artifact_path=model_path,
#             #     registered_model_name=model_name,
#             #     signature=signature,
#             #     input_example=input_example,
#             # )

#             # logger.info(f"Modèle enregistré sous le nom : {model_name}")

#             # # Ajout au Model Registry
#             # client = MlflowClient()
#             # model_uri = f"runs:/{run.info.run_id}/{model_path}"

#             # try:
#             #     # Vérifier si le modèle existe déjà avant de l'enregistrer
#             #     registered_models = [m.name for m in client.search_registered_models()]
#             #     if model_name not in registered_models:
#             #         client.create_registered_model(model_name)

#             #     # Créer une nouvelle version du modèle
#             #     client.create_model_version(
#             #         name=model_name,
#             #         source=model_uri,
#             #         run_id=run.info.run_id,
#             #     )
#             # except mlflow.exceptions.MlflowException as e:
#             #     logger.error(
#             #         f"Erreur lors de l'ajout du modèle au Model Registry : {e}"
#             #     )

#             # Log du modèle dans MLflow avec gestion des erreurs
#             input_example = X_test.iloc[:1]  # Exemple pour faciliter l'inférence
#             signature = infer_signature(
#                 X_train, model_xgb.best_estimator_.predict(X_train)
#             )
#             model_path = "xgb_model"

#             try:
#                 mlflow.sklearn.log_model(
#                     sk_model=model_xgb.best_estimator_,
#                     artifact_path=model_path,
#                     registered_model_name=model_name,
#                     signature=signature,
#                     input_example=input_example,
#                 )
#                 logger.info(f"Modèle enregistré sous le nom : {model_name}")
#             except Exception as e:
#                 logger.error(
#                     f"Erreur lors de l'enregistrement du modèle avec mlflow.sklearn.log_model : {e}"
#                 )
#                 raise

#             # Ajout du modèle au Model Registry avec vérification et gestion des exceptions
#             client = MlflowClient()
#             model_uri = f"runs:/{run.info.run_id}/{model_path}"

#             try:
#                 # Vérifier si le modèle existe déjà dans le registry
#                 registered_models = [m.name for m in client.search_registered_models()]
#                 if model_name not in registered_models:
#                     client.create_registered_model(model_name)
#                     logger.info(f"Modèle '{model_name}' créé dans le registry.")

#                 # Créer une nouvelle version du modèle
#                 new_version = client.create_model_version(
#                     name=model_name,
#                     source=model_uri,
#                     run_id=run.info.run_id,
#                 )
#                 logger.info(
#                     f"Nouvelle version du modèle créée : Version {new_version.version}"
#                 )
#             except mlflow.exceptions.MlflowException as e:
#                 logger.error(
#                     f"Erreur lors de l'ajout du modèle au Model Registry : {e}"
#                 )

#             print("\nRésumé de l'entraînement :")
#             print(f"R² du modèle : {r2_xgb:.4f}")
#             print(f"Meilleurs hyperparamètres : {best_params}")

#     except Exception as e:
#         logger.error(f"Erreur dans le pipeline d'entraînement : {e}")
#         raise


# if __name__ == "__main__":
#     main()
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
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from urllib.parse import urlparse

# Création du dossier logs si besoin
os.makedirs("logs", exist_ok=True)

# Configuration du logging
logging.basicConfig(
    filename=os.path.join("logs", "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("training")


def is_mlflow_active(mlflow_uri: str, timeout: int = 3) -> bool:
    """
    Vérifie si le serveur MLflow est actif en utilisant les identifiants présents dans l'URI.
    La fonction extrait le nom d'utilisateur et le token (mot de passe) si disponibles,
    reconstruit l'URI sans ces informations pour la requête et utilise l'authentification HTTP Basic.
    """
    try:
        # Extraction des informations d'authentification depuis l'URI
        parsed_uri = urlparse(mlflow_uri)
        auth = None
        if parsed_uri.username and parsed_uri.password:
            auth = (parsed_uri.username, parsed_uri.password)
            # Recomposer l'URI sans les informations d'authentification
            netloc = parsed_uri.hostname
            if parsed_uri.port:
                netloc += f":{parsed_uri.port}"
            sanitized_uri = f"{parsed_uri.scheme}://{netloc}{parsed_uri.path}"
        else:
            sanitized_uri = mlflow_uri

        # Envoi de la requête avec authentification si disponible
        response = requests.get(
            f"{sanitized_uri}/api/2.0/mlflow/experiments/list",
            timeout=timeout,
            auth=auth,
        )
        logger.info(f"Vérification MLflow, code HTTP reçu : {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la vérification de MLflow: {e}")
        return False


def main():
    """Pipeline d'entraînement XGBoost avec versioning MLflow."""
    try:
        # Lecture des paramètres depuis params.yaml
        with open("params.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        train_params = config["train"]
        model_config = config["model_config"]

        # Configuration MLflow
        mlflow_uri = model_config["mlflow_tracking_uri"]
        experiment_name = model_config["mlflow_experiment_name"]
        model_name = model_config["mlflow_model_name"]

        # Vérification et configuration de MLflow
        if is_mlflow_active(mlflow_uri):
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow activé et configuré.")
        else:
            logger.warning("MLflow non accessible. Le training se fera sans tracking.")

        # Chargement des données prétraitées
        data_path = train_params["input"]
        if not os.path.exists(data_path):
            logger.error(f"Fichier prétraité introuvable : {data_path}")
            return

        df = pd.read_csv(data_path)
        logger.info(f"Données prétraitées chargées : shape={df.shape}")

        if "Prix" not in df.columns:
            logger.error("La colonne 'Prix' est manquante dans le DataFrame.")
            return

        # Séparation X / y (on retire Prix, SKU, Timestamp)
        y = df["Prix"].values
        X = df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=train_params["test_size"],
            random_state=train_params["random_state"],
        )

        # Définition des hyperparamètres pour RandomizedSearchCV
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
                dist_params["colsample_bytree_max"]
                - dist_params["colsample_bytree_min"],
            ),
            "gamma": uniform(
                dist_params["gamma_min"],
                dist_params["gamma_max"] - dist_params["gamma_min"],
            ),
        }

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

        # Entraînement et log MLflow
        with mlflow.start_run(run_name="XGBoost_RandSearch") as run:
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

            # Log du modèle dans MLflow avec gestion des erreurs
            input_example = X_test.iloc[:1]  # Exemple pour faciliter l'inférence
            signature = infer_signature(
                X_train, model_xgb.best_estimator_.predict(X_train)
            )
            model_path = "xgb_model"

            try:
                mlflow.sklearn.log_model(
                    sk_model=model_xgb.best_estimator_,
                    artifact_path=model_path,
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=input_example,
                )
                logger.info(f"Modèle enregistré sous le nom : {model_name}")
            except Exception as e:
                logger.error(
                    f"Erreur lors de l'enregistrement du modèle avec mlflow.sklearn.log_model : {e}"
                )
                raise

            # Ajout du modèle au Model Registry avec vérification et gestion des exceptions
            client = MlflowClient()
            model_uri = f"runs:/{run.info.run_id}/{model_path}"

            try:
                # Vérifier si le modèle existe déjà dans le registry
                registered_models = [m.name for m in client.search_registered_models()]
                if model_name not in registered_models:
                    client.create_registered_model(model_name)
                    logger.info(f"Modèle '{model_name}' créé dans le registry.")

                # Créer une nouvelle version du modèle
                new_version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run.info.run_id,
                )
                logger.info(
                    f"Nouvelle version du modèle créée : Version {new_version.version}"
                )
            except mlflow.exceptions.MlflowException as e:
                logger.error(
                    f"Erreur lors de l'ajout du modèle au Model Registry : {e}"
                )

            # Enregistrement manuel du modèle via mlflow.register_model (optionnel)
            try:
                registered_model = mlflow.register_model(
                    f"runs:/{run.info.run_id}/{model_path}", model_name
                )
                logger.info(
                    f"Modèle manuellement enregistré dans le Model Registry : {registered_model.name} (version {registered_model.version})"
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'enregistrement manuel du modèle : {e}")
                raise

            print("\nRésumé de l'entraînement :")
            print(f"R² du modèle : {r2_xgb:.4f}")
            print(f"Meilleurs hyperparamètres : {best_params}")

    except Exception as e:
        logger.error(f"Erreur dans le pipeline d'entraînement : {e}")
        raise


if __name__ == "__main__":
    main()
