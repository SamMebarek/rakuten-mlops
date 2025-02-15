# training.py
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

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("training")

# Définition des chemins
DATA_PATH = "Data/preprocessed_data.csv"
MODEL_DIR = "models"
MLFLOW_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model")

# Configuration du serveur MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def main():
    """
    Pipeline d'entraînement :
      - Charge le CSV prétraité (contenant la colonne 'Prix'),
      - Sépare X (features) et y (cible),
      - Convertit toutes les features en numérique (gestion des NaN),
      - Réalise le train/test split,
      - Lance une recherche d'hyperparamètres avec RandomizedSearchCV sur XGBRegressor,
      - Calcule et loggue les métriques dans MLflow,
      - Sauvegarde le meilleur modèle dans MLflow ET localement dans 'models/'.
    """

    # 1. Chargement du CSV prétraité
    if not os.path.exists(DATA_PATH):
        logger.error("Fichier prétraité introuvable: %s", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    logger.info("Données prétraitées chargées : shape=%s", df.shape)

    # 2. Vérifier la présence de la colonne 'Prix'
    if "Prix" not in df.columns:
        logger.error("La colonne 'Prix' est manquante dans le DataFrame prétraité.")
        return

    # 3. Séparation X / y
    y = df["Prix"].values
    X = df.drop(columns=["Prix", "SKU", "Timestamp"])
    logger.info("Séparation X / y : X.shape=%s, y.shape=%s", X.shape, y.shape)

    # 4. Conversion de toutes les colonnes en numérique (pour éviter les erreurs XGBoost)
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        logger.warning(
            "Des valeurs NaN ont été trouvées après conversion en numérique dans X. Remplissage par 0."
        )
        X.fillna(0, inplace=True)

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=17
    )
    logger.info(
        "Split train/test => X_train=%s, X_test=%s", X_train.shape, X_test.shape
    )

    # 6. Définition de la grille d'hyperparamètres pour RandomizedSearchCV
    param_dist = {
        "n_estimators": randint(50, 300),
        "learning_rate": uniform(0.01, 0.2),
        "max_depth": randint(3, 7),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 0.3),
    }

    model_xgb = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=17),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=17,
    )

    # 7. Enregistrement dans MLflow
    mlflow.set_experiment("DynamicPricing")
    with mlflow.start_run(run_name="XGBoost_RandSearch") as run:
        logger.info("Démarrage de la recherche d'hyperparamètres XGB")

        # Entraînement
        model_xgb.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model_xgb.predict(X_test)

        # Calcul des métriques
        mse_xgb = mean_squared_error(y_test, y_pred)
        rmse_xgb = np.sqrt(mse_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred)
        r2_xgb = r2_score(y_test, y_pred)

        logger.info("Meilleurs paramètres: %s", model_xgb.best_params_)
        logger.info(
            "Métriques XGBoost => RMSE=%.4f, MAE=%.4f, R²=%.4f",
            rmse_xgb,
            mae_xgb,
            r2_xgb,
        )

        print("\nMétriques d'évaluation pour XGBoost :")
        print(f"RMSE : {rmse_xgb:.4f}")
        print(f"MAE  : {mae_xgb:.4f}")
        print(f"R²   : {r2_xgb:.4f}")

        # Log MLflow : hyperparamètres et métriques
        mlflow.log_params(model_xgb.best_params_)
        mlflow.log_metric("rmse", rmse_xgb)
        mlflow.log_metric("mae", mae_xgb)
        mlflow.log_metric("r2", r2_xgb)

        # 8. Enregistrement du modèle dans MLflow et localement avec versioning
        input_example = X_test.iloc[0:1]  # Extrait une ligne pour exemple d'input
        os.makedirs(MODEL_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format : YYYYMMDD_HHMMSS
        model_filename = f"xgb_model_{timestamp}"
        model_path = os.path.join(MODEL_DIR, model_filename)

        # Sauvegarde MLflow
        mlflow.sklearn.log_model(
            model_xgb.best_estimator_,
            artifact_path="xgb_model",
            input_example=input_example,
        )
        logger.info(
            f"Modèle XGBoost sauvegardé dans MLflow (artifact_path='xgb_model')"
        )

        # Sauvegarde locale avec versioning
        mlflow.sklearn.save_model(model_xgb.best_estimator_, path=model_path)
        logger.info(f"Modèle sauvegardé localement dans {model_path}")

        # Mettre à jour le fichier de référence vers le dernier modèle
        latest_model_path = os.path.join(MODEL_DIR, "latest_model.txt")
        with open(latest_model_path, "w") as f:
            f.write(model_filename)

        logger.info(f"Fichier {latest_model_path} mis à jour avec {model_filename}")

    logger.info("Fin du script training.py")


if __name__ == "__main__":
    main()
