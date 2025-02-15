# src/prediction.py

"""
Ce script illustre la prédiction à partir d'un modèle XGBoost 
précédemment entraîné et sauvegardé.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import os


def predict(model_path: str, data_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    1. Charge un modèle XGBoost sauvegardé (ex. .json).
    2. Charge un CSV de features prétraitées (ou partiellement).
    3. Fait la prédiction et renvoie/publie les résultats.
    4. Si output_csv est spécifié, sauvegarde dans un CSV.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier de modèle {model_path} est introuvable.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Le fichier de données {data_csv} est introuvable.")

    df_data = pd.read_csv(data_csv)
    # Supposons que df_data contient les mêmes colonnes de features que lors du training
    # Sinon, adapter la transformation (scaling, encodage, etc.)
    X = df_data.drop(columns=["Prix"], errors="ignore")  # s'il y a une colonne Prix
    preds = model.predict(X)

    results_df = df_data.copy()
    results_df["Prediction"] = preds

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Prédictions sauvegardées dans {output_csv}")

    return results_df


if __name__ == "__main__":
    # Exemple d'utilisation
    preds_df = predict(
        model_path="models/xgb_model.json",
        data_csv="Data/donnees_validation_prepro.csv",
        output_csv="Data/predictions.csv",
    )
    print("Exemple de quelques prédictions :")
    print(preds_df.head())


# --------------------------

# # src/prediction.py

# import mlflow
# import mlflow.sklearn
# import pandas as pd

# def load_model(run_id: str = None, model_uri: str = None):
#     """
#     Charge un modèle XGBoost depuis MLflow.
#     - Soit via un run_id (dernier run)
#     - Soit via un URI spécifique (ex: "runs:/<run_id>/xgboost_model")
#     Returns:
#         model: objet modèle XGBoost
#     """
#     if run_id:
#         model_uri = f"runs:/{run_id}/xgboost_model"
#     if not model_uri:
#         raise ValueError("Veuillez spécifier un run_id ou un model_uri valide.")

#     print(f"[prediction] Chargement du modèle depuis {model_uri}")
#     model = mlflow.sklearn.load_model(model_uri)
#     return model

# def predict(model, df_features: pd.DataFrame):
#     """
#     Applique le modèle pour faire une prédiction sur un DataFrame de features déjà prétraitées.
#     Returns:
#         np.array: vecteur de prédictions
#     """
#     y_pred = model.predict(df_features)
#     return y_pred

# if __name__ == "__main__":
#     # Exemple d'usage
#     # 1. Charger le modèle via un run_id connu
#     run_id = "xxxxxxxxxxxx"
#     model = load_model(run_id=run_id)

#     # 2. Appliquer la prédiction sur un set de features
#     # df_features = ...
#     # y_pred = predict(model, df_features)
#     pass
