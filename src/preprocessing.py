# preprocessing.py

import logging
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocessing")


# Suppression des colonnes inutiles
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        columns=[
            "DateLancement",
            "PrixPlancher",
            "PlancherPourcentage",
            "ErreurAleatoire",
            "Annee",
            "Mois",
            "Jour",
            "Heure",
            "Promotion",
            "Categorie",
        ],
        errors="ignore",
    )
    return df


# Conversion des types
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SKU"] = df["SKU"].astype("string")  # Conservation de SKU
    df["PrixInitial"] = df["PrixInitial"].astype("float64")
    df["AgeProduitEnJours"] = df["AgeProduitEnJours"].astype("int64")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Prix"] = df["Prix"].astype("float64")
    df["QuantiteVendue"] = df["QuantiteVendue"].astype("int64")
    df["UtiliteProduit"] = df["UtiliteProduit"].astype("float64")
    df["ElasticitePrix"] = df["ElasticitePrix"].astype("float64")
    df["Remise"] = df["Remise"].astype("float64")
    df["Qualite"] = df["Qualite"].astype("float64")
    return df


# Ajout des variables temporelles
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Mois_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["Mois_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["Heure_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["Heure_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)
    return df


# Suppression des valeurs extrêmes dans ElasticitePrix
def clip_elasticite(df: pd.DataFrame) -> pd.DataFrame:
    q_low, q_high = df["ElasticitePrix"].quantile([0.01, 0.99])
    df["ElasticitePrix"] = np.clip(df["ElasticitePrix"], q_low, q_high)
    return df


# Définition du pipeline
preprocessing_pipeline = Pipeline(
    [
        ("convert_types", FunctionTransformer(convert_types, validate=False)),
        ("time_features", FunctionTransformer(add_time_features, validate=False)),
        ("clip_elasticite", FunctionTransformer(clip_elasticite, validate=False)),
        ("drop_unused", FunctionTransformer(drop_unused_columns, validate=False)),
    ]
)


# Exécution principale du preprocessing
def run_preprocessing(
    input_csv="Data/donnees_synthetiques.csv", output_csv="Data/preprocessed_data.csv"
):
    logger.info("Chargement des données...")
    df = pd.read_csv(input_csv)

    # **Conserver SKU avant transformation**
    sku_column = df["SKU"].copy()

    # Application du pipeline
    df_processed = preprocessing_pipeline.fit_transform(df)

    # Colonnes finales après transformation
    remaining_columns = [
        "Prix",  # La cible est toujours présente
        "PrixInitial",
        "Timestamp",
        "AgeProduitEnJours",
        "QuantiteVendue",
        "UtiliteProduit",
        "ElasticitePrix",
        "Remise",
        "Qualite",
        "Mois_sin",
        "Mois_cos",
        "Heure_sin",
        "Heure_cos",
    ]

    # Création du DataFrame final
    df_final = pd.DataFrame(df_processed, columns=remaining_columns)

    # Réintégration de SKU
    df_final.insert(0, "SKU", sku_column.values)

    # Vérification avant sauvegarde
    logger.info(f"Shape finale du DataFrame : {df_final.shape}")
    logger.info(f"Colonnes finales : {df_final.columns.tolist()}")

    # Sauvegarde
    df_final.to_csv(output_csv, index=False)

    logger.info("Données prétraitées sauvegardées avec succès.")


if __name__ == "__main__":
    run_preprocessing()
