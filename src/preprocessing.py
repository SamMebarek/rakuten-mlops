# src/preprocessing.py

import logging
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Configuration du logging
logging.basicConfig(
    filename="Logs/preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocessing")

# Fichier contenant la dernière version ingérée et prétraitée
LATEST_INGESTED_FILE = "Data/latest_ingested.txt"
LATEST_PREPROCESSED_FILE = "Data/latest_preprocessed.txt"


def get_latest_file(file_path):
    """Récupère le dernier fichier stocké dans un fichier de référence"""
    if not os.path.exists(file_path):
        logger.error(f"Fichier de référence {file_path} introuvable.")
        raise FileNotFoundError(f"Fichier de référence {file_path} introuvable.")
    with open(file_path, "r") as f:
        return f.read().strip()


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles en vérifiant leur existence"""
    columns_to_drop = [
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
    ]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns, errors="ignore")


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les types des colonnes et gère les valeurs manquantes"""
    df = df.copy()
    df["SKU"] = df["SKU"].astype("string")
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables temporelles en utilisant la colonne Timestamp"""
    df = df.copy()
    df["Mois_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["Mois_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["Heure_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["Heure_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)
    return df


# Définition du pipeline de preprocessing
preprocessing_pipeline = Pipeline(
    [
        ("convert_types", FunctionTransformer(convert_types, validate=False)),
        ("time_features", FunctionTransformer(add_time_features, validate=False)),
        ("drop_unused", FunctionTransformer(drop_unused_columns, validate=False)),
    ]
)


def run_preprocessing():
    """Exécute le pipeline de preprocessing"""
    try:
        input_csv = get_latest_file(LATEST_INGESTED_FILE)
        output_csv = "Data/preprocessed_data.csv"
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        versioned_output_csv = output_csv.replace(".csv", f"_{timestamp}.csv")

        logger.info(f"Chargement des données depuis {input_csv}")
        df = pd.read_csv(input_csv, encoding="utf-8")

        if df.empty:
            logger.error("Le fichier est vide après chargement.")
            raise ValueError("Le fichier est vide après chargement.")

        # Conserver SKU avant transformation
        sku_column = df["SKU"].copy()

        # Application du pipeline
        df_processed = preprocessing_pipeline.fit_transform(df)

        # Colonnes finales attendues
        expected_columns = [
            "Prix",
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

        # Vérification finale
        df_final = pd.DataFrame(df_processed, columns=expected_columns)
        df_final.insert(0, "SKU", sku_column.values)
        logger.info(f"Shape finale du DataFrame : {df_final.shape}")

        # Sauvegarde du fichier prétraité
        df_final.to_csv(versioned_output_csv, index=False, encoding="utf-8")
        logger.info(f"Données prétraitées sauvegardées dans {versioned_output_csv}")

        # Enregistrement de la dernière version
        with open(LATEST_PREPROCESSED_FILE, "w") as f:
            f.write(versioned_output_csv)
        logger.info(
            f"Dernier fichier prétraité enregistré dans {LATEST_PREPROCESSED_FILE}"
        )

    except Exception as e:
        logger.error(f"Erreur dans le preprocessing : {e}")
        raise


if __name__ == "__main__":
    run_preprocessing()
