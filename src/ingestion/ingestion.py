# src/ingestion/ingestion.py

import pandas as pd
import os
import logging
import hashlib
from datetime import datetime
from typing import Optional
import yaml

# Configuration du logging pour le module ingestion
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "ingestion.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def setup_directories():
    """
    Crée les répertoires nécessaires si absents.
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def get_file_hash(file_path: str) -> str:
    """
    Calcule le hash MD5 d'un fichier pour garantir son intégrité.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def ingest_csv(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Importe un CSV depuis 'input_path' et renvoie un DataFrame.
    Si 'output_path' est spécifié, le DataFrame est sauvegardé dans ce fichier CSV.

    Parameters
    ----------
    input_path : str
        Chemin du fichier CSV d'entrée.
    output_path : str, optional
        Chemin du fichier CSV de sortie.
        Si None, le DataFrame n'est pas réexporté.

    Returns
    -------
    df : pd.DataFrame
        DataFrame contenant les données importées.

    Raises
    ------
    FileNotFoundError
        Si le fichier d'entrée n'existe pas.
    ValueError
        Si le fichier est vide ou ne contient pas les colonnes attendues.
    """
    if not os.path.exists(input_path):
        logging.error(f"Le fichier {input_path} est introuvable.")
        raise FileNotFoundError(f"Le fichier {input_path} est introuvable.")

    try:
        df = pd.read_csv(input_path, sep=",", encoding="utf-8", low_memory=False)
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        raise ValueError(f"Erreur lors de la lecture du fichier CSV : {e}")

    if df.empty:
        logging.error(f"Le fichier {input_path} est vide.")
        raise ValueError(f"Le fichier {input_path} est vide.")

    # Vérification de colonnes essentielles
    required_columns = ["SKU", "Prix"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes : {missing_columns} dans {input_path}")
        raise ValueError(f"Colonnes manquantes : {missing_columns} dans {input_path}")

    logging.info(f"Fichier chargé depuis : {input_path} (shape={df.shape})")
    file_hash = get_file_hash(input_path)
    logging.info(f"Hash du fichier ingéré : {file_hash}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logging.info(f"Fichier sauvegardé dans : {output_path} (shape={df.shape})")

    return df


def run_ingestion(input_path: str, output_path: str):
    """
    Fonction principale exécutant l'ingestion des données.
    """
    setup_directories()
    try:
        df_data = ingest_csv(input_path=input_path, output_path=output_path)
        logging.info(f"Nombre d'enregistrements ingérés : {len(df_data)}")
    except Exception as e:
        logging.error(f"Erreur dans l'ingestion : {e}")


if __name__ == "__main__":
    # Chargement des paramètres depuis params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["ingestion"]
    input_path = params.get("input", "data/donnees_synthetiques.csv")
    output_path = params.get("output", "data/raw/ingested_data.csv")
    run_ingestion(input_path, output_path)
