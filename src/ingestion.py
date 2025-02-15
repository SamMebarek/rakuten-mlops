# src/ingestion.py

import pandas as pd
import os

from typing import Optional


def ingest_csv(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Importe un CSV depuis 'input_path' et renvoie un DataFrame.
    Si 'output_path' est spécifié, le DataFrame est également
    réécrit dans ce fichier CSV.

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
    """

    # Vérifier l'existence du fichier
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} est introuvable.")

    # Lecture du CSV
    df = pd.read_csv(input_path)
    print(f"[ingestion] Fichier chargé depuis : {input_path} (shape={df.shape})")

    # Export éventuel du DataFrame
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[ingestion] Fichier réécrit dans : {output_path} (shape={df.shape})")

    return df


def main():

    input_path = "Data/donnees_synthetiques.csv"
    output_path = "Data/ingested_data.csv"

    df_data = ingest_csv(input_path=input_path, output_path=output_path)
    print(f"[ingestion] Nombre d’enregistrements ingérés : {len(df_data)}")


if __name__ == "__main__":
    main()
