import os
import pytest
import pandas as pd
from src.ingestion.ingestion import ingest_csv, setup_directories, get_file_hash


def test_ingest_valid_csv(tmp_path):
    """
    Vérifie que l'ingestion fonctionne avec un fichier CSV valide.
    """
    csv_content = "SKU,Prix\nSKU1,10.5\nSKU2,15.3"
    input_csv = tmp_path / "test_valid.csv"
    input_csv.write_text(csv_content)
    output_csv = tmp_path / "output.csv"

    df = ingest_csv(str(input_csv), str(output_csv))
    assert not df.empty, "Le DataFrame retourné est vide."
    assert (
        "SKU" in df.columns and "Prix" in df.columns
    ), "Les colonnes attendues ne sont pas présentes."
    assert df.shape[0] == 2, "Le nombre de lignes est incorrect."


def test_ingest_missing_file():
    """
    Vérifie que l'ingestion d'un fichier inexistant lève une exception FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        ingest_csv("fichier_inexistant.csv")


def test_ingest_missing_columns(tmp_path):
    """
    Vérifie que l'absence de colonnes essentielles (SKU et Prix) déclenche une ValueError.
    """
    csv_content = "Nom,Quantite\nProduitA,5\nProduitB,10"
    input_csv = tmp_path / "test_missing_columns.csv"
    input_csv.write_text(csv_content)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        ingest_csv(str(input_csv))
