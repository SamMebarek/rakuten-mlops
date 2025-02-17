# tests/test_ingestion.py
import os
import pytest
import pandas as pd
from src.ingestion import ingest_csv, setup_directories, get_file_hash

# Définition des chemins pour les tests
test_data_dir = "tests/data"
test_logs_dir = "tests/logs"
test_latest_ingested = os.path.join(test_data_dir, "latest_ingested.txt")


def test_setup_directories():
    """
    Vérifie que les répertoires nécessaires sont bien créés.
    """
    setup_directories()
    assert os.path.exists("Logs"), "Le dossier Logs n'a pas été créé."
    assert os.path.exists("Data"), "Le dossier Data n'a pas été créé."


def test_ingest_valid_csv(tmp_path):
    """
    Vérifie que l'ingestion fonctionne avec un fichier valide.
    """
    # Création d'un fichier CSV valide
    csv_content = """SKU,Prix\nSKU1,10.5\nSKU2,15.3"""
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
    Vérifie que l'ingestion d'un fichier inexistant lève une exception.
    """
    with pytest.raises(FileNotFoundError):
        ingest_csv("fichier_inexistant.csv")


def test_ingest_missing_columns(tmp_path):
    """
    Vérifie que l'absence de colonnes essentielles entraîne une erreur.
    """
    csv_content = """Nom,Quantite\nProduitA,5\nProduitB,10"""
    input_csv = tmp_path / "test_missing_columns.csv"
    input_csv.write_text(csv_content)

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        ingest_csv(str(input_csv))


def test_ingest_file_hash(tmp_path):
    """
    Vérifie que la fonction de hash retourne une valeur correcte.
    """
    csv_content = """SKU,Prix\nSKU1,10.5\nSKU2,15.3"""
    test_file = tmp_path / "test_hash.csv"
    test_file.write_text(csv_content)

    file_hash = get_file_hash(str(test_file))
    assert (
        isinstance(file_hash, str) and len(file_hash) == 32
    ), "Le hash MD5 est incorrect."
