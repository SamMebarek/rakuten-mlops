# tests/generation.py

import pytest
import sys
import pandas as pd
from pathlib import Path
from src.generation.generation import ParametresSynthese, GenerateurDonnees, main

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def test_config_loading(params):
    """Vérifie que la configuration est correctement chargée."""
    assert params.seed == 42
    assert params.n_skus == 5
    assert params.n_periodes == 3
    assert "Electronique" in params.parametres_prix_par_categorie


def test_generer_produits(generator):
    """Vérifie que la génération des produits renvoie un DataFrame avec les colonnes attendues."""
    df_produits = generator.produits_df
    expected_columns = [
        "SKU",
        "Categorie",
        "PrixInitial",
        "Qualite",
        "PlancherPourcentage",
    ]
    for col in expected_columns:
        assert col in df_produits.columns
    # Le nombre total de produits doit correspondre à n_skus (ici 5)
    assert len(df_produits) == 5


def test_generer_dates(generator):
    """Vérifie que la génération des dates produit un nombre de timestamps cohérent."""
    df_dates = generator.dates_df
    # Pour n_periodes = 3, on s'attend à avoir entre 9 et 12 timestamps (si la journée courante est partiellement générée)
    nb_ts = len(df_dates)
    assert nb_ts >= 3 * 3 and nb_ts <= 4 * 3


def test_ajuster_prix_et_calculs(generator):
    """Vérifie que le DataFrame final contient bien les colonnes essentielles et n'est pas vide."""
    df_final = generator.ajuster_prix_et_calculs()
    expected_columns = [
        "SKU",
        "Categorie",
        "Timestamp",
        "Date",
        "Prix",
        "PrixInitial",
        "Remise",
        "ErreurAleatoire",
        "Promotion",
        "Qualite",
        "AgeProduitEnJours",
        "PlancherPourcentage",
        "PrixPlancher",
        "Delta",
        "UtiliteProduit",
        "ProbabiliteAchat",
        "DemandeLambda",
        "QuantiteVendue",
        "ElasticitePrix",
        "DateLancement",
    ]
    for col in expected_columns:
        assert col in df_final.columns, f"Il manque la colonne {col}"
    assert not df_final.empty


def test_main_creates_csv(tmp_path):
    """
    Test d'intégration de main() :
    - Vérifie que le CSV est créé dans le chemin spécifié.
    - Vérifie que le CSV contient des colonnes clés et n'est pas vide.
    """
    output_csv = tmp_path / "donnees_synthetiques.csv"
    main(output_path=str(output_csv))
    assert output_csv.exists(), "Le fichier CSV n'a pas été créé."
    df = pd.read_csv(output_csv)
    assert not df.empty, "Le CSV généré est vide."
    for col in ["SKU", "Categorie", "Prix", "QuantiteVendue"]:
        assert col in df.columns, f"Le CSV doit contenir la colonne {col}."
