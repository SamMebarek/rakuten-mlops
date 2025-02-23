import pytest
import pandas as pd
import numpy as np
import builtins
from src.preprocessing.preprocessing import (
    drop_unused_columns,
    convert_types,
    add_time_features,
    preprocessing_pipeline,
    run_preprocessing,
)


def test_drop_unused_columns(sample_dataframe):
    """
    Vérifie que drop_unused_columns supprime les colonnes inutiles.
    """
    df_clean = drop_unused_columns(sample_dataframe)
    for col in [
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
    ]:
        assert col not in df_clean.columns, f"La colonne {col} n'a pas été supprimée."
    assert "Prix" in df_clean.columns, "La colonne 'Prix' doit être conservée."


def test_convert_types(sample_dataframe):
    """
    Vérifie que convert_types transforme correctement les types des colonnes.
    """
    df_converted = convert_types(sample_dataframe)
    assert df_converted["SKU"].dtype == "string"
    assert df_converted["PrixInitial"].dtype == "float64"
    assert df_converted["Prix"].dtype == "float64"
    assert (
        df_converted["AgeProduitEnJours"].dtype == "int64"
    ), "AgeProduitEnJours doit être int64."
    assert pd.api.types.is_datetime64_any_dtype(
        df_converted["Timestamp"]
    ), "Timestamp doit être datetime64."


def test_add_time_features(sample_dataframe):
    """
    Vérifie que add_time_features ajoute les colonnes temporelles attendues.
    """
    df = sample_dataframe.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df_transformed = add_time_features(df)
    for col in ["Mois_sin", "Mois_cos", "Heure_sin", "Heure_cos"]:
        assert col in df_transformed.columns, f"La colonne {col} doit être présente."


def test_preprocessing_pipeline(sample_dataframe):
    """
    Vérifie que le pipeline de preprocessing renvoie un array avec un nombre de colonnes cohérent.
    """
    transformed = preprocessing_pipeline.fit_transform(sample_dataframe)
    assert (
        transformed.shape[1] >= 13
    ), "Le nombre de colonnes après transformation est insuffisant."


def test_run_preprocessing(tmp_path, temp_preprocessing_config, monkeypatch):
    """
    Test d'intégration de run_preprocessing:
    - Utilise un fichier d'ingestion et une configuration temporaire.
    - Redirige l'accès à "params.yaml" vers le fichier de configuration temporaire.
    - Vérifie que le CSV de sortie est créé et contient les colonnes clés.
    """
    config_file, output_csv = temp_preprocessing_config

    # Patch builtins.open pour rediriger "params.yaml" vers notre fichier temporaire
    import builtins

    original_open = builtins.open

    def fake_open(filename, *args, **kwargs):
        if filename == "params.yaml":
            return original_open(str(config_file), *args, **kwargs)
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    # Exécuter run_preprocessing
    from src.preprocessing.preprocessing import run_preprocessing

    run_preprocessing()

    # Vérifier que le CSV de sortie est créé et contient des colonnes clés
    assert output_csv.exists(), "Le fichier de sortie n'a pas été créé."
    df_output = pd.read_csv(str(output_csv), encoding="utf-8")
    for col in ["SKU", "Prix", "QuantiteVendue"]:
        assert (
            col in df_output.columns
        ), f"Le CSV de sortie doit contenir la colonne {col}."
    assert not df_output.empty, "Le CSV de sortie est vide."
