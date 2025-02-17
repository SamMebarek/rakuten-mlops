# test/test_generation_calculs.py

import pytest
import os
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from src.generation import ParametresSynthese, GenerateurDonnees

# Configuration du logging
logging.basicConfig(
    filename="Logs/test_generation_calculs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_generation_calculs")


@pytest.fixture
def setup_generation(tmp_path):
    """
    Prépare un environnement de test avec une configuration minimaliste.
    Retourne le DataFrame généré par ajuster_prix_et_calculs.
    """
    config_content = """
    seed: 42
    random_seed_range: 10001
    n_skus: 5
    n_periodes: 3

    categories:
      Electronique:
        prix_moyen: 100
        prix_ecart_type: 20
        plancher_pct: 0.5
        delta: 0.01
        gamma: 0.001
        demande_lambda: 2
        remise_prob: 0.05

    beta_prix_mean: 0.05
    beta_prix_std: 0.005
    beta_qualite_mean: 0.05
    beta_qualite_std: 0.005
    beta_promo_mean: 0.03
    beta_promo_std: 0.002
    erreur_std: 1
    remise_valeur: 0.1
    prix_minimum: 0.01
    """
    config_file = tmp_path / "test_config_calculs.yaml"
    config_file.write_text(config_content)

    params = ParametresSynthese(config_path=str(config_file))
    generateur = GenerateurDonnees(params)
    df_result = generateur.ajuster_prix_et_calculs()

    return df_result, params


def test_prix_coherence(setup_generation):
    """
    Vérifie que les prix ne descendent jamais sous le plancher.
    """
    df_result, _ = setup_generation
    assert (
        df_result["Prix"] >= df_result["PrixPlancher"]
    ).all(), "Le prix final ne doit jamais être inférieur au plancher."
    logger.info("✅ Test prix_coherence : OK")


def test_probabilite_achat(setup_generation):
    """
    Vérifie que la probabilité d'achat est comprise entre 0 et 1.
    """
    df_result, _ = setup_generation
    assert (df_result["ProbabiliteAchat"] >= 0).all() and (
        df_result["ProbabiliteAchat"] <= 1
    ).all(), "ProbabiliteAchat doit être entre 0 et 1."
    logger.info("✅ Test probabilite_achat : OK")


def test_quantite_vendue(setup_generation):
    """
    Vérifie que la quantité vendue ne peut pas être négative.
    """
    df_result, _ = setup_generation
    assert (
        df_result["QuantiteVendue"] >= 0
    ).all(), "QuantiteVendue ne doit pas être négative."
    logger.info("✅ Test quantite_vendue : OK")


def test_elasticite(setup_generation):
    """
    Vérifie que l'élasticité prix est comprise entre 0 et 1.
    """
    df_result, _ = setup_generation
    assert (df_result["ElasticitePrix"] >= 0).all() and (
        df_result["ElasticitePrix"] <= 1
    ).all(), "ElasticitePrix doit être comprise entre 0 et 1."
    logger.info("✅ Test elasticite : OK")


def test_remise(setup_generation):
    """
    Vérifie que les remises appliquées sont soit 0, soit la valeur définie dans la configuration.
    """
    df_result, params = setup_generation
    remise_vals = df_result["Remise"].unique()
    for val in remise_vals:
        assert val in [
            0,
            params.remise_valeur,
        ], f"La remise devrait être 0 ou {params.remise_valeur}, trouvé {val}"
    logger.info("✅ Test remise : OK")
