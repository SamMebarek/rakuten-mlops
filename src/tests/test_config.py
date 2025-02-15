# tests/test_config.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import yaml
from src.generation import ParametresSynthese


def test_config_chargement_ok():
    """
    Vérifie que la classe ParametresSynthese lit correctement le fichier 'config.yaml'
    existant dans le projet.
    """
    # GIVEN: un chemin vers le fichier de configuration par défaut
    base_dir = os.path.dirname(__file__)  # Emplacement de test_config.py
    config_path = os.path.join(base_dir, "..", "config.yaml")

    # On s'assure que le chemin est absolu pour éviter des problèmes de résolution
    config_path = os.path.abspath(config_path)

    assert os.path.exists(config_path), "Le fichier config.yaml est introuvable."

    # WHEN: on instancie ParametresSynthese
    params = ParametresSynthese(config_path=config_path)

    # THEN: on s'assure que certains attributs clés existent et sont cohérents
    assert params.seed is not None, "Le seed n'a pas été correctement chargé."
    assert params.n_skus > 0, "Le nombre de SKUs devrait être > 0."
    assert params.n_periodes > 0, "Le nombre de périodes devrait être > 0."
    assert (
        len(params.parametres_prix_par_categorie) > 0
    ), "La config devrait contenir au moins une catégorie."


def test_config_loading_valid(tmp_path):
    """
    Vérifie qu'un config.yaml minimal et valide est correctement chargé
    depuis un fichier temporaire.
    """
    # 1. Créer un faux fichier config.yaml minimal
    config_content = """
seed: 123
random_seed_range: 10001
n_skus: 50
n_periodes: 10
categories:
  Electronique:
    prix_moyen: 450
    prix_ecart_type: 50
    plancher_pct: 0.4
    delta: 0.01
    gamma: 0.0015
    demande_lambda: 3
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
    config_file = tmp_path / "test_config_valid.yaml"
    config_file.write_text(config_content)

    # 2. Initialiser ParametresSynthese avec ce fichier
    params = ParametresSynthese(config_path=str(config_file))

    # 3. Vérifier que les champs sont correctement chargés
    assert params.seed == 123, "Le seed devrait valoir 123"
    assert params.n_skus == 50, "Le nombre de SKUs devrait valoir 50"
    assert params.n_periodes == 10, "Le nombre de périodes devrait valoir 10"
    assert (
        "Electronique" in params.parametres_prix_par_categorie
    ), "La catégorie 'Electronique' devrait exister"
    assert params.remise_valeur == 0.1, "La remise valeur devrait valoir 0.1"


def test_config_missing_field(tmp_path):
    """
    Vérifie la réaction du code si un champ essentiel est manquant.
    Selon la stratégie, le code peut lever une exception ou gérer un fallback.
    """
    config_content = """
random_seed_range: 9999
n_skus: 40
"""
    config_file = tmp_path / "test_config_missing.yaml"
    config_file.write_text(config_content)

    # Si vous voulez que le code lève une exception (KeyError, par ex.)
    with pytest.raises(KeyError):
        _ = ParametresSynthese(config_path=str(config_file))

    # Si vous préférez gérer un fallback, remplacez par :
    # params = ParametresSynthese(config_path=str(config_file))
    # assert params.seed is not None, "Le seed devrait être défini par défaut."


def test_config_valeurs_incorrectes(tmp_path):
    """
    Vérifie la réaction du code si un champ critique a un type ou une valeur invalide.
    Par exemple, seed doit être un int, n_skus doit être positif, etc.
    """
    data_invalide = {
        "seed": "should_be_an_int",  # Mauvais type
        "n_skus": -10,  # Valeur négative
        # n_periodes manquant
    }

    config_file = tmp_path / "test_config_invalide.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_invalide, f)

    # On s'attend à une exception (ValueError, TypeError, KeyError, etc. selon l'implémentation)
    with pytest.raises(Exception):
        _ = ParametresSynthese(config_path=str(config_file))
