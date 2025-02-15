# test_generation_calculs.py

import pytest
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.generation import ParametresSynthese, GenerateurDonnees


def test_ajuster_prix_calculs(tmp_path):
    """
    Vérifie la cohérence des colonnes et des valeurs produites par 'ajuster_prix_et_calculs'.
    On teste notamment :
    - Prix final >= plancher
    - ProbabiliteAchat entre 0 et 1
    - QuantiteVendue >= 0
    - Présence et cohérence des colonnes (ElasticitePrix, UtiliteProduit, etc.)
    """

    # 1. Créer un fichier de config minimal
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

    # 2. Initialiser ParametresSynthese et GenerateurDonnees
    params = ParametresSynthese(config_path=str(config_file))
    generateur = GenerateurDonnees(params)

    # 3. Générer le DataFrame complet
    df_result = generateur.ajuster_prix_et_calculs()

    # 4. Vérifier la présence des colonnes clés
    colonnes_attendues = [
        "SKU",
        "Categorie",
        "Timestamp",
        "Prix",
        "PrixPlancher",
        "Remise",
        "Promotion",
        "Qualite",
        "AgeProduitEnJours",
        "Delta",
        "UtiliteProduit",
        "ProbabiliteAchat",
        "QuantiteVendue",
        "ElasticitePrix",
    ]
    for col in colonnes_attendues:
        assert (
            col in df_result.columns
        ), f"La colonne '{col}' devrait exister dans le DataFrame."

    # 5. Vérifier que le prix final ne descend pas sous le plancher
    assert (
        df_result["Prix"] >= df_result["PrixPlancher"]
    ).all(), "Le prix final ne doit jamais être inférieur au plancher."

    # 6. Probabilité d'achat entre 0 et 1
    assert (df_result["ProbabiliteAchat"] >= 0).all() and (
        df_result["ProbabiliteAchat"] <= 1
    ).all(), "ProbabiliteAchat doit être entre 0 et 1."

    # 7. Quantité vendue (poisson) >= 0
    assert (
        df_result["QuantiteVendue"] >= 0
    ).all(), "QuantiteVendue ne doit pas être négative."

    # 8. Vérifier la validité de l'élasticité (entre 0 et 1)
    assert (df_result["ElasticitePrix"] >= 0).all() and (
        df_result["ElasticitePrix"] <= 1
    ).all(), "ElasticitePrix doit être comprise entre 0 et 1."

    # 9. Vérifier la remise (0 ou remise_valeur)
    # Selon la logique binomiale, la remise sera 0 ou 0.1 (dans cet exemple),
    # ou un multiple si vous modifiez la config.
    remise_vals = df_result["Remise"].unique()
    for val in remise_vals:
        assert val in [
            0,
            params.remise_valeur,
        ], f"La remise devrait être 0 ou {params.remise_valeur}, trouvé {val}"

    # 10. Vérifier l'AgeProduitEnJours et la dépréciation
    # Ici, on s'assure juste qu'AgeProduitEnJours >= 0 et correspond à la fourchette possible
    # sur la période. Avec n_periodes=3, plus la journée courante, on peut aller jusqu'à 3 ou 4
    # selon la logique d'inclusion du jour. On tolère un peu plus.
    max_age = df_result["AgeProduitEnJours"].max()
    assert (
        max_age <= params.n_periodes + 1
    ), f"AgeProduitEnJours semble trop élevé. max_age={max_age}, n_periodes={params.n_periodes}."
    assert (
        df_result["AgeProduitEnJours"] >= 0
    ).all(), "AgeProduitEnJours ne doit pas être négatif."

    # 11. (Optionnel) Vérifier la cohérence statistique
    # Par exemple, la moyenne du prix est censée être < prix_moyen du config,
    # à cause de la dépréciation.
    electronique_moy = df_result[df_result["Categorie"] == "Electronique"][
        "Prix"
    ].mean()
    # On vérifie juste qu'elle ne dépasse pas trop le prix_moyen
    # On a mis 100 comme prix moyen, on s'attend à ce que la moyenne soit inférieure à ~100.
    assert (
        electronique_moy < 120
    ), f"Le prix moyen pour Electronique semble trop élevé : {electronique_moy:.2f}"

    # On peut faire d'autres assertions spécifiques si nécessaire.

    # Si on arrive ici, c'est que tous les checks sont passés
    print("Les calculs de prix et de demande sont cohérents.")
