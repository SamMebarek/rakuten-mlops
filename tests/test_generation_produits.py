# test/test_generation_produits.py

import pytest
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# On importe les classes à tester
from src.generation import ParametresSynthese, GenerateurDonnees


def test_generer_produits(tmp_path):
    """
    Vérifie la cohérence de la génération de produits (SKU, Catégories, PrixInit, etc.).
    """

    # 1. Créer un faux fichier config.yaml dédié
    config_content = """
seed: 123
random_seed_range: 10001
n_skus: 6
n_periodes: 7
categories:
  Electronique:
    prix_moyen: 100
    prix_ecart_type: 10
    plancher_pct: 0.4
    delta: 0.01
    gamma: 0.001
    demande_lambda: 2
    remise_prob: 0.05
  Livres:
    prix_moyen: 20
    prix_ecart_type: 3
    plancher_pct: 0.2
    delta: 0.005
    gamma: 0.0005
    demande_lambda: 5
    remise_prob: 0.1

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
    config_file = tmp_path / "test_config_generation.yaml"
    config_file.write_text(config_content)

    # 2. Initialiser ParametresSynthese et GenerateurDonnees
    params = ParametresSynthese(config_path=str(config_file))
    gen = GenerateurDonnees(params)

    # 3. Récupérer le DataFrame produits
    df_produits = (
        gen.produits_df
    )  # ou gen.generer_produits() si vous préférez appeler directement la méthode

    # 4. Vérifier la structure et la cohérence
    assert len(df_produits) == 6, "On devrait avoir 6 SKUs au total"
    for col in ["SKU", "Categorie", "PrixInitial", "Qualite", "PlancherPourcentage"]:
        assert col in df_produits.columns, f"La colonne {col} devrait exister"

    # Vérifier la répartition entre les catégories
    value_counts = df_produits["Categorie"].value_counts()
    # Exemple : si on a 2 catégories et 6 SKUs, on s'attend à une répartition (3, 3) ou (4, 2) selon la logique
    # de distribution dynamique. On vérifie simplement que la somme = 6
    assert value_counts.sum() == 6, "La somme des SKUs doit être égale à 6"

    # Vérifier que les prix initiaux sont positifs (loi normale tronquée)
    assert (
        df_produits["PrixInitial"] > 0
    ).all(), "PrixInitial doit être strictement positif"

    # Vérifier la plage de Qualite
    assert (df_produits["Qualite"] >= 0).all() and (
        df_produits["Qualite"] <= 1
    ).all(), "Qualite doit être entre 0 et 1"

    # (Optionnel) Inspecter la distribution par catégories
    # Par exemple, s'il y a 2 catégories, la différence n'excède pas 1 SKU
    # selon la logique. On peut faire un test plus précis en fonction de la config:
    counts = value_counts.to_dict()
    # e.g. On vérifie qu'aucune catégorie n'a 5 SKUs tandis que l'autre en a 1 (à voir selon votre logique).
    for cat, cnt in counts.items():
        assert cnt > 0, f"La catégorie {cat} ne doit pas être vide"

    # BONUS : vérifier le plancher_pct
    # On s'assure que le plancher = 0.4 pour Electronique et 0.2 pour Livres
    # comme défini dans le config_content
    electronique_df = df_produits[df_produits["Categorie"] == "Electronique"]
    livres_df = df_produits[df_produits["Categorie"] == "Livres"]
    assert (
        electronique_df["PlancherPourcentage"] == 0.4
    ).all(), "Le plancher pour Electronique doit être 0.4"
    assert (
        livres_df["PlancherPourcentage"] == 0.2
    ).all(), "Le plancher pour Livres doit être 0.2"
