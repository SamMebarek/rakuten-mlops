# src/generation/generation.py

"""
Ce script illustre une génération de données synthétiques en se basant sur une configuration YAML.
Il génère les dates sur n_periodes jours avant la date et l'heure actuelles, avec 3 observations par jour.
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from datetime import timedelta
import logging

# Création du dossier des logs
os.makedirs("logs", exist_ok=True)

# Configuration du logging pour le module "generation"
logging.basicConfig(
    filename=os.path.join("logs", "generation.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Récupération du répertoire actuel de ce module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin par défaut vers config.yaml (situé dans le même dossier que ce module)
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "config.yaml")

# Création du dossier data si inexistant
os.makedirs("data", exist_ok=True)


class ParametresSynthese:
    """
    Charge les paramètres de génération à partir d'un fichier YAML.
    Ici, n_periodes indique le nombre de jours à remonter depuis le lancement du script.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        # Lecture du fichier YAML
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Paramètres généraux
        self.seed = config["seed"]
        self.random_seed_range = config["random_seed_range"]
        np.random.seed(self.seed)

        # Paramètres des catégories
        self.parametres_prix_par_categorie = config["categories"]

        # Paramètres globaux
        self.n_skus = config["n_skus"]
        self.n_periodes = config["n_periodes"]  # Nombre de jours à remonter

        # Coefficients pour le modèle
        self.beta_prix = -abs(
            np.random.normal(config["beta_prix_mean"], config["beta_prix_std"])
        )
        self.beta_qualite = abs(
            np.random.normal(config["beta_qualite_mean"], config["beta_qualite_std"])
        )
        self.beta_promo = abs(
            np.random.normal(config["beta_promo_mean"], config["beta_promo_std"])
        )
        self.erreur_std = config["erreur_std"]

        # Paramètres pour les remises
        self.remise_valeur = config["remise_valeur"]
        self.prix_minimum = config["prix_minimum"]


class GenerateurDonnees:
    """
    Regroupe les méthodes pour générer un DataFrame de données synthétiques
    (produits, timestamps, prix ajustés, utilité, probabilité d'achat, etc.),
    sur n_periodes jours avant l'instant présent.
    """

    def __init__(self, params: ParametresSynthese):
        self.params = params

        # Génération des produits
        self.produits_df = self.generer_produits()

        # Génération dynamique des dates (3 timestamps/jour) depuis (maintenant - n_periodes jours) jusqu'à maintenant
        self.dates_df = self.generer_dates()

    def generer_produits(self) -> pd.DataFrame:
        """Génère la liste des produits (SKU), leur catégorie et leurs attributs (prix init., qualité...)."""
        skus = []
        categories = []
        prix_initials = []
        qualites = []
        plancher_pourcentages = []

        categories_list = list(self.params.parametres_prix_par_categorie.keys())
        total_categories = len(categories_list)

        # Répartition dynamique des SKUs entre les catégories
        base_skus_par_categorie = self.params.n_skus // total_categories
        skus_reste = self.params.n_skus % total_categories
        skus_par_categorie = [base_skus_par_categorie] * total_categories
        for i in range(skus_reste):
            skus_par_categorie[i] += 1

        for idx, categorie_nom in enumerate(categories_list):
            param_categorie = self.params.parametres_prix_par_categorie[categorie_nom]
            for sku_id in range(1, skus_par_categorie[idx] + 1):
                sku = f"SKU{idx + 1}_{sku_id}"
                skus.append(sku)
                categories.append(categorie_nom)

                # Génération du prix initial via loi normale tronquée (positif uniquement)
                prix_moyen = param_categorie["prix_moyen"]
                prix_ecart_type = param_categorie["prix_ecart_type"]
                a, b = (0 - prix_moyen) / prix_ecart_type, float("inf")
                prix_initial = truncnorm.rvs(
                    a, b, loc=prix_moyen, scale=prix_ecart_type
                )
                prix_initials.append(prix_initial)

                # Qualité entre 0 et 1
                qualite = np.random.uniform(0, 1)
                qualites.append(qualite)

                # Stocker le pourcentage plancher
                plancher_pourcentages.append(param_categorie["plancher_pct"])

        produits_df = pd.DataFrame(
            {
                "SKU": skus,
                "Categorie": categories,
                "PrixInitial": prix_initials,
                "Qualite": qualites,
                "PlancherPourcentage": plancher_pourcentages,
            }
        )

        return produits_df

    def generer_dates(self) -> pd.DataFrame:
        """
        Génère un DataFrame de timestamps couvrant n_periodes jours avant 'maintenant',
        avec 3 observations par jour (matin, après-midi et soir).
        """
        now = pd.Timestamp.now().floor("h")  # arrondi à l'heure
        start_datetime = now - pd.Timedelta(days=self.params.n_periodes)

        # Génération d'une liste de jours entiers entre start_datetime et now
        day_range = pd.date_range(
            start=start_datetime.normalize(), end=now.normalize(), freq="D"
        )

        timestamps = []
        intervals_per_day = 3
        hours_between = 24 // intervals_per_day

        for day in day_range:
            for interval in range(intervals_per_day):
                current_ts = day + pd.Timedelta(hours=interval * hours_between)
                # On ne génère pas au-delà de "now"
                if current_ts > now:
                    break
                timestamps.append(current_ts)

        return pd.DataFrame({"Timestamp": timestamps})

    def ajuster_prix_et_calculs(self) -> pd.DataFrame:
        """
        Génère le DataFrame final contenant les prix ajustés dans le temps,
        ainsi que la probabilité d'achat, la quantité vendue, etc.
        """
        prix_liste = []
        # On fixe un point de repère
        earliest_ts = self.dates_df["Timestamp"].min().normalize()

        for _, produit in self.produits_df.iterrows():
            categorie_nom = produit["Categorie"]
            param_categorie = self.params.parametres_prix_par_categorie[categorie_nom]

            gamma = param_categorie["gamma"]
            prix_plancher = produit["PrixInitial"] * param_categorie["plancher_pct"]
            remise_prob = param_categorie["remise_prob"]

            for _, ts_row in self.dates_df.iterrows():
                timestamp = ts_row["Timestamp"]
                # Calcul de l'âge du produit par rapport au début de la fenêtre simulée
                age_produit = (timestamp.normalize() - earliest_ts).days

                # Dépréciation exponentielle
                depreciation = np.exp(-gamma * age_produit)
                prix = produit["PrixInitial"] * depreciation

                # Remise aléatoire
                remise = np.random.binomial(1, remise_prob) * self.params.remise_valeur
                prix_apres_remise = prix * (1 - remise)

                # Variation intra-journalière
                variation_intra_jour = np.random.normal(0, self.params.erreur_std / 2)
                prix_final = prix_apres_remise + variation_intra_jour

                # Application du plancher
                prix_final = max(prix_final, prix_plancher)

                # Promotion binaire
                promotion = 1 if remise > 0 else 0

                # Construction des observations
                prix_liste.append(
                    {
                        "SKU": produit["SKU"],
                        "Categorie": categorie_nom,
                        "Timestamp": timestamp,
                        "Date": timestamp.normalize(),
                        "Prix": prix_final,
                        "PrixInitial": produit["PrixInitial"],
                        "Remise": remise,
                        "ErreurAleatoire": variation_intra_jour,
                        "Promotion": promotion,
                        "Qualite": produit["Qualite"],
                        "AgeProduitEnJours": age_produit,
                        "PlancherPourcentage": param_categorie["plancher_pct"],
                        "PrixPlancher": prix_plancher,
                    }
                )

        prix_df = pd.DataFrame(prix_liste)

        # Calculs supplémentaires : utilité, probabilité, etc.
        # 1. Fusion avec delta
        prix_df["Delta"] = prix_df["Categorie"].map(
            lambda x: self.params.parametres_prix_par_categorie[x]["delta"]
        )

        # 2. Erreur aléatoire pour l'utilité
        erreur_utilite = np.random.normal(0, self.params.erreur_std, size=len(prix_df))
        prix_df["UtiliteProduit"] = (
            self.params.beta_prix * prix_df["Prix"]
            + self.params.beta_qualite * prix_df["Qualite"]
            + self.params.beta_promo * prix_df["Promotion"]
            - prix_df["Delta"] * prix_df["AgeProduitEnJours"]
            + erreur_utilite
        )

        # 3. Probabilité d'achat (sigmoïde)
        prix_df["ProbabiliteAchat"] = 1 / (1 + np.exp(-prix_df["UtiliteProduit"]))

        # 4. Quantité vendue (poisson)
        prix_df["DemandeLambda"] = prix_df["Categorie"].map(
            lambda x: self.params.parametres_prix_par_categorie[x]["demande_lambda"]
        )
        prix_df["QuantiteVendue"] = np.random.poisson(
            lam=prix_df["ProbabiliteAchat"] * prix_df["DemandeLambda"]
        )

        # 5. Élasticité prix
        prix_df["ElasticitePrix"] = abs(self.params.beta_prix) * (
            1 - prix_df["ProbabiliteAchat"]
        )
        prix_df["ElasticitePrix"] = prix_df["ElasticitePrix"].clip(lower=0, upper=1)

        # Ajout d'une référence (DateLancement = date la plus ancienne dans la fenêtre)
        prix_df["DateLancement"] = earliest_ts

        return prix_df


def main(output_path="data/donnees_synthetiques.csv"):
    """
    Point d'entrée principal : génère les données à partir des paramètres définis,
    et sauvegarde le résultat dans le dossier data/.
    """
    logging.info("Début de la génération de données synthétiques.")

    # Chargement des paramètres par défaut (config.yaml local)
    params = ParametresSynthese()
    logging.info(
        f"Paramètres chargés avec seed={params.seed}, n_periodes={params.n_periodes}"
    )

    # Initialisation du générateur
    generateur = GenerateurDonnees(params)

    # Génération finale du DataFrame
    df_final = generateur.ajuster_prix_et_calculs()

    # Sauvegarde du DataFrame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8")
    logging.info(f"Données synthétiques sauvegardées dans {output_path}")

    logging.info("Génération de données synthétiques terminée avec succès.")


if __name__ == "__main__":
    main()
