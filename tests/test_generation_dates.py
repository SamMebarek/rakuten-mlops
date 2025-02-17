# test/test_generation_dates.py

import pytest
import pandas as pd
from src.generation import ParametresSynthese, GenerateurDonnees


def test_generer_dates_coherent_inclut_jour_courant(tmp_path):
    """
    Vérifie que la fonction generer_dates() crée un DataFrame de timestamps
    couvrant n_periodes jours avant maintenant PLUS la journée actuelle (jusqu'à 3 timestamps).
    """
    # GIVEN: un fichier de config contrôlé, n_periodes=5
    config_content = """
seed: 42
random_seed_range: 10001

n_skus: 10
n_periodes: 5   # 5 jours

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
    config_file = tmp_path / "test_config_dates.yaml"
    config_file.write_text(config_content)

    # WHEN: on instancie la config et le générateur
    params = ParametresSynthese(config_path=str(config_file))
    gen = GenerateurDonnees(params)
    df_dates = gen.dates_df

    # THEN: on s'attend à (n_periodes + 1) jours couverts (les n_periodes passés + la journée en cours)
    # donc maximum (n_periodes + 1) * 3 timestamps
    n_periodes = params.n_periodes
    nb_ts = len(df_dates)
    max_attendu = (n_periodes + 1) * 3  # 1 jour supplémentaire pour "aujourd'hui"
    min_attendu = (
        n_periodes * 3
    )  # Au minimum, on devrait avoir les n_periodes jours passés complets

    # Vérification : le nb de timestamps doit être entre min_attendu et max_attendu
    # Exemple : pour n_periodes=5, on attend [15, 18] timestamps
    assert (
        min_attendu <= nb_ts <= max_attendu
    ), f"Le nombre de timestamps devrait être entre {min_attendu} et {max_attendu}, trouvé {nb_ts}."

    # Aucune date ne dépasse l'instant présent
    now = pd.Timestamp.now()
    assert (df_dates["Timestamp"] <= now).all(), "Aucun timestamp ne doit être futur."

    # Vérifier la cohérence des intervalles
    ts_sorted = df_dates["Timestamp"].sort_values()
    diffs = ts_sorted.diff().dropna()
    # Chaque saut principal devrait être ~8h, sauf au passage d'un nouveau jour.
    # On fait une vérification large : 4h < gap moyen < 12h (pour tolérer la journée incomplète)
    if len(diffs) > 1:
        moyenne_h = diffs.dt.total_seconds().mean() / 3600
        assert (
            4 <= moyenne_h <= 12
        ), f"Les intervalles semblent trop grands ou trop petits. Moyenne ~{moyenne_h:.1f}h."


@pytest.mark.parametrize("periodes", [0, 1, 10])
def test_generer_dates_cas_limites_inclut_jour_courant(tmp_path, periodes):
    """
    Teste la logique quand n_periodes = 0, 1, ou 10,
    en incluant la journée actuelle dans la génération.
    """
    config_content = f"""
seed: 42
random_seed_range: 10001
n_skus: 5
n_periodes: {periodes}

categories:
  Electronique:
    prix_moyen: 100
    prix_ecart_type: 10
    plancher_pct: 0.4
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
    config_file = tmp_path / "test_config_limites.yaml"
    config_file.write_text(config_content)

    params = ParametresSynthese(config_path=str(config_file))
    gen = GenerateurDonnees(params)
    df_dates = gen.dates_df

    nb_ts = len(df_dates)
    now = pd.Timestamp.now()
    assert (df_dates["Timestamp"] <= now).all(), "Aucun timestamp ne doit être futur."

    # Pour n_periodes = N, on attend [N*3, (N+1)*3] timestamps
    # car on inclut la journée courante
    min_attendu = periodes * 3
    max_attendu = (periodes + 1) * 3

    assert min_attendu <= nb_ts <= max_attendu, (
        f"Trop ou pas assez de timestamps pour n_periodes={periodes} : "
        f"attendus entre {min_attendu} et {max_attendu}, trouvé {nb_ts}."
    )

    # Cas particulier: n_periodes=0 -> on peut avoir de 0 à 3 timestamps
    # Le test précédent déjà gère min=0, max=3 => c'est OK.
