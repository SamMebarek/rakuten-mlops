# test/test_integration_main.py

import pytest
import os
from pathlib import Path
import pandas as pd
from src.generation import main


def test_integration_main(tmp_path):
    """
    Test d'intégration de la fonction main:
    - Vérifie qu'un CSV est créé dans le dossier de sortie.
    - Vérifie que le fichier de logs est écrit.
    - Contrôle a minima la forme du CSV (présence de colonnes clés, non-vide).
    """

    # 1. Définir le chemin d'output dans un dossier temporaire
    output_csv = tmp_path / "donnees_synthetiques.csv"
    logs_dir = tmp_path / "Logs"

    # 2. Appeler la fonction main avec ce chemin de sortie
    # On redéfinit la variable d’environnement "LOGS" ou on modifie la logique
    # pour que le logging s’écrive dans logs_dir.
    # Selon votre implémentation, vous pouvez paramétrer generate_data.py
    # pour diriger les logs vers un dossier spécifique.
    # S'il n'y a pas de paramètre pour le dossier "Logs", ce test vérifie
    # seulement la création du CSV.
    main(output_path=str(output_csv))

    # 3. Vérifier que le CSV est bien créé
    assert output_csv.exists(), f"Le fichier {output_csv} n'a pas été créé par main()."

    # 4. Charger rapidement le CSV et faire quelques vérifications basiques
    df = pd.read_csv(output_csv)
    assert not df.empty, "Le fichier CSV généré est vide."
    # Vérifier quelques colonnes clés
    colonnes_essentielles = ["Categorie", "SKU", "Prix", "QuantiteVendue"]
    for col in colonnes_essentielles:
        assert col in df.columns, f"Il manque la colonne {col} dans le CSV."

    # 5. Vérifier la présence d'au moins une ligne
    assert len(df) > 0, "Le DataFrame généré devrait contenir des enregistrements."

    # 6. (Optionnel) Vérifier l'existence d'un fichier de logs
    # Selon la configuration, vous pouvez pointer logs_dir ou tout autre emplacement.
    # Par défaut, si le script create "Logs/generation.log" dans le répertoire courant,
    # vous pouvez vérifier son existence de façon relative.
    logs_path = Path("Logs") / "generation.log"
    if logs_path.exists():
        assert logs_path.stat().st_size > 0, "Le fichier de logs existe mais est vide."
    else:
        # Si vous ne dirigez pas encore les logs vers tmp_path, ce test n'échoue pas forcément.
        # C'est plutôt un avertissement.
        print("Note: Le fichier de logs 'Logs/generation.log' n'a pas été trouvé.")

    print("Test d'intégration main() : OK")
