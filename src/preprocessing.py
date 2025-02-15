# preprocessing.py

import logging
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocessing")


# -------------------------------------------------------------------
# 1) FunctionTransformer #1 : initial_cleaning
# -------------------------------------------------------------------
def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Conversions de types (SKU->string, Promotion->bool->int, etc.)
    - Clipping sur ElasticitePrix
    - NE PAS supprimer 'Prix'
    - Supprime en 1 bloc:
      [DateLancement, PrixPlancher, PlancherPourcentage, ErreurAleatoire, ProbabiliteAchat, Date]
      (si existantes)
    """
    df = df.copy()

    # Conversions
    if "Categorie" in df.columns:
        df["Categorie"] = df["Categorie"].astype("category")
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].astype("string")
    for col in [
        "PrixInitial",
        "Prix",
        "Remise",
        "Qualite",
        "UtiliteProduit",
        "ElasticitePrix",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    for col in ["AgeProduitEnJours", "QuantiteVendue"]:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "Promotion" in df.columns:
        df["Promotion"] = df["Promotion"].astype(bool).astype(int)

    # Clipping ElasticitePrix
    if "ElasticitePrix" in df.columns:
        q_low, q_high = df["ElasticitePrix"].quantile([0.01, 0.99])
        df["ElasticitePrix"] = np.clip(df["ElasticitePrix"], q_low, q_high)

    # Suppression en une seule étape
    cols_to_drop = [
        "DateLancement",
        "PrixPlancher",
        "PlancherPourcentage",
        "ErreurAleatoire",
        "ProbabiliteAchat",
        "Date",
    ]
    for c in cols_to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    logger.info(
        "[initial_cleaning] shape final=%s, columns=%s", df.shape, df.columns.tolist()
    )
    return df


cleaning_transformer = FunctionTransformer(initial_cleaning, validate=False)


# -------------------------------------------------------------------
# 2) FunctionTransformer #2 : time_features
# -------------------------------------------------------------------
def time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait Annee, Mois, Jour, Heure depuis Timestamp,
    puis crée d'éventuelles variables cycliques.
    """
    df = df.copy()

    if "Timestamp" in df.columns:
        df["Annee"] = df["Timestamp"].dt.year
        df["Mois"] = df["Timestamp"].dt.month
        df["Jour"] = df["Timestamp"].dt.day
        df["Heure"] = df["Timestamp"].dt.hour

        # Optionnel : variables cycliques
        df["Mois_sin"] = np.sin(2 * np.pi * df["Mois"] / 12)
        df["Mois_cos"] = np.cos(2 * np.pi * df["Mois"] / 12)
        df["Heure_sin"] = np.sin(2 * np.pi * df["Heure"] / 24)
        df["Heure_cos"] = np.cos(2 * np.pi * df["Heure"] / 24)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


time_transformer = FunctionTransformer(time_features, validate=False)


# -------------------------------------------------------------------
# 3) ColumnTransformer => OneHot sur 'Categorie'
# -------------------------------------------------------------------
# remainder='passthrough' => on laisse les autres variables
# drop=None => on encode toutes les modalités
from sklearn.compose import ColumnTransformer

cat_col = ["Categorie"]
onehot = OneHotEncoder(drop=None)
col_trans = ColumnTransformer(
    transformers=[
        ("cat", onehot, cat_col),
    ],
    remainder="passthrough",
)


# -------------------------------------------------------------------
# 4) Pipeline final
# -------------------------------------------------------------------
from sklearn.pipeline import Pipeline

preprocessing_pipeline = Pipeline(
    [
        ("cleaning", cleaning_transformer),  # Conversions de types, suppression groupée
        ("time_features", time_transformer),  # Extraction Annee,Mois,Heure...
        ("onehot", col_trans),  # OneHot sur 'Categorie'
    ]
)


# -------------------------------------------------------------------
# 5) run_preprocessing
# -------------------------------------------------------------------
def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le pipeline et reconstitue un DataFrame final (incluant 'Prix').
    """
    logger.info("[run_preprocessing] Lancement du pipeline scikit-learn.")

    # Fit_transform pour obtenir un np.array
    array_final = preprocessing_pipeline.fit_transform(df)

    # Récupérer la liste finale de colonnes
    # => partie OneHot + remainder
    ct = preprocessing_pipeline["onehot"]
    feature_names = []
    for name, trans, cols in ct.transformers_:
        if name == "cat":
            # OneHot
            oh_enc = trans
            if hasattr(oh_enc, "get_feature_names_out"):
                oh_cols = oh_enc.get_feature_names_out(cols)
                feature_names.extend(oh_cols)
        elif name == "remainder":
            feature_names.extend(cols)

    # Construire le DataFrame final
    df_preprocessed = pd.DataFrame(array_final, columns=feature_names)
    logger.info(
        "[run_preprocessing] shape=%s columns=%s", df_preprocessed.shape, feature_names
    )

    return df_preprocessed


# -------------------------------------------------------------------
# 6) main => lit un CSV, applique run_preprocessing, sauvegarde un CSV final
# -------------------------------------------------------------------
def main():
    from ingestion import ingest_csv

    input_csv = "Data/donnees_synthetiques.csv"
    output_csv = "Data/preprocessed_data.csv"

    df_raw = ingest_csv(input_csv)
    logger.info(
        "[main] df_raw shape=%s, columns=%s", df_raw.shape, df_raw.columns.tolist()
    )

    # On s'assure de ne pas drop la colonne 'Prix'
    if "Prix" not in df_raw.columns:
        logger.warning(
            "Attention, la colonne 'Prix' est introuvable dans %s", input_csv
        )

    df_final = run_preprocessing(df_raw)
    logger.info("[main] df_final shape=%s", df_final.shape)

    # Conserver la colonne 'Prix' si elle n'a pas été transformée (cad si 'Prix' n'est pas dans cat_col, etc.)
    # => Si 'Prix' fait partie du remainder => il est déjà présent dans df_final
    # => Sinon, on peut le recoller manuellement
    if "Prix" not in df_final.columns and "Prix" in df_raw.columns:
        # col_trans a transformé 'Categorie', remainder inclut 'Prix' => normalement, on le trouve
        # si on ne le trouve pas, on recolle manuellement
        df_final["Prix"] = df_raw["Prix"].values
        logger.info("On recolle la colonne 'Prix' manuellement.")

    # Sauvegarde
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    logger.info(
        "[main] CSV final sauvegardé dans %s, shape=%s", output_csv, df_final.shape
    )


if __name__ == "__main__":
    main()
