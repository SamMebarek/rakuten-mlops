# **Rakuten MLOps - Dynamic Pricing**




Ce projet implémente une solution de **pricing dynamique** basée sur le Machine Learning, permettant d'optimiser la tarification des produits en fonction de divers facteurs influençant la demande.

Il repose sur un pipeline automatisé intégrant **l'ingestion des données de vente**, **le prétraitement avancé des données**, **l'entraînement d'un modèle XGBoost pour la prédiction des prix optimaux**, et **le déploiement via une API REST FastAPI** permettant d'obtenir des recommandations tarifaires en temps réel.

---------

## **Pipeline de traitement**

Le projet est structuré en plusieurs étapes automatisées :

1. **Ingestion des données** : Chargement et validation des données brutes.
2. **Prétraitement** : Nettoyage, transformations et génération des features.
3. **Entraînement du modèle** : Optimisation des hyperparamètres et tracking avec MLflow.
4. **Déploiement via FastAPI** : Exposition du modèle via une API REST.

---

## **Installation et Configuration**

### **Prérequis**

- Python 3.8+ recommandé

- [MLflow](https://mlflow.org/) (optionnel mais recommandé pour le suivi des expériences)

### **Installation**

#### **1. Cloner le repository**

```bash
git clone https://github.com/SamMebarek/rakuten-mlops.git
cd rakuten-mlops
```

#### **2. Installer les dépendances**

```bash
pip install -r requirements.txt
```

#### **3. Lancer le serveur MLflow** (optionnel)

MLflow est utilisé pour suivre les expériences d'entraînement du modèle. Il peut être lancé localement avec :

```bash
mlflow server --host 127.0.0.1 --port 8080
```

L’interface MLflow est accessible sur **[http://127.0.0.1:8080](http://127.0.0.1:8080)**.

#### **4. Lancer l'API**

```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload
```

L'API est maintenant accessible sur **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## **Utilisation de l'API**

L'API permet d'effectuer des prédictions de prix à partir d'un **SKU** donné.

### **Endpoints**

|Méthode|Endpoint|Description|
|---|---|---|
|`GET`|`/health`|Vérifier que l'API est en ligne|
|`GET`|`/status`|Vérifier l'état du modèle et des données|
|`POST`|`/predict`|Faire une prédiction sur un SKU donné|
|`POST`|`/reload-model`|Recharger le dernier modèle sauvegardé|

---

### **Exemple de requête `POST /predict`**

#### **Requête**

```bash
curl -X 'POST' 'http://localhost:8000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"sku": "SKU1_1"}'
```

#### **Réponse**

```json
{
  "sku": "SKU1_1",
  "timestamp": "2025-02-17 14:30:00",
  "predicted_price": 19.99
}
```