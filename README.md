# **Dynamic Pricing v1**

Ce projet vise à prédire le prix optimal de produits grâce à un modèle de machine learning. Il intègre une chaîne MLOps utilisant **DVC** pour la gestion des pipelines de données, **MLflow** pour le suivi des expérimentations et l’enregistrement des modèles, et **FastAPI** pour exposer une API REST de prédiction.

---

## **Table des matières**
- [**Dynamic Pricing v1**](#dynamic-pricing-v1)
  - [**Table des matières**](#table-des-matières)
  - [**1. Introduction**](#1-introduction)
  - [**2. Prérequis**](#2-prérequis)
  - [**3. Création d’un compte DagsHub et connexion du projet GitHub**](#3-création-dun-compte-dagshub-et-connexion-du-projet-github)
    - [**Étapes pour connecter votre projet :**](#étapes-pour-connecter-votre-projet-)
    - [**1. Cloner votre projet depuis GitHub**](#1-cloner-votre-projet-depuis-github)
    - [**2. Créer un compte sur DagsHub**](#2-créer-un-compte-sur-dagshub)
    - [**3. Connecter votre repository GitHub à DagsHub**](#3-connecter-votre-repository-github-à-dagshub)
    - [**4. Récupérer les credentials et endpoints**](#4-récupérer-les-credentials-et-endpoints)
  - [**4. Installation**](#4-installation)
  - [**5. Configuration**](#5-configuration)
    - [**Ajouter les credentials DVC**](#ajouter-les-credentials-dvc)
    - [**Fichier `.env`**](#fichier-env)
  - [**6. Utilisation du projet**](#6-utilisation-du-projet)
    - [**1. Génération des données**](#1-génération-des-données)
    - [**2. Exécution du pipeline avec DVC**](#2-exécution-du-pipeline-avec-dvc)
    - [**3. Lancement de l’API FastAPI**](#3-lancement-de-lapi-fastapi)
  - [**7. Endpoints de l’API FastAPI**](#7-endpoints-de-lapi-fastapi)
    - [**GET /health**](#get-health)
    - [**POST /predict**](#post-predict)
      - [**Exemple de requête :**](#exemple-de-requête-)
      - [**Réponse attendue :**](#réponse-attendue-)
    - [**POST /reload-model**](#post-reload-model)
  - [**8. Tests**](#8-tests)
    - [**Pour exécuter les tests :**](#pour-exécuter-les-tests-)

---

## **1. Introduction**
Le projet **Dynamic Pricing v1** permet de générer des données synthétiques, de réaliser l’ingestion, le prétraitement et l’entraînement d’un modèle, puis d’exposer une API REST pour obtenir des prédictions de prix.  
La solution repose sur plusieurs technologies clés :

- **DVC** : Pour versionner et orchestrer le pipeline de données.
- **MLflow** : Pour suivre les expérimentations et enregistrer les modèles.
- **FastAPI** : Pour créer et exposer une API de prédiction.
- **Python-dotenv** : Pour gérer les informations sensibles via un fichier `.env`.

---

## **2. Prérequis**
Assurez-vous d’avoir installé sur votre machine :

- **Python (>=3.8)**
- **Git** (pour cloner le dépôt)
- **DVC** (gestion des pipelines de données)
- **MLflow** (suivi des expérimentations)
- **Uvicorn** (pour lancer l’API FastAPI)
- Les packages listés dans le fichier `requirements.txt`

> **Note :** Git doit être installé globalement. Les autres dépendances seront installées via `pip install -r requirements.txt`.

---

## **3. Création d’un compte DagsHub et connexion du projet GitHub**
Pour exploiter pleinement l'intégration cloud (suivi MLflow, stockage DVC), il est **indispensable** de connecter votre projet à DagsHub. Cette démarche vous permettra de bénéficier d’un serveur MLflow dédié, d’un remote S3 pour DVC et d’outils collaboratifs adaptés aux projets ML.

### **Étapes pour connecter votre projet :**
### **1. Cloner votre projet depuis GitHub**
```bash
git clone https://github.com/SamMebarek/rakuten-mlops.git
cd rakuten-mlops
```

### **2. Créer un compte sur DagsHub**
- Rendez-vous sur [DagsHub](https://dagshub.com/) et cliquez sur **"Sign Up"**.
- Complétez le formulaire d’inscription ou associez votre compte GitHub.

### **3. Connecter votre repository GitHub à DagsHub**
- Sur DagsHub, cliquez sur **"Create +"** en haut à droite, puis sélectionnez **"+ New Repository"**.
- Choisissez **"Import Repository"** et cliquez sur **"GitHub Connect"** pour autoriser DagsHub à accéder à vos repositories GitHub.
- Sélectionnez le repository que vous souhaitez connecter (`rakuten-mlops`), puis cliquez sur **"Connect Repository"**.
- DagsHub créera automatiquement un serveur MLflow et un espace de stockage S3 pour DVC.

### **4. Récupérer les credentials et endpoints**
- Dans l’interface de votre repository DagsHub, récupérez :
  - L’URL du serveur MLflow (`https://dagshub.com/your_username/rakuten-mlops.mlflow`)
  - L’endpoint S3 pour DVC (`https://dagshub.com/your_username/rakuten-mlops.s3`).

---

## **4. Installation**
1. **Créer et activer un environnement virtuel**


2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## **5. Configuration**
### **Ajouter les credentials DVC**
Après avoir configuré votre repository DagsHub, exécutez les commandes suivantes pour ajouter vos credentials DVC :
```bash
dvc remote modify origin --local access_key_id your_token
dvc remote modify origin --local secret_access_key your_token
```

### **Fichier `.env`**
Créez un fichier `.env` à la racine du projet et ajoutez-y vos credentials :
```env
AWS_ACCESS_KEY_ID=your_token
AWS_SECRET_ACCESS_KEY=your_token
MLFLOW_TRACKING_URI=https://your_username:your_token@dagshub.com/your_username/rakuten-mlops.mlflow
```

---

## **6. Utilisation du projet**
### **1. Génération des données**
Exécutez le script suivant pour générer des données synthétiques :
```bash
python src/generation/generation.py
```
Ce script crée le fichier `data/donnees_synthetiques.csv`.

### **2. Exécution du pipeline avec DVC**
```bash
dvc repro
```
Cette commande exécute successivement :
- **Ingestion**
- **Preprocessing**
- **Training** avec MLflow.

### **3. Lancement de l’API FastAPI**
```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload
```
L’API sera accessible sur [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## **7. Endpoints de l’API FastAPI**
### **GET /health**
Vérifie l’état de l’API, du modèle et des données.
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
Réponse attendue :
```json
{
  "status": "OK",
  "model_status": "chargé",
  "data_status": "chargée"
}
```

### **POST /predict**
Prédit le prix pour un SKU donné.
#### **Exemple de requête :**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"sku": "SKU1_1"}'
```
#### **Réponse attendue :**
```json
{
  "sku": "SKU1_1",
  "timestamp": "2025-02-23 15:30:00",
  "predicted_price": 24.99
}
```

### **POST /reload-model**
Recharge le modèle depuis MLflow.
```bash
curl -X POST "http://127.0.0.1:8000/reload-model"
```
Réponse attendue :
```json
{
  "message": "Modèle rechargé avec succès"
}
```

---

## **8. Tests**
Le projet inclut des tests unitaires et d’intégration avec **pytest**.

### **Pour exécuter les tests :**
```bash
pytest -v
```
Les tests vérifient :
- Le bon chargement et la transformation des données.
- Le fonctionnement des endpoints de l’API FastAPI.
- Le pipeline complet (ingestion, preprocessing, training).

