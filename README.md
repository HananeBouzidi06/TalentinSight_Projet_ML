# TalentInsight Analytics

**Prédiction du Salaire & Détection de l'Attrition des Employés** — Human Resources Analytics · Machine Learning
Projet du module Machine Learning (formateur : SALHI Abderrahman).

Application web Flask qui transforme le projet ML en dashboard interactif : exploration des données, preprocessing, régression du salaire, classification de l'attrition et simulation RH.

## Objectifs
- Prédire le salaire mensuel (MAD) d'un employé (régression linéaire).
- Détecter les employés à risque d'attrition (SVM, KNN, Decision Tree, Random Forest), avec focus sur le **Recall de la classe 1**.
- Analyser les données RH pour éclairer les décisions.

## Technologies
Python · Flask · HTML5/CSS3/JavaScript · Pandas · NumPy · Scikit-learn · SciPy · Matplotlib/Seaborn · Joblib
(pas de React, Node.js, Docker ni base de données).

## Fonctionnalités
- **Dashboard** : KPI (employés, variables, salaire moyen, taux d'attrition…).
- **EDA** : valeurs manquantes, doublons, distributions, corrélations, attrition par département / satisfaction.
- **Preprocessing** : doublons, médiane, aberrants, Label Encoding, `ratio_satisfaction`, StandardScaler, avant/après.
- **Régression** : MAE, MSE, RMSE, R², CV 5 folds, réel vs prédit, résidus, formulaire d'estimation.
- **Classification** : 4 modèles, rapports, matrices de confusion, CV `f1_macro`, faux négatifs/positifs, feature importances.
- **Simulation RH** : salaire estimé, prédiction et probabilité d'attrition, explication.

## Méthodologie
- CSV original jamais modifié ; échantillon stratifié de 200 employés max (`random_state=42`).
- Split 80/20 stratifié **avant** d'apprendre le preprocessing (pas de data leakage) ; validation croisée avec preprocessing refait dans chaque fold.
- Aucun GridSearchCV / RandomizedSearchCV. Tous les chiffres affichés sont calculés (`models/results.json`).
- Les modèles sont entraînés une fois puis rechargés (Joblib) ; Flask ne recalcule rien à la navigation.

## Installation (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancement
```powershell
python ml_pipeline.py   # optionnel : réentraîne et régénère modèles, graphiques et résultats
python app.py           # http://127.0.0.1:5002
```
Si `models/` est vide, `app.py` lance l'entraînement automatiquement au premier démarrage.

## Structure
```
TalentInsight-Analytics/
├── app.py                # routes Flask + API de prédiction (port 5002)
├── ml_pipeline.py        # preprocessing, entraînement, évaluation, graphiques
├── requirements.txt
├── data/rh_employes.csv
├── models/               # regression_model.pkl, classification_model.pkl, preprocessing.pkl, results.json
├── templates/            # base, index, eda, preprocessing, regression, classification, simulation
└── static/               # css/style.css, js/main.js, images/ (graphiques générés)
```

## Avertissement
Les prédictions sont **expérimentales** (petit échantillon) et ne doivent pas être utilisées seules pour prendre des décisions concernant des employés.
## Lien du Projet :
https://talentinsightprojetml.vercel.app/
