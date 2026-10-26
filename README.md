# Prévision de stock et alertes de rupture

Projet de Machine Learning de bout en bout : prévoir les ventes à 7 jours par produit et alerter en cas de risque de rupture de stock.

**Stack** : Python · scikit-learn · XGBoost · SHAP · FastAPI · Docker

> ⚠️ Les données utilisées sont **simulées** (générées par `src/simulate.py`). Le projet vise à démontrer la démarche ML complète, pas à modéliser un commerce réel.

---

## 1. Fonctionnalités

| # | Fonctionnalité | Description |
|---|----------------|-------------|
| 1 | Simulation de données | Ventes quotidiennes de N produits sur 2 ans : tendance, saisonnalité hebdomadaire et annuelle, promotions, bruit, stock et réapprovisionnements |
| 2 | Feature engineering | Lags (1, 7, 14 jours), moyennes et écarts-types glissants, jour de la semaine, promo, stock actuel (uniquement à partir du passé, sans fuite de données) |
| 3 | Prévision des ventes (régression) | Ventes cumulées sur les 7 prochains jours, par produit |
| 4 | Alerte de rupture (classification) | Probabilité qu'un produit tombe en rupture dans les 7 jours |
| 5 | Comparaison de modèles | Baseline, Random Forest, XGBoost |
| 6 | Validation temporelle | `TimeSeriesSplit` (jamais de split aléatoire) |
| 7 | Explicabilité | SHAP : importance globale et explication d'une prédiction individuelle |
| 8 | API REST | FastAPI : `/predict`, `/alerts`, `/health` |
| 9 | Interface web | Tableau des alertes et courbe de prévision par produit |
| 10 | Conteneurisation | Image Docker unique pour tout lancer avec une commande |

---

## 2. Structure du projet

```
stock-forecast/
├── data/                  # données générées (CSV)
├── models/                # modèles entraînés (.joblib)
├── reports/               # métriques, graphiques SHAP
├── src/
│   ├── simulate.py        # génération des données simulées
│   ├── explore.py         # exploration et graphiques
│   ├── features.py        # création des features et des cibles
│   ├── train.py           # entraînement + validation temporelle
│   ├── evaluate.py        # métriques et comparaison des modèles
│   ├── explain.py         # analyses SHAP
│   └── api.py             # API FastAPI
├── web/
│   └── index.html         # interface web
├── tests/                 # tests unitaires (pytest)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 3. Logiciels à installer (Ubuntu)

### 3.1 Mise à jour du système
```bash
sudo apt update && sudo apt upgrade -y
```

### 3.2 Outils de base : Python, pip, venv, Git
```bash
sudo apt install -y python3 python3-pip python3-venv git build-essential libgomp1
python3 --version   # 3.10 ou plus récent
git --version
```
`libgomp1` est nécessaire au fonctionnement d'XGBoost.

### 3.3 Docker
```bash
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER    # puis se déconnecter / reconnecter
docker --version
```

### 3.4 Éditeur de code (au choix)
- **VS Code** : télécharger le `.deb` sur https://code.visualstudio.com puis
  ```bash
  sudo apt install ./code_*.deb
  ```
  Extensions utiles : *Python*, *Pylance*, *Docker*, *Jupyter*.
- ou **PyCharm Community** via `sudo snap install pycharm-community --classic`.

### 3.5 Compte GitHub (pour publier le projet)
Créer un dépôt public et y pousser le code. Il sera visible dans ton CV.

---

## 4. Installation du projet

```bash
git clone <url-du-depot> stock-forecast
cd stock-forecast

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Contenu de `requirements.txt`
```
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
joblib
fastapi
uvicorn[standard]
pydantic
pytest
```

---

## 5. Utilisation

```bash
# 1. Générer les données simulées
python -m src.simulate

# 1b. Explorer les données (graphiques dans reports/figures/)
python -m src.explore

# 2. Construire les features et entraîner les modèles
python -m src.train

# 3. Évaluer et comparer les modèles
python -m src.evaluate

# 4. Générer les explications SHAP
python -m src.explain

# 5. Lancer l'API
uvicorn src.api:app --reload
```

- Documentation interactive de l'API : http://localhost:8000/docs
- Interface web : http://localhost:8000/

### Avec Docker
```bash
docker build -t stock-forecast .
docker run -p 8000:8000 stock-forecast
```

---

## 6. API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Vérifie que l'API fonctionne |
| POST | `/predict` | Entrée : `product_id` + date. Sortie : ventes prévues à 7 jours + probabilité de rupture |
| GET | `/alerts` | Liste des produits dont le risque de rupture dépasse un seuil |

Exemple :
```json
POST /predict
{ "product_id": 12, "date": "2025-06-01" }

→ { "forecast_7d": 143.2, "stockout_probability": 0.78 }
```

---

## 7. Évaluation

- **Régression** : MAE, RMSE, comparaison à la baseline (« même jour la semaine dernière » ou moyenne mobile).
- **Classification** : précision, rappel, F1, PR-AUC (les ruptures sont rares, l'accuracy seule est trompeuse).
- **Validation** : `TimeSeriesSplit` pour respecter l'ordre chronologique.
- **Explicabilité** : SHAP summary plot et explication locale d'une alerte.

---

## 8. Feuille de route (étapes de réalisation)

- [x] **Étape 1** : installation de l'environnement (section 3 et 4)
- [x] **Étape 2** : simulation des données (`simulate.py`)
- [x] **Étape 3** : exploration des données (graphiques, vérification de la saisonnalité) (`explore.py`)
- [ ] **Étape 4** : feature engineering et définition des deux cibles (`features.py`)
- [ ] **Étape 5** : baseline + Random Forest + XGBoost avec `TimeSeriesSplit` (`train.py`)
- [ ] **Étape 6** : évaluation et comparaison (`evaluate.py`)
- [ ] **Étape 7** : explicabilité SHAP (`explain.py`)
- [ ] **Étape 8** : API FastAPI (`api.py`)
- [ ] **Étape 9** : interface web (`web/index.html`)
- [ ] **Étape 10** : tests et Dockerfile
- [ ] **Étape 11** : rédaction des résultats dans ce README et publication sur GitHub

---

## 9. Résultats

*À compléter après l'étape 6 : tableau comparatif des modèles, graphiques SHAP, capture de l'interface.*

---

## 10. Pistes d'amélioration

- Remplacer les données simulées par un vrai jeu de données (ex. Kaggle, M5 Forecasting)
- Ajouter du suivi d'expériences (MLflow)
- Réentraînement automatique et CI/CD (GitHub Actions)
- Prévision probabiliste (intervalles de confiance)

---

## Auteur

Houssam Eddine BAHHOUS, https://github.com/houssam-b
