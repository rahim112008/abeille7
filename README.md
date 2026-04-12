# 🐝 ApiTrack Pro — Application de gestion apicole professionnelle

## Description
ApiTrack Pro est une application web complète pour apiculteurs professionnels, développée avec **Streamlit**, **Python** et **SQLite**. Elle intègre la gestion des ruches, les analyses morphométriques IA (classification raciale Ruttner 1988), la cartographie satellite et un tableau de bord analytique complet.

---

## Installation rapide

```bash
# 1. Cloner ou copier les fichiers dans un dossier
cd apitrack_pro/

# 2. Créer un environnement virtuel (recommandé)
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur http://localhost:8501

**Identifiants démo :** `admin` / `admin1234`

---

## Fonctionnalités

| Module | Description |
|--------|-------------|
| 🏠 Dashboard | Métriques temps réel, graphiques production, carte rucher, alertes varroa |
| 🐝 Ruches | CRUD complet, tableau comparatif, suppression irréversible avec cascade |
| 🔍 Inspections | Enregistrement poids/cadres/varroa/reine/comportement, graphique évolution varroa |
| 💊 Traitements | Suivi vétérinaire avec barre de progression, historique |
| 🍯 Productions | Miel / Pollen / Gelée royale — indicateurs qualité (humidité, pH, 10-HDA) |
| 🧬 Morphométrie IA | Saisie mesures → classification Ruttner 1988 avec % confiance + export JSON |
| 🗺️ Cartographie | Carte satellite Folium, ajout zones mellifères, marqueurs colorés |
| ☀️ Météo & Miellée | Prévisions 7 jours, indice de butinage, conseils saisonniers |
| 📊 Génétique | Registre reines, score VSH, top 3 candidates élevage, radar chart |
| 🌿 Flore | Tableau espèces mellifères, calendrier nectar/pollen interactif |
| ⚠️ Alertes | Détection varroa critique (>3%) / attention (>2%), VSH faible, top GR |
| 📋 Journal | Traçabilité complète de toutes les actions, export CSV |
| ⚙️ Admin | Profil rucher, changement mot de passe, sauvegarde SQLite, statistiques |

---

## Structure des fichiers

```
apitrack_pro/
├── app.py              # Application principale (tout-en-un)
├── requirements.txt    # Dépendances Python
├── runtime.txt         # Version Python (Streamlit Cloud)
├── README.md           # Ce fichier
└── apitrack.db         # Base SQLite (créée au premier lancement)
```

---

## Base de données SQLite

Tables créées automatiquement :

| Table | Description |
|-------|-------------|
| `users` | Utilisateurs avec mots de passe hashés SHA256 |
| `ruches` | Ruches avec coordonnées GPS |
| `inspections` | Inspections (poids, cadres, varroa, reine, comportement) |
| `traitements` | Traitements vétérinaires |
| `recoltes` | Récoltes (miel, pollen, gelée royale) avec contrôle qualité |
| `morph_analyses` | Analyses morphométriques avec JSON de classification |
| `zones` | Zones mellifères géoréférencées avec NDVI |
| `journal` | Journal de toutes les actions |
| `settings` | Paramètres de l'application |

---

## Classification raciale (Ruttner 1988)

L'algorithme de classification utilise trois mesures principales :

- **Longueur de l'aile antérieure** (mm)
- **Indice cubital** = rapport distances nervures cubitales
- **Longueur de la glossa** (mm)

Races reconnues :
- *Apis mellifera intermissa* (Afrique du Nord)
- *Apis mellifera sahariensis* (Sahara)
- *Apis mellifera ligustica* (Italienne)
- *Apis mellifera carnica* (Carniolienne)
- *Hybride*

Le résultat est exportable en JSON selon le format défini dans le prompt ApiTrack Pro.

---

## Modules optionnels

### TensorFlow (deep learning)
```bash
pip install tensorflow  # Python ≤ 3.12 requis
```
Active la segmentation des zones mellifères via modèle U-Net (`melliferous_model.h5`).

### SentinelHub (NDVI satellite)
```bash
pip install sentinelhub
```
Configurez vos identifiants dans `⚙️ Administration → Sentinel Hub`.

---

## Déploiement Streamlit Cloud

1. Pushez les fichiers sur GitHub
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Sélectionnez le repo et `app.py`
4. Le `runtime.txt` force Python 3.12 (compatibilité TensorFlow)

---

## Sécurité

- Mots de passe hashés **SHA256**
- Session utilisateur via `st.session_state`
- Déconnexion dans la barre latérale
- Changement de mot de passe possible depuis l'administration

---

## Crédits

**ApiTrack Pro v2.0** — Développé selon les spécifications du prompt ApiTrack Pro intégrant l'analyse morphométrique IA (Gemini/GLK) et la classification raciale Ruttner (1988).

```
🐝 Bon miellée !
```
