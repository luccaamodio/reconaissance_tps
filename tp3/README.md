# TP3 - Réseaux de Neurones Convolutifs pour Classification CIFAR10

**Implémentation complète d'une architecture CNN pour la classification d'images avec Keras/TensorFlow**

## 📋 Structure du Projet (version finale)

```
tp3/
├── TP3_CNN.ipynb                         # Notebook interactif du TP (Keras, Jupyter)
├── keras_solution.py                     # Script principal (réplique du notebook du prof)
├── keras_complete_analysis.py            # Script d'analyse avancée (DeepCNN, activations, courbes)
├── generate_complete_rapport.py          # Générateur du rapport scientifique PDF (ReportLab)
├── TP3_RAPPORT_COMPLET_20260324_202426.pdf  # Rapport final remis
├── TP3_ANALYSE_COMPLETE.txt              # Analyse PyTorch détaillée (annexe du rapport)
├── MI204-TPCNN.pdf                       # Énoncé officiel du TP
├── activation_evolution_temporal_keras.png
├── activation_maps_deep_layers_keras.png
├── activation_maps_keras.png
├── activation_maps_simple_keras.png
├── confusion_matrix_deep_keras.png
├── confusion_matrix_deep_normalized_keras.png
├── confusion_matrix_keras.png
├── confusion_matrix_normalized_keras.png
├── confusion_matrix_simple_keras.png
├── confusion_matrix_simple_normalized_keras.png
├── training_curves_comparison_keras.png
├── training_curves_keras.png
├── deepcnn_architecture.png              # Schéma bloc du réseau DeepCNN
├── _archive/                             # Scripts/anciens PDFs et artefacts intermédiaires
└── .venv_keras/                          # Environnement Python 3.12 (optionnel)
```

## ⚙️ Installation et Configuration

### 1. Créer l'environnement Python (première fois seulement)

```bash
cd /path/to/tp3
python3.12 -m venv .venv_keras
source .venv_keras/bin/activate
pip install --upgrade pip
pip install tensorflow keras numpy matplotlib scikit-learn pillow jupyter
```

### 2. Activer l'environnement (à chaque utilisation)

```bash
source .venv_keras/bin/activate
```

## 🚀 Exécution

### Option A: Notebook Interactif (Recommandé pour visualisation)

```bash
source .venv_keras/bin/activate
jupyter notebook TP3_CNN.ipynb
```

Exécutez les cellules dans l'ordre. Les visualisations s'affichent directement.

### Option B: Scripts Python

#### 1. **keras_solution.py** – Implémentation de base du TP
```bash
source .venv_keras/bin/activate
python keras_solution.py
```
**Sortie:** Figures de base (courbes, matrices, activations) et modèle Keras.  
**Durée:** ~3–5 minutes

#### 2. **keras_complete_analysis.py** – Analyse avancée (DeepCNN, évolutions temporelles)
```bash
source .venv_keras/bin/activate
python keras_complete_analysis.py
```
**Sortie:** Toutes les figures détaillées utilisées dans le rapport (DeepCNN, activations profondes, évolution temporelle).

#### 3. **generate_complete_rapport.py** – Génération du PDF final
```bash
source .venv_keras/bin/activate
python generate_complete_rapport.py
```
**Sortie:** Nouveau fichier `TP3_RAPPORT_COMPLET_YYYYMMDD_HHMMSS.pdf` construit à partir des figures PNG et des textes d'analyse.

## 📊 Résultats Attendus

### Performance du modèle:
- **Accuracy entraînement:** ~89%
- **Accuracy validation:** ~41%
- **Accuracy test:** ~40.6%
- **Paramètres entraînables:** 132,010

### Fichiers générés (principaux):
- `TP3_RAPPORT_COMPLET_*.pdf` – versions du rapport scientifique
- `my_model_keras.h5` / `my_model_simple_keras.h5` / `my_model_deep_keras.h5` – modèles sauvegardés (dans `_archive/`)
- Figures PNG listées dans la structure du projet – courbes, matrices de confusion, activation maps, schéma DeepCNN

## 📖 Contenu du TP (7 Parties)

| Partie | Contenus | Fichiers |
|--------|----------|----------|
| **I** | Préliminaires, fonctions utiles | TP3_CNN.ipynb (cellules 1-3) |
| **II.1** | Chargement et dimensionnement CIFAR10 | keras_solution.py, TP3_CNN.ipynb |
| **II.2-3** | Architecture CNN, optimisation | keras_solution.py, `_archive/01_basic_cnn.py` |
| **II.4** | Entraînement | keras_solution.py |
| **III** | Test et évaluation | keras_solution.py |
| **IV** | Cartes d'activation | keras_complete_analysis.py, `_archive/03_activation_visualization.py` |
| **V-VI** | Expériences hyperparamètres | keras_complete_analysis.py, `_archive/02_experiments.py` |

## 🧠 Architectures CNN (SimpleCNN et DeepCNN)

```
Entrée (32×32×3)
    ↓
Conv2D (8 filtres, 3×3)
    ↓
ReLU
    ↓
MaxPool2D (2×2)
    ↓
Flatten (2048 neurones)
    ↓
Dense (64 neurones, ReLU)
    ↓
Dense (10 neurones, Softmax)
    ↓
Sortie (10 classes)

Total SimpleCNN: 132,010 paramètres entraînables

DeepCNN (modèle final utilisé dans le rapport) ajoute deux étages convolutifs et une couche dense élargie (156,074 paramètres), avec Dropout pour la régularisation. Le détail couche par couche et les formules de paramètres sont donnés dans la Section 3 du rapport PDF.
```

## 📝 Analyse Détaillée

Consultez **TP3_ANALYSE_COMPLETE.txt** pour:
- Réponses aux questions théoriques
- Explications détaillées des choix d'architecture
- Résultats quantitatifs des expériences
- Interprétation des cartes d'activation
- Techniques de régularisation et mitigation overfitting

## ⚠️ Dépannage

### Erreur: `No module named 'tensorflow'`
```bash
pip install --upgrade tensorflow
```

### Erreur: `No module named 'jupyter'`
```bash
pip install jupyter
```

### Plotlib ne s'affiche pas
Ajouter en début du script:
```python
import matplotlib
matplotlib.use('TkAgg')  # macOS
```

### Modèle corrupu
Supprimer `my_model_keras.h5` et recréer en réexécutant `keras_solution.py`

## 🔗 Ressources

- **CIFAR10 Dataset:** 60,000 images 32×32, 10 classes
- **Framework:** Keras/TensorFlow 2.x
- **Python:** 3.12+
- **Citations:** Krizhevsky, Hinton (CIFAR10)

## 📌 Notes Importantes

- Les scripts PyTorch (`01_basic_cnn.py`, `02_experiments.py`, `03_activation_visualization.py`) sont des implémentations alternatives en PyTorch pour comparaison
- La version Keras (`TP3_CNN.ipynb`, `keras_solution.py`) est la solution principale du TP
- Les résultats visuels (PNG) sont identiques peu importe la framework
- Le modèle `my_model_keras.h5` peut être rechargé et réutilisé

## ✅ Checklist de Vérification

- [ ] Environnement Python créé (`python3.12 -m venv .venv_keras`)
- [ ] TensorFlow/Keras installés (`pip install tensorflow`)
- [ ] Notebook exécutable ou scripts Python fonctionnels
- [ ] Fichiers PNG générés dans le répertoire courant
- [ ] Modèle H5 créé (~546 KB)
- [ ] Analyse complète consultée (TP3_ANALYSE_COMPLETE.txt)

---

**TP Complété:** Mars 2026  
**Étudiant:** Lucca Amodio et João Victor Ferro  
**Cours:** MI204 - Réseaux de Neurones et Deep Learning  
**Institution:** ENSTA
