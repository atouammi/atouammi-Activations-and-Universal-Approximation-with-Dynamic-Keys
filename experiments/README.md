# Benchmark des activations CSwish / KGELU

Compare **CSwish** et **KGELU** (définies dans `cswish_activation/cswish.py`,
cf. Tiebekabe, *"Cryptographically-Inspired Transformer Architectures"*,
ICMAI 2026) à des activations classiques — ReLU, Sigmoid, Tanh, GELU, SiLU —
sur plusieurs tâches et datasets.

## Structure

| Fichier | Contenu |
|---|---|
| `activations.py` | Registre des activations (`make_activation`), avec support du `key_seed` pour CSwish/KGELU |
| `datasets.py` | Loaders tabulaires (`digits`, `wine`, `breast_cancer` via scikit-learn) et vision (`mnist`, `fashion_mnist` via `fetch_openml`) |
| `models.py` | `MLP` (tabulaire) et `SmallCNN` (vision) |
| `run_experiment.py` | Script d'entraînement/benchmark en CLI |
| `results/benchmark.csv` | Résultats bruts, une ligne par run |

**Note environnement** : les loaders vision utilisent `sklearn.fetch_openml`
plutôt que `torchvision.datasets`, car cette machine a un conflit de version
`torch`/`torchvision` (`torchvision::nms does not exist`). Si cet environnement
est corrigé un jour, `datasets.py::load_vision` pourrait repasser par
torchvision, mais ce n'est pas nécessaire pour ce benchmark.

## Lancer un run

```bash
cd experiments

# Un dataset tabulaire, toutes les activations du jeu "tabular", 3 seeds
python3 run_experiment.py --task tabular --dataset digits --seeds 0 1 2

# Un dataset vision
python3 run_experiment.py --task vision --dataset mnist --seeds 0 1 2 --epochs 8

# Restreindre aux activations qui nous intéressent
python3 run_experiment.py --task tabular --dataset wine --activations relu cswish kgelu --seeds 0 1 2

# Sonder la sensibilité à la clé (alpha, theta) de CSwish/KGELU à graine réseau fixée
python3 run_experiment.py --task vision --dataset mnist --activations cswish kgelu --seeds 0 --key-seeds 10 11 12
```

Chaque run ajoute des lignes à `results/benchmark.csv` (ou au fichier passé via `--out`).

### Options principales

| Flag | Défaut | Rôle |
|---|---|---|
| `--task` | — (requis) | `tabular` ou `vision` |
| `--dataset` | — (requis) | `digits`, `wine`, `breast_cancer` / `mnist`, `fashion_mnist` |
| `--activations` | jeu par défaut de la tâche (voir `TASK_ACTIVATIONS` dans `activations.py`) | sous-ensemble d'activations à tester |
| `--seeds` | `0 1 2` | graines d'initialisation réseau |
| `--key-seeds` | `10 11 12` | graines de clé `(alpha, theta)`, utilisées seulement pour `cswish`/`kgelu` — un run par combinaison `(seed, key_seed)` |
| `--epochs` | `15` | nombre d'époques |
| `--lr` | `1e-3` | taux d'apprentissage (Adam) |
| `--batch-size` | `64` | taille de batch |
| `--hidden` | `64` | largeur cachée du MLP (tabulaire uniquement) |
| `--out` | `results/benchmark.csv` | fichier CSV de sortie (append) |

Chaque ligne du CSV contient : `best_val_acc`, `final_val_acc`,
`final_train_loss`, `final_val_loss`, `epochs_to_90pct_of_best` (vitesse de
convergence), `noise_val_acc` (accuracy sous bruit gaussien σ=0.1, en plus du
diagnostic de robustesse), et `wall_clock_s`.

### Reproduire la batterie complète déjà exécutée

```bash
cd experiments
python3 run_experiment.py --task tabular --dataset digits        --seeds 0 1 2 --epochs 20
python3 run_experiment.py --task tabular --dataset wine           --seeds 0 1 2 --epochs 20
python3 run_experiment.py --task tabular --dataset breast_cancer  --seeds 0 1 2 --epochs 20
python3 run_experiment.py --task vision  --dataset mnist          --seeds 0 1 2 --epochs 8
python3 run_experiment.py --task vision  --dataset fashion_mnist  --seeds 0 1 2 --epochs 8
```

## Premiers résultats

159 runs : 5 datasets × 7 activations (Sigmoid exclu en vision, cf. notes de
stratégie dans `activations.py`) × 3 seeds réseau, et pour CSwish/KGELU × 3
seeds de clé par seed réseau.

**Accuracy de validation (meilleure époque, moyenne ± écart-type, %)**

| Activation | Digits | Wine | Breast Cancer | MNIST | FashionMNIST |
|---|---|---|---|---|---|
| ReLU | 96.96 ± 0.68 | 97.04 ± 2.57 | 96.27 ± 1.07 | 97.97 ± 0.24 | 88.99 ± 0.85 |
| Sigmoid | 92.22 ± 0.97 | 71.11 ± 5.88 | 96.27 ± 0.40 | n/a | n/a |
| Tanh | 96.74 ± 0.46 | 98.52 ± 1.28 | 97.20 ± 0.00 | 98.14 ± 0.37 | 89.07 ± 0.67 |
| GELU | 96.89 ± 0.59 | 96.30 ± 2.57 | 96.97 ± 0.40 | 97.96 ± 0.25 | 89.19 ± 0.99 |
| SiLU | 96.67 ± 0.39 | 97.04 ± 1.28 | 96.97 ± 0.40 | 97.97 ± 0.25 | 88.76 ± 0.68 |
| **CSwish** | 96.72 ± 0.38 | 96.30 ± 2.22 | 96.97 ± 0.35 | 97.95 ± 0.27 | 88.79 ± 0.64 |
| **KGELU** | 96.81 ± 0.48 | 96.05 ± 2.67 | 96.97 ± 0.35 | 97.98 ± 0.23 | 89.19 ± 0.81 |

Tableau de bord interactif (courbes, robustesse au bruit, analyse de la
variance due à la rotation de clé) :
https://claude.ai/code/artifact/28ce2f45-84f7-4efc-9cee-f80f40e6a2d1

### Lecture

1. **CSwish/KGELU ≈ GELU/SiLU partout**, jamais en dessous de ReLU — cohérent
   avec KGELU → GELU quand k=0, β=1, et avec la contrainte de stabilité
   (Eq. 6 du papier) qui borde fortement la perturbation apportée par le
   terme de clé.
2. **La rotation de clé coûte quasiment rien.** À graine réseau fixée,
   l'écart-type dû au seul changement de `(alpha, theta)` est 5 à 50× plus
   petit que l'écart-type dû à la ré-initialisation du réseau (ex. Wine :
   ~0.004-0.006 vs ~0.026-0.030). C'est la validation empirique la plus nette
   de ce premier passage.
3. **Sigmoid décroche sur Wine** (réseau peu profond, peu de données) —
   comportement attendu, sert de repère de calibration du protocole plutôt
   que d'enseignement sur CSwish/KGELU.
4. **Pas d'avantage de robustesse au bruit détecté** pour l'instant (σ=0.1,
   k=3, β=1) : CSwish/KGELU se comportent comme GELU/SiLU, ni mieux ni pire.
5. **Portée limitée** : petits modèles (MLP 2 couches, CNN 2 convolutions),
   peu d'époques, un seul `(k, β)`. La tâche Transformer/séquence — le vrai
   terrain du papier — reste à tester.
