# kkbox_churn_ML
KKbox churn prediction challenge

**Audit → Nettoyage → Feature Engineering → LightGBM → Soumission Kaggle**

> Prédire si un utilisateur du service de streaming musical KKBox résiliera son abonnement après expiration, à partir de ses transactions, logs d'écoute et données démographiques.

---

## Table des matières

- [Contexte métier](#contexte-métier)
- [Définition du churn](#définition-du-churn)
- [Données](#données)
- [Architecture du pipeline](#architecture-du-pipeline)
- [Pourquoi DuckDB ?](#pourquoi-duckdb-)
- [Installation et exécution](#installation-et-exécution)
- [Feature Engineering](#feature-engineering)
- [Modélisation](#modélisation)
- [Résultats](#résultats)
- [Prédicteurs clés](#prédicteurs-clés)
- [Techniques DuckDB utilisées](#techniques-duckdb-utilisées)
- [Pistes d'amélioration](#pistes-damélioration)

---

## Contexte métier

[KKBox](https://www.kkbox.com/) est un service de streaming musical par abonnement (comparable à Spotify ou Deezer). Les utilisateurs souscrivent un plan — le plus souvent de **30 jours** — qu'ils peuvent renouveler manuellement ou automatiquement, ou annuler à tout moment.

L'objectif de cette [compétition Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) est de prédire si un utilisateur **résiliera** (*churn*) après l'expiration de son abonnement, c'est-à-dire s'il ne souscrira pas un nouvel abonnement dans les **30 jours** suivant la date d'expiration.

## Définition du churn

> *« Subscription cancellation does not imply the user has churned. »*

Le champ `is_cancel` indique qu'un utilisateur a **activement annulé** son abonnement. Mais cela ne signifie **pas** qu'il a résilié : un utilisateur peut annuler puis se réabonner dans les 30 jours (par exemple pour changer de plan). Le **vrai critère de churn** est l'absence de toute nouvelle transaction valide dans les 30 jours suivant l'expiration.

C'est pourquoi `is_cancel` est utilisé comme **feature prédictive** (et non comme proxy direct du label). L'EDA valide que ce signal est légitime et ne constitue pas une fuite de données.

## Données

### Structure temporelle

| Ensemble | Utilisateurs concernés | Période de prédiction |
|----------|----------------------|----------------------|
| **Train** (v1 + v2) | Abonnements expirant en **février** et **mars** 2017 | Churn en ~mars et ~avril 2017 |
| **Test** (v2) | Abonnements expirant en **mars** 2017 | Churn en ~avril 2017 |

Les fichiers `v2` **complètent** les fichiers `v1` — le pipeline fusionne systématiquement v1 + v2.

### Les 5 tables

| Table | Contenu | Volume |
|-------|---------|--------|
| `train` / `train_v2` | Label de churn par utilisateur (`msno`, `is_churn`) | ~2 M lignes |
| `sample_submission_v2` | Identifiants du jeu de test | ~900 K lignes |
| `members_v3` | Profil utilisateur : ville, âge, genre, canal d'inscription | ~6 M lignes |
| `transactions` / `transactions_v2` | Historique complet des transactions | ~22 M lignes |
| `user_logs` / `user_logs_v2` | Logs d'écoute quotidiens | **~400 M lignes / ~30 Go** |

## Architecture du pipeline

```
Fichiers CSV  ──► DuckDB (conversion Parquet, une seule fois)  ──► Parquet sur disque
              ──► DuckDB SQL (audit + nettoyage + features)    ──► DataFrames Pandas
              ──► LightGBM (validation croisée 5-folds)        ──► submission.csv
```

### Étapes détaillées

| # | Étape | Description |
|---|-------|-------------|
| 0 | Configuration | Connexion DuckDB en mémoire, limites RAM (6.4 Go), spill sur disque |
| 1 | CSV → Parquet | Conversion one-shot avec streaming par row groups (100K lignes) |
| 2 | Audit | Comptages, schémas, nulls — Parquet footer pour les metadata, sample 1 % pour `user_logs` |
| 3 | Nettoyage | Vues SQL : dédup `ROW_NUMBER()`, nettoyage âge/genre, parsing dates YYYYMMDD |
| 4 | Feature Engineering | Features transactionnelles (snapshot + agrégats) + features d'écoute, cache Parquet |
| 5 | EDA | Distribution du churn, patterns transactionnels, écoute, démographie, corrélations |
| 6 | Entraînement | LightGBM 5-fold stratifié, early stopping (patience = 50) |
| 7 | Évaluation | ROC, Précision-Rappel, matrice de confusion, importance des features |
| 8 | Soumission | Moyenne des probabilités sur 5 folds → `submission.csv` |

## Pourquoi DuckDB ?

Les 30 Go de `user_logs` rendent Pandas inutilisable. DuckDB est un moteur SQL **analytique embarqué** qui résout ce problème :

- **Zéro infrastructure** — `pip install duckdb`, pas de serveur externe
- **Lecture Parquet native** — projection pushdown (seules les colonnes utiles sont lues) + predicate pushdown (les blocs filtrés sont sautés)
- **Débordement automatique sur disque** — `user_logs` n'est jamais chargé intégralement en mémoire
- **SQL standard** — requêtes lisibles et débuggables
- **Pont Arrow zero-copy** — conversion rapide vers Pandas via `.df()`

## Installation et exécution

### Prérequis

```bash
pip install duckdb lightgbm scikit-learn pandas numpy matplotlib seaborn
```

### Données

Télécharger les fichiers depuis la [page de la compétition Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) et les placer dans un répertoire `DATA_DIR`. Fichiers attendus :

```
DATA_DIR/
├── train.csv
├── train_v2.csv
├── members_v3.csv
├── transactions.csv
├── transactions_v2.csv
├── user_logs.csv
├── user_logs_v2.csv
└── sample_submission_v2.csv
```

### Exécution

Ouvrir le notebook et exécuter toutes les cellules. La première exécution convertit les CSV en Parquet (quelques minutes pour `user_logs`). Les exécutions suivantes sont quasi-instantanées grâce au cache Parquet.

> **Configuration mémoire** : le pipeline est calibré pour **8 Go de RAM** (`memory_limit = 6.4 Go`). Ajuster `SET memory_limit` et `SET threads` selon votre machine.

## Feature Engineering

### Features transactionnelles (§4.1)

Deux sous-ensembles combinés par utilisateur :

**Snapshot (dernière transaction)** — `last_payment_method`, `last_plan_days`, `last_amount_paid`, `last_auto_renew`, `last_is_cancel`, `days_to_expire`, `last_discount_pct`, etc.

**Agrégats historiques** — `txn_count`, `total_paid`, `avg_paid`, `std_paid`, `cancel_pct`, `cancel_count`, `n_payment_methods`, `n_unique_plans`, `price_range`, etc.

### Features d'écoute (§4.2)

Agrégation streaming de ~30 Go :

- **Volume** : `total_logs`, `total_secs_sum/max/min/mean/std`
- **Décomptes** : `num_25_sum`, `num_50_sum`, `num_75_sum`, `num_985_sum`, `num_100_sum`, `num_unq_sum`
- **Ratios d'engagement** : `completion_rate`, `skip_rate`, `diversity_ratio`, `avg_secs_per_song`, `daily_plays`
- **Portée temporelle** : `active_days_span`

### Features démographiques (§3.1)

`city`, `bd` (âge nettoyé), `gender_enc`, `registered_via`, `membership_duration_days`

## Modélisation

### Hyperparamètres LightGBM

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `learning_rate` | 0.05 | Taux bas — early stopping trouve le nombre optimal d'arbres |
| `num_leaves` | 63 | Équilibre expressivité / sur-apprentissage |
| `max_depth` | 7 | Régularisateur secondaire |
| `min_child_samples` | 50 | Empêche le sur-découpage |
| `subsample` | 0.8 | Sous-échantillonnage de lignes |
| `colsample_bytree` | 0.8 | Sous-échantillonnage de features |
| `n_estimators` | 1500 | Borne supérieure, early stopping (patience = 50) |

### Validation

`StratifiedKFold` à 5 folds préserve le taux de churn (~7 %) dans chaque fold. Les prédictions OOF (Out-Of-Fold) fournissent une estimation non biaisée de la généralisation.

## Résultats

- **OOF AUC** : ~0.96+
- **OOF LogLoss** : ~0.10
- **Variance inter-folds** : faible (modèle stable)

Le taux de churn est d'environ **7 %**, ce qui rend les métriques AUC et Précision-Rappel plus informatives que l'accuracy brute.

## Prédicteurs clés

1. **`last_is_cancel`** — annulation explicite, signal le plus fort
2. **`last_auto_renew`** — risque de churn passif quand désactivé
3. **`days_to_expire`** — proximité de l'expiration
4. **`completion_rate`** — qualité de l'engagement musical
5. **`membership_duration_days`** — signal de fidélité

## Techniques DuckDB utilisées

| Technique | Rôle dans le pipeline |
|-----------|----------------------|
| Inférence automatique du schéma Parquet | Pas de définition manuelle |
| Parallélisme automatique multi-cœurs | Pas de tuning de partitions |
| Écriture Parquet + relecture | Cache simple et persistant |
| `QUALIFY ROW_NUMBER() OVER (...)` | Déduplication propre |
| `COALESCE(x, 0.0)` | Remplacement de `NULL` par défaut |
| `USING SAMPLE 1 PERCENT` | Échantillonnage intégré |
| `CREATE VIEW AS SELECT ... CASE WHEN` | Nettoyage déclaratif |
| `.df()` via pont Arrow zero-copy | Conversion rapide vers Pandas |
| `NULLIF(x, 0)` | Division sûre |

## Pistes d'amélioration

- **Optuna** — recherche d'hyperparamètres (`num_leaves`, `learning_rate`, `min_child_samples`)
- **Valeurs SHAP** — interprétabilité par utilisateur via `shap.TreeExplainer`
- **Features à décroissance temporelle** — pondérer les logs récents (ex. `completion_rate` des 30 derniers jours)
- **Target encoding** — pour `city` et `registered_via`, à l'intérieur de chaque fold CV
- **Stacking** — combiner LightGBM avec XGBoost / CatBoost

---
