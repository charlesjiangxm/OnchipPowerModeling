# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amr/aq_lsu_amr_internal_deep_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amr_internal_deep_gbdt_20260616_114010`
- Algorithm: **GBDT**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 36 |
| after_preprocess | 5461 | 1365 | 34205 | 33 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.130 | 0.130 | 0.00000 | 0.00000 | 0.9990 |
| val | 0.364 | 0.365 | 0.00000 | 0.00000 | 0.9870 |
| test | 0.844 | 0.854 | 0.00000 | 0.00000 | 0.7539 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1060` |
| `max_depth` | `7` |
| `learning_rate` | `0.010969899313832441` |
| `subsample` | `0.6391499290124204` |
| `colsample_bytree` | `0.8391207274502442` |
| `tree_method` | `hist` |
| `n_jobs` | `-1` |
| `early_stopping_rounds` | `30` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### interaction_heatmap

![interaction_heatmap](artifacts/interaction_heatmap.png)

### convergence

![convergence](artifacts/convergence.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)
