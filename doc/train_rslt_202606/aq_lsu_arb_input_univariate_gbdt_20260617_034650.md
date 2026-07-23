# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_arb/aq_lsu_arb_input_univariate_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_arb_input_univariate_gbdt_20260617_034650`
- Algorithm: **GBDT**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 86 |
| after_preprocess | 5461 | 1365 | 34205 | 72 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 31.014 | 76.586 | 0.00000 | 0.00000 | 0.9963 |
| val | 38.960 | 92.988 | 0.00000 | 0.00000 | 0.9797 |
| test | 54.049 | 102.649 | 0.00001 | 0.00001 | 0.5445 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1138` |
| `max_depth` | `5` |
| `learning_rate` | `0.05151489889286242` |
| `subsample` | `0.7679033786651158` |
| `colsample_bytree` | `0.8761581748026752` |
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
