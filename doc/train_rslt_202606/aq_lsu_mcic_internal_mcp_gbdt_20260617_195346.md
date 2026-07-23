# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_mcic/aq_lsu_mcic_internal_mcp_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_mcic_internal_mcp_gbdt_20260617_195346`
- Algorithm: **GBDT**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 16 |
| after_preprocess | 5461 | 1365 | 34205 | 13 |
| after_feature_selection | 5461 | 1365 | 34205 | 11 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 4.910 | 4.893 | 0.00000 | 0.00000 | 0.8097 |
| val | 5.131 | 5.334 | 0.00000 | 0.00000 | 0.7016 |
| test | 5.962 | 5.759 | 0.00000 | 0.00000 | 0.5147 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `865` |
| `max_depth` | `8` |
| `learning_rate` | `0.2716428797162291` |
| `subsample` | `0.6397093908340729` |
| `colsample_bytree` | `0.8011873577181269` |
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
