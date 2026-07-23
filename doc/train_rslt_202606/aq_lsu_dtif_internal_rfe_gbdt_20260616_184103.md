# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_dtif/aq_lsu_dtif_internal_rfe_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_dtif_internal_rfe_gbdt_20260616_184103`
- Algorithm: **GBDT**
- Feature selection: **rfe** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 61 |
| after_preprocess | 5461 | 1365 | 34205 | 28 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.007 | 0.007 | 0.00000 | 0.00000 | 0.9987 |
| val | 0.014 | 0.014 | 0.00000 | 0.00000 | 0.9848 |
| test | 0.058 | 0.058 | 0.00000 | 0.00000 | 0.5051 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `827` |
| `max_depth` | `6` |
| `learning_rate` | `0.03805212070432476` |
| `subsample` | `0.7615117419183001` |
| `colsample_bytree` | `0.9258360514720573` |
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
