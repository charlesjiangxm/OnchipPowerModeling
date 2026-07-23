# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amo_alu/aq_lsu_amo_alu_internal_deep_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amo_alu_internal_deep_gbdt_20260616_065921`
- Algorithm: **GBDT**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 49 |
| after_preprocess | 5461 | 1365 | 34205 | 48 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.079 | 0.079 | 0.00000 | 0.00000 | 0.9999 |
| val | 0.553 | 0.556 | 0.00000 | 0.00000 | 0.9944 |
| test | 3.897 | 3.571 | 0.00005 | 0.00002 | -0.1106 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `624` |
| `max_depth` | `8` |
| `learning_rate` | `0.053011269715584376` |
| `subsample` | `0.6071847502459279` |
| `colsample_bytree` | `0.6061470949312417` |
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
