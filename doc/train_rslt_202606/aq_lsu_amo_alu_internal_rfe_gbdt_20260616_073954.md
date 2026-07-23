# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amo_alu/aq_lsu_amo_alu_internal_rfe_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amo_alu_internal_rfe_gbdt_20260616_073954`
- Algorithm: **GBDT**
- Feature selection: **rfe** (top_k=20)
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
| train | 0.166 | 0.166 | 0.00000 | 0.00000 | 0.9996 |
| val | 0.592 | 0.595 | 0.00000 | 0.00000 | 0.9936 |
| test | 3.908 | 3.589 | 0.00005 | 0.00002 | -0.1003 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1140` |
| `max_depth` | `8` |
| `learning_rate` | `0.011123015817281598` |
| `subsample` | `0.7605217215152844` |
| `colsample_bytree` | `0.7487305806655858` |
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
