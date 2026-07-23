# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_internal_deep_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_internal_deep_gbdt_20260618_204035`
- Algorithm: **GBDT**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 190 |
| after_preprocess | 5461 | 1365 | 34205 | 137 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.035 | 0.035 | 0.00000 | 0.00000 | 0.9404 |
| val | 0.053 | 0.053 | 0.00000 | 0.00000 | 0.8411 |
| test | 0.115 | 0.115 | 0.00001 | 0.00001 | 0.2741 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `321` |
| `max_depth` | `8` |
| `learning_rate` | `0.12788315237262698` |
| `subsample` | `0.8525337074571916` |
| `colsample_bytree` | `0.6135860870222191` |
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
