# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_internal_rfe_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_internal_rfe_gbdt_20260619_014028`
- Algorithm: **GBDT**
- Feature selection: **rfe** (top_k=20)
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
| train | 0.037 | 0.037 | 0.00000 | 0.00000 | 0.9337 |
| val | 0.053 | 0.053 | 0.00000 | 0.00000 | 0.8445 |
| test | 0.099 | 0.099 | 0.00001 | 0.00000 | 0.3845 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `900` |
| `max_depth` | `8` |
| `learning_rate` | `0.15688028252360947` |
| `subsample` | `0.6428990434400849` |
| `colsample_bytree` | `0.6259182439496608` |
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
