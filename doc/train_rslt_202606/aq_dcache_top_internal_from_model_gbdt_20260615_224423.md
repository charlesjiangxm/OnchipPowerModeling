# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_dcache_top/aq_dcache_top_internal_from_model_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_dcache_top_internal_from_model_gbdt_20260615_224423`
- Algorithm: **GBDT**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 15 |
| after_preprocess | 5461 | 1365 | 34205 | 15 |
| after_feature_selection | 5461 | 1365 | 34205 | 15 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 1.181 | 1.185 | 0.00016 | 0.00008 | 0.9996 |
| val | 1.891 | 1.927 | 0.00037 | 0.00014 | 0.9978 |
| test | 6.035 | 5.906 | 0.00075 | 0.00038 | 0.9321 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `534` |
| `max_depth` | `6` |
| `learning_rate` | `0.03457632271077726` |
| `subsample` | `0.7848459033604512` |
| `colsample_bytree` | `0.8473506698280117` |
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
