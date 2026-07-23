# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/ifu/ifu_input_deep_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/ifu_input_deep_gbdt_20260620_215544`
- Algorithm: **GBDT**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 63 |
| after_preprocess | 5461 | 1365 | 34205 | 54 |
| after_feature_selection | 5461 | 1365 | 34205 | 18 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.282 | 0.282 | 0.00007 | 0.00005 | 0.9998 |
| val | 0.368 | 0.368 | 0.00010 | 0.00006 | 0.9996 |
| test | 3.248 | 3.335 | 0.00064 | 0.00049 | 0.9720 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1475` |
| `max_depth` | `3` |
| `learning_rate` | `0.05158169637249858` |
| `subsample` | `0.6617327572040236` |
| `colsample_bytree` | `0.9570287169475074` |
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
