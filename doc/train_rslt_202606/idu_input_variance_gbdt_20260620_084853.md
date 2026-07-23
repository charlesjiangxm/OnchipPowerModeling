# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_input_variance_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_input_variance_gbdt_20260620_084853`
- Algorithm: **GBDT**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 60 |
| after_preprocess | 5461 | 1365 | 34205 | 45 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.162 | 0.162 | 0.00003 | 0.00002 | 0.9978 |
| val | 0.240 | 0.240 | 0.00006 | 0.00003 | 0.9939 |
| test | 0.789 | 0.783 | 0.00015 | 0.00010 | 0.9232 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1021` |
| `max_depth` | `4` |
| `learning_rate` | `0.06774584721230024` |
| `subsample` | `0.9165069232671154` |
| `colsample_bytree` | `0.8332263047016858` |
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
