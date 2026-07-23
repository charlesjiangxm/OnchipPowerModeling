# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_icc/aq_lsu_icc_input_univariate_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_icc_input_univariate_gbdt_20260616_194553`
- Algorithm: **GBDT**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 21 |
| after_preprocess | 5461 | 1365 | 34205 | 17 |
| after_feature_selection | 5461 | 1365 | 34205 | 17 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.003 | 0.003 | 0.00000 | 0.00000 | 0.9996 |
| val | 0.006 | 0.006 | 0.00000 | 0.00000 | 0.9933 |
| test | 0.006 | 0.006 | 0.00000 | 0.00000 | 0.8790 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `473` |
| `max_depth` | `7` |
| `learning_rate` | `0.1590177133226669` |
| `subsample` | `0.881089887751614` |
| `colsample_bytree` | `0.9251271028071485` |
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
