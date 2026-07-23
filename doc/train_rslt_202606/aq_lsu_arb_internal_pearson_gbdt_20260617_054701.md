# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_arb/aq_lsu_arb_internal_pearson_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_arb_internal_pearson_gbdt_20260617_054701`
- Algorithm: **GBDT**
- Feature selection: **pearson** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 10 |
| after_preprocess | 5461 | 1365 | 34205 | 10 |
| after_feature_selection | 5461 | 1365 | 34205 | 10 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 4.894 | 5.249 | 0.00000 | 0.00000 | 0.9996 |
| val | 6.628 | 7.133 | 0.00000 | 0.00000 | 0.9947 |
| test | 17.501 | 16.845 | 0.00001 | 0.00000 | 0.8695 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `224` |
| `max_depth` | `7` |
| `learning_rate` | `0.1969925774326483` |
| `subsample` | `0.701697601007438` |
| `colsample_bytree` | `0.6116799654903184` |
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
