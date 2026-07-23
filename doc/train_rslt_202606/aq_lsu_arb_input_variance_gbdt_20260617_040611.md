# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_arb/aq_lsu_arb_input_variance_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_arb_input_variance_gbdt_20260617_040611`
- Algorithm: **GBDT**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 86 |
| after_preprocess | 5461 | 1365 | 34205 | 72 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 102.804 | 1138.744 | 0.00003 | 0.00002 | 0.2936 |
| val | 89.194 | 1447.584 | 0.00002 | 0.00001 | 0.4377 |
| test | 70.553 | 675.248 | 0.00001 | 0.00001 | 0.2700 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1472` |
| `max_depth` | `8` |
| `learning_rate` | `0.1502777539696764` |
| `subsample` | `0.6189218308655252` |
| `colsample_bytree` | `0.834607914568382` |
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
