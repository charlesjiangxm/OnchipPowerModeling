# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_internal_mcp_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_internal_mcp_gbdt_20260620_091153`
- Algorithm: **GBDT**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 132 |
| after_preprocess | 5461 | 1365 | 34205 | 122 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.075 | 0.075 | 0.00001 | 0.00001 | 0.9996 |
| val | 0.176 | 0.176 | 0.00004 | 0.00002 | 0.9971 |
| test | 0.451 | 0.450 | 0.00008 | 0.00006 | 0.9774 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1178` |
| `max_depth` | `6` |
| `learning_rate` | `0.0166755042116576` |
| `subsample` | `0.6249722219796175` |
| `colsample_bytree` | `0.9566964813459903` |
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
