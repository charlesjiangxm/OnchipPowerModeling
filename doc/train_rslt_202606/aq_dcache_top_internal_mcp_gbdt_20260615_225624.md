# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_dcache_top/aq_dcache_top_internal_mcp_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_dcache_top_internal_mcp_gbdt_20260615_225624`
- Algorithm: **GBDT**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 15 |
| after_preprocess | 5461 | 1365 | 34205 | 15 |
| after_feature_selection | 5461 | 1365 | 34205 | 8 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 1.358 | 1.363 | 0.00015 | 0.00008 | 0.9997 |
| val | 2.237 | 2.249 | 0.00036 | 0.00015 | 0.9979 |
| test | 7.761 | 7.308 | 0.00092 | 0.00046 | 0.8961 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1120` |
| `max_depth` | `5` |
| `learning_rate` | `0.03230061469147422` |
| `subsample` | `0.7900071197083152` |
| `colsample_bytree` | `0.754545808777302` |
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
