# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_ag/aq_lsu_ag_input_sequential_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_ag_input_sequential_gbdt_20260616_034521`
- Algorithm: **GBDT**
- Feature selection: **sequential** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 50 |
| after_preprocess | 5461 | 1365 | 34205 | 38 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.389 | 0.388 | 0.00001 | 0.00001 | 0.9977 |
| val | 1.106 | 1.118 | 0.00004 | 0.00002 | 0.9687 |
| test | 6.518 | 6.216 | 0.00015 | 0.00011 | 0.0618 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1363` |
| `max_depth` | `8` |
| `learning_rate` | `0.012567727382596752` |
| `subsample` | `0.8890023337594057` |
| `colsample_bytree` | `0.6850155855165673` |
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
