# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/ifu/ifu_all_univariate_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/ifu_all_univariate_rulefit_20260620_211212`
- Algorithm: **RuleFit**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 258 |
| after_preprocess | 5461 | 1365 | 34205 | 239 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 3.343 | 3.370 | 0.00060 | 0.00044 | 0.9855 |
| val | 3.389 | 3.461 | 0.00053 | 0.00044 | 0.9889 |
| test | 8.456 | 8.915 | 0.00113 | 0.00106 | 0.9122 |

## Best HPO trial

| key | value |
|---|---|
| `tree_size` | `3` |
| `max_rules` | `500` |
| `memory_par` | `0.002051110418843397` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### interaction_heatmap

![interaction_heatmap](artifacts/interaction_heatmap.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)
