# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_dcache_top/aq_dcache_top_internal_variance_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_dcache_top_internal_variance_rulefit_20260616_004411`
- Algorithm: **RuleFit**
- Feature selection: **variance** (top_k=20)
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
| train | 3.430 | 3.491 | 0.00059 | 0.00028 | 0.9950 |
| val | 3.268 | 3.338 | 0.00053 | 0.00026 | 0.9955 |
| test | 7.663 | 7.264 | 0.00097 | 0.00047 | 0.8848 |

## Best HPO trial

| key | value |
|---|---|
| `tree_size` | `2` |
| `max_rules` | `500` |
| `memory_par` | `0.08706020878304858` |

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
