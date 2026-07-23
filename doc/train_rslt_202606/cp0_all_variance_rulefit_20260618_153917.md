# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_all_variance_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_all_variance_rulefit_20260618_153917`
- Algorithm: **RuleFit**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 394 |
| after_preprocess | 5461 | 1365 | 34205 | 276 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.141 | 0.141 | 0.00001 | 0.00001 | 0.3156 |
| val | 0.131 | 0.131 | 0.00001 | 0.00001 | 0.3294 |
| test | 0.128 | 0.129 | 0.00001 | 0.00001 | -0.1461 |

## Best HPO trial

| key | value |
|---|---|
| `tree_size` | `4` |
| `max_rules` | `1000` |
| `memory_par` | `0.06161049539380966` |

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
