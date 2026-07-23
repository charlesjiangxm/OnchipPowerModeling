# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_icc/aq_lsu_icc_input_variance_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_icc_input_variance_rulefit_20260616_195048`
- Algorithm: **RuleFit**
- Feature selection: **variance** (top_k=20)
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
| train | 0.005 | 0.005 | 0.00000 | 0.00000 | 0.9991 |
| val | 0.006 | 0.006 | 0.00000 | 0.00000 | 0.9976 |
| test | 0.008 | 0.008 | 0.00000 | 0.00000 | 0.8327 |

## Best HPO trial

| key | value |
|---|---|
| `tree_size` | `5` |
| `max_rules` | `4000` |
| `memory_par` | `0.01120760621186057` |

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
