# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_ag/aq_lsu_ag_input_variance_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_ag_input_variance_rulefit_20260616_041013`
- Algorithm: **RuleFit**
- Feature selection: **variance** (top_k=20)
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
| train | 1.562 | 1.568 | 0.00004 | 0.00003 | 0.9614 |
| val | 1.619 | 1.621 | 0.00004 | 0.00003 | 0.9557 |
| test | 4.140 | 4.224 | 0.00009 | 0.00007 | 0.6773 |

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
