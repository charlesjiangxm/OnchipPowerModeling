# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_output_mcp_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_output_mcp_rulefit_20260620_142342`
- Algorithm: **RuleFit**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 70 |
| after_preprocess | 5461 | 1365 | 34205 | 63 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.936 | 0.941 | 0.00016 | 0.00012 | 0.9525 |
| val | 0.885 | 0.889 | 0.00015 | 0.00012 | 0.9597 |
| test | 0.872 | 0.878 | 0.00015 | 0.00011 | 0.9309 |

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
