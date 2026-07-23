# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_input_mcp_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_input_mcp_rulefit_20260620_034710`
- Algorithm: **RuleFit**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 60 |
| after_preprocess | 5461 | 1365 | 34205 | 45 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.252 | 0.252 | 0.00005 | 0.00003 | 0.9950 |
| val | 0.292 | 0.292 | 0.00006 | 0.00004 | 0.9928 |
| test | 1.593 | 1.577 | 0.00025 | 0.00020 | 0.7991 |

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
