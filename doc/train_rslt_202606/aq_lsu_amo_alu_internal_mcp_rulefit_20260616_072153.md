# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amo_alu/aq_lsu_amo_alu_internal_mcp_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amo_alu_internal_mcp_rulefit_20260616_072153`
- Algorithm: **RuleFit**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 49 |
| after_preprocess | 5461 | 1365 | 34205 | 48 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 2.141 | 2.147 | 0.00001 | 0.00001 | 0.9424 |
| val | 2.119 | 2.121 | 0.00001 | 0.00001 | 0.9490 |
| test | 4.929 | 4.694 | 0.00006 | 0.00002 | -0.2072 |

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
