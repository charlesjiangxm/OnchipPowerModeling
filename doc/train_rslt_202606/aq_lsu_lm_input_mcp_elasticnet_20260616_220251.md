# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_lm/aq_lsu_lm_input_mcp_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_lm_input_mcp_elasticnet_20260616_220251`
- Algorithm: **ElasticNetCV**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 12 |
| after_preprocess | 5461 | 1365 | 34205 | 9 |
| after_feature_selection | 5461 | 1365 | 34205 | 9 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.624 | 0.622 | 0.00000 | 0.00000 | 0.1066 |
| val | 0.518 | 0.517 | 0.00000 | 0.00000 | 0.1310 |
| test | 0.453 | 0.455 | 0.00000 | 0.00000 | -1.3891 |

## Best HPO trial

| key | value |
|---|---|
| `n_alphas` | `20` |
| `l1_ratio` | `[0.9]` |
| `cv` | `5` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)

## Interaction heatmap

Not produced for **ElasticNetCV** (interaction extraction not defined or unavailable in this environment).
