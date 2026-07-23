# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_ag/aq_lsu_ag_input_sequential_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_ag_input_sequential_elasticnet_20260616_034519`
- Algorithm: **ElasticNetCV**
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
| train | 4.426 | 4.451 | 0.00010 | 0.00008 | 0.7696 |
| val | 5.078 | 5.159 | 0.00012 | 0.00009 | 0.6790 |
| test | 6.676 | 6.970 | 0.00015 | 0.00011 | 0.0592 |

## Best HPO trial

| key | value |
|---|---|
| `n_alphas` | `20` |
| `l1_ratio` | `[0.5]` |
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
