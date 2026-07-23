# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amr/aq_lsu_amr_internal_from_model_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amr_internal_from_model_elasticnet_20260616_235014`
- Algorithm: **ElasticNetCV**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 36 |
| after_preprocess | 5461 | 1365 | 34205 | 33 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 2.430 | 2.407 | 0.00001 | 0.00001 | 0.7062 |
| val | 2.706 | 2.652 | 0.00001 | 0.00001 | 0.5147 |
| test | 3.785 | 4.207 | 0.00004 | 0.00001 | -12.5580 |

## Best HPO trial

| key | value |
|---|---|
| `n_alphas` | `10` |
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
