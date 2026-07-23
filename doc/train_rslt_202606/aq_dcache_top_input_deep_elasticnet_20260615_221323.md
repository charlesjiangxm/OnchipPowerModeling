# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_dcache_top/aq_dcache_top_input_deep_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_dcache_top_input_deep_elasticnet_20260615_221323`
- Algorithm: **ElasticNetCV**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 24 |
| after_preprocess | 5461 | 1365 | 34205 | 21 |
| after_feature_selection | 5461 | 1365 | 34205 | 12 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 20.372 | 25.770 | 0.00173 | 0.00094 | 0.9558 |
| val | 18.789 | 25.478 | 0.00138 | 0.00067 | 0.9690 |
| test | 21.291 | 30.207 | 0.00389 | 0.00155 | -0.8354 |

## Best HPO trial

| key | value |
|---|---|
| `n_alphas` | `10` |
| `l1_ratio` | `[0.7]` |
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
