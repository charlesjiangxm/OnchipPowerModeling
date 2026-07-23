# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_internal_pearson_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_internal_pearson_elasticnet_20260620_130909`
- Algorithm: **ElasticNetCV**
- Feature selection: **pearson** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 132 |
| after_preprocess | 5461 | 1365 | 34205 | 122 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 1.101 | 1.099 | 0.00018 | 0.00015 | 0.9350 |
| val | 1.032 | 1.032 | 0.00017 | 0.00014 | 0.9491 |
| test | 1.608 | 1.599 | 0.00024 | 0.00021 | 0.8174 |

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
