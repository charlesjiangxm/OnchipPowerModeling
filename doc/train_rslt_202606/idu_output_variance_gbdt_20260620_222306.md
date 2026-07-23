# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_output_variance_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_output_variance_gbdt_20260620_222306`
- Algorithm: **GBDT**
- Feature selection: **variance** (top_k=20)
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
| train | 0.127 | 0.127 | 0.00003 | 0.00002 | 0.9985 |
| val | 0.251 | 0.250 | 0.00006 | 0.00003 | 0.9941 |
| test | 1.055 | 1.048 | 0.00018 | 0.00014 | 0.8994 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1182` |
| `max_depth` | `7` |
| `learning_rate` | `0.014119641044925826` |
| `subsample` | `0.7023026457100969` |
| `colsample_bytree` | `0.7500377129903372` |
| `tree_method` | `hist` |
| `n_jobs` | `-1` |
| `early_stopping_rounds` | `30` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### interaction_heatmap

![interaction_heatmap](artifacts/interaction_heatmap.png)

### convergence

![convergence](artifacts/convergence.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)
