# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_output_pearson_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_output_pearson_gbdt_20260620_193502`
- Algorithm: **GBDT**
- Feature selection: **pearson** (top_k=20)
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
| train | 0.128 | 0.128 | 0.00003 | 0.00002 | 0.9984 |
| val | 0.316 | 0.316 | 0.00007 | 0.00004 | 0.9906 |
| test | 0.655 | 0.657 | 0.00012 | 0.00008 | 0.9556 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1493` |
| `max_depth` | `8` |
| `learning_rate` | `0.016441583897426497` |
| `subsample` | `0.608224233463583` |
| `colsample_bytree` | `0.6310699770810441` |
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
