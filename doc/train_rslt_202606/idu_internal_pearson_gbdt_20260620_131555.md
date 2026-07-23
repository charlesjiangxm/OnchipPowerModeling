# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_internal_pearson_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_internal_pearson_gbdt_20260620_131555`
- Algorithm: **GBDT**
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
| train | 0.187 | 0.187 | 0.00004 | 0.00002 | 0.9971 |
| val | 0.309 | 0.309 | 0.00007 | 0.00004 | 0.9919 |
| test | 0.783 | 0.786 | 0.00013 | 0.00010 | 0.9449 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1489` |
| `max_depth` | `5` |
| `learning_rate` | `0.013582787299631978` |
| `subsample` | `0.6339771426492528` |
| `colsample_bytree` | `0.8841032391633499` |
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
