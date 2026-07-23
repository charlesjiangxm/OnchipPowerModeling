# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_ag/aq_lsu_ag_internal_pearson_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_ag_internal_pearson_gbdt_20260616_113129`
- Algorithm: **GBDT**
- Feature selection: **pearson** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 211 |
| after_preprocess | 5461 | 1365 | 34205 | 159 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 1.514 | 1.498 | 0.00004 | 0.00002 | 0.9560 |
| val | 2.042 | 2.052 | 0.00006 | 0.00003 | 0.9264 |
| test | 2.718 | 2.650 | 0.00008 | 0.00005 | 0.7445 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `516` |
| `max_depth` | `8` |
| `learning_rate` | `0.0775613622427431` |
| `subsample` | `0.6792979049330922` |
| `colsample_bytree` | `0.6325649579442049` |
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
