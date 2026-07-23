# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_mcic/aq_lsu_mcic_internal_from_model_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_mcic_internal_from_model_gbdt_20260617_193712`
- Algorithm: **GBDT**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 16 |
| after_preprocess | 5461 | 1365 | 34205 | 13 |
| after_feature_selection | 5461 | 1365 | 34205 | 13 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 4.924 | 4.913 | 0.00000 | 0.00000 | 0.8093 |
| val | 5.132 | 5.344 | 0.00000 | 0.00000 | 0.6985 |
| test | 5.674 | 5.641 | 0.00000 | 0.00000 | 0.5672 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `657` |
| `max_depth` | `8` |
| `learning_rate` | `0.21172004795499297` |
| `subsample` | `0.6354502620595482` |
| `colsample_bytree` | `0.9003934616597797` |
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
