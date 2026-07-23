# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_dc/aq_lsu_dc_input_variance_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_dc_input_variance_gbdt_20260617_054909`
- Algorithm: **GBDT**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 124 |
| after_preprocess | 5461 | 1365 | 34205 | 100 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.147 | 0.147 | 0.00001 | 0.00000 | 0.9990 |
| val | 0.329 | 0.330 | 0.00001 | 0.00001 | 0.9913 |
| test | 4.427 | 4.466 | 0.00012 | 0.00010 | -0.7424 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1151` |
| `max_depth` | `6` |
| `learning_rate` | `0.19158732208551613` |
| `subsample` | `0.778576255227293` |
| `colsample_bytree` | `0.9698039386386264` |
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
