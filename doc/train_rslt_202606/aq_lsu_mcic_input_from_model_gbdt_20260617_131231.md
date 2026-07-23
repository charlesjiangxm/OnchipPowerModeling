# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_mcic/aq_lsu_mcic_input_from_model_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_mcic_input_from_model_gbdt_20260617_131231`
- Algorithm: **GBDT**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 17 |
| after_preprocess | 5461 | 1365 | 34205 | 12 |
| after_feature_selection | 5461 | 1365 | 34205 | 12 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 4.839 | 4.830 | 0.00000 | 0.00000 | 0.8132 |
| val | 5.082 | 5.297 | 0.00000 | 0.00000 | 0.6970 |
| test | 6.029 | 5.781 | 0.00000 | 0.00000 | 0.4946 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `877` |
| `max_depth` | `8` |
| `learning_rate` | `0.007118208725011703` |
| `subsample` | `0.695683479831051` |
| `colsample_bytree` | `0.8273574991182768` |
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
