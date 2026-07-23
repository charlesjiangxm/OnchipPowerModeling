# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_arb/aq_lsu_arb_internal_from_model_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_arb_internal_from_model_gbdt_20260617_044457`
- Algorithm: **GBDT**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 10 |
| after_preprocess | 5461 | 1365 | 34205 | 10 |
| after_feature_selection | 5461 | 1365 | 34205 | 10 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 5.470 | 6.003 | 0.00000 | 0.00000 | 0.9994 |
| val | 6.834 | 7.437 | 0.00000 | 0.00000 | 0.9940 |
| test | 17.844 | 17.074 | 0.00001 | 0.00000 | 0.8331 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `795` |
| `max_depth` | `7` |
| `learning_rate` | `0.036650675204369744` |
| `subsample` | `0.7772428375636871` |
| `colsample_bytree` | `0.6018960478612357` |
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
