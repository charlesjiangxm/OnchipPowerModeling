# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_arb/aq_lsu_arb_internal_deep_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_arb_internal_deep_gbdt_20260617_042620`
- Algorithm: **GBDT**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 10 |
| after_preprocess | 5461 | 1365 | 34205 | 10 |
| after_feature_selection | 5461 | 1365 | 34205 | 8 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 10.593 | 12.426 | 0.00000 | 0.00000 | 0.9994 |
| val | 12.303 | 14.230 | 0.00000 | 0.00000 | 0.9935 |
| test | 18.404 | 18.412 | 0.00001 | 0.00000 | 0.8364 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `839` |
| `max_depth` | `8` |
| `learning_rate` | `0.008997869628841132` |
| `subsample` | `0.6577398235259567` |
| `colsample_bytree` | `0.7419395807016659` |
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
