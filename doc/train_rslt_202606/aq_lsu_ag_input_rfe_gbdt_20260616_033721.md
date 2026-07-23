# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_ag/aq_lsu_ag_input_rfe_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_ag_input_rfe_gbdt_20260616_033721`
- Algorithm: **GBDT**
- Feature selection: **rfe** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 50 |
| after_preprocess | 5461 | 1365 | 34205 | 38 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.337 | 0.337 | 0.00001 | 0.00001 | 0.9983 |
| val | 0.832 | 0.837 | 0.00003 | 0.00001 | 0.9839 |
| test | 3.839 | 3.730 | 0.00011 | 0.00007 | 0.5347 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `625` |
| `max_depth` | `7` |
| `learning_rate` | `0.04818698083227556` |
| `subsample` | `0.6207679281222178` |
| `colsample_bytree` | `0.6068168224647184` |
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
