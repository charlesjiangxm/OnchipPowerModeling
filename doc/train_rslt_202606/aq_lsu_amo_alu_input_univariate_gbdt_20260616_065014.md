# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amo_alu/aq_lsu_amo_alu_input_univariate_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amo_alu_input_univariate_gbdt_20260616_065014`
- Algorithm: **GBDT**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 11 |
| after_preprocess | 5461 | 1365 | 34205 | 8 |
| after_feature_selection | 5461 | 1365 | 34205 | 8 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.421 | 0.421 | 0.00000 | 0.00000 | 0.9973 |
| val | 1.156 | 1.163 | 0.00001 | 0.00000 | 0.9712 |
| test | 4.451 | 4.099 | 0.00005 | 0.00002 | -0.0428 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1452` |
| `max_depth` | `7` |
| `learning_rate` | `0.03955799307227324` |
| `subsample` | `0.6313849539867122` |
| `colsample_bytree` | `0.8524500150055455` |
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
