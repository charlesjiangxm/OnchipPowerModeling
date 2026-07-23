# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_lfb/aq_lsu_lfb_input_mcp_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_lfb_input_mcp_gbdt_20260616_205334`
- Algorithm: **GBDT**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 64 |
| after_preprocess | 5461 | 1365 | 34205 | 48 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.020 | 0.020 | 0.00000 | 0.00000 | 0.9862 |
| val | 0.036 | 0.036 | 0.00000 | 0.00000 | 0.9347 |
| test | 0.117 | 0.117 | 0.00001 | 0.00001 | 0.7682 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `1452` |
| `max_depth` | `7` |
| `learning_rate` | `0.007065703402403497` |
| `subsample` | `0.7071305342118264` |
| `colsample_bytree` | `0.7236140061769333` |
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
