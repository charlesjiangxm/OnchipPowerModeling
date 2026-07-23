# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_vb/aq_lsu_vb_internal_mcp_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_vb_internal_mcp_mlp_20260616_040734`
- Algorithm: **MLP**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 48 |
| after_preprocess | 5461 | 1365 | 34205 | 36 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.056 | 0.056 | 0.00000 | 0.00000 | 0.8180 |
| val | 0.056 | 0.056 | 0.00000 | 0.00000 | 0.7449 |
| test | 0.532 | 0.532 | 0.00002 | 0.00001 | 0.0667 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `128` |
| `hidden2` | `32` |
| `dropout` | `0.27833824391884876` |
| `lr` | `0.009892453393873447` |
| `weight_decay` | `1.0374946842447067e-05` |
| `batch_size` | `256` |
| `max_epochs` | `300` |
| `patience` | `40` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### convergence

![convergence](artifacts/convergence.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)

## Interaction heatmap

Not produced for **MLP** (interaction extraction not defined or unavailable in this environment).
