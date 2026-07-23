# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_dc/aq_lsu_dc_input_mcp_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_dc_input_mcp_mlp_20260617_005548`
- Algorithm: **MLP**
- Feature selection: **mcp** (top_k=20)
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
| train | 0.203 | 0.203 | 0.00001 | 0.00000 | 0.9984 |
| val | 0.358 | 0.358 | 0.00001 | 0.00001 | 0.9906 |
| test | 8.232 | 8.631 | 0.00026 | 0.00019 | -6.4377 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `256` |
| `hidden2` | `128` |
| `dropout` | `0.0514924800382936` |
| `lr` | `0.004943018624063405` |
| `weight_decay` | `0.00013242388036060113` |
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
