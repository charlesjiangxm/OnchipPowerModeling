# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_internal_mcp_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_internal_mcp_mlp_20260619_002343`
- Algorithm: **MLP**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 190 |
| after_preprocess | 5461 | 1365 | 34205 | 137 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.047 | 0.047 | 0.00000 | 0.00000 | 0.9036 |
| val | 0.054 | 0.054 | 0.00000 | 0.00000 | 0.8411 |
| test | 0.089 | 0.089 | 0.00001 | 0.00000 | 0.4485 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `128` |
| `hidden2` | `32` |
| `dropout` | `0.26367836686375823` |
| `lr` | `0.00961076233538644` |
| `weight_decay` | `3.731808269104097e-05` |
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
