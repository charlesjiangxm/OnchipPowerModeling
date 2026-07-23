# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_vb/aq_lsu_vb_internal_sequential_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_vb_internal_sequential_mlp_20260616_041248`
- Algorithm: **MLP**
- Feature selection: **sequential** (top_k=20)
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
| train | 0.075 | 0.075 | 0.00000 | 0.00000 | 0.4644 |
| val | 0.069 | 0.069 | 0.00000 | 0.00000 | 0.4344 |
| test | 0.577 | 0.579 | 0.00002 | 0.00002 | -0.0677 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `128` |
| `hidden2` | `32` |
| `dropout` | `0.2921508211200195` |
| `lr` | `0.006105057960669897` |
| `weight_decay` | `7.653560946832502e-05` |
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
