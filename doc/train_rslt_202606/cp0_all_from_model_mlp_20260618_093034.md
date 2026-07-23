# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_all_from_model_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_all_from_model_mlp_20260618_093034`
- Algorithm: **MLP**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 394 |
| after_preprocess | 5461 | 1365 | 34205 | 276 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.028 | 0.028 | 0.00000 | 0.00000 | 0.9516 |
| val | 0.034 | 0.034 | 0.00000 | 0.00000 | 0.9143 |
| test | 0.065 | 0.065 | 0.00000 | 0.00000 | 0.7148 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `64` |
| `hidden2` | `128` |
| `dropout` | `0.17273903302611862` |
| `lr` | `0.007254558458645523` |
| `weight_decay` | `4.910314560178836e-05` |
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
