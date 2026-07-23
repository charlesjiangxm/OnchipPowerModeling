# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_output_univariate_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_output_univariate_mlp_20260619_082259`
- Algorithm: **MLP**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 131 |
| after_preprocess | 5461 | 1365 | 34205 | 92 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.058 | 0.058 | 0.00000 | 0.00000 | 0.8459 |
| val | 0.060 | 0.060 | 0.00000 | 0.00000 | 0.8038 |
| test | 0.967 | 0.972 | 0.00004 | 0.00004 | -37.2687 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `128` |
| `hidden2` | `128` |
| `dropout` | `0.011829635934195972` |
| `lr` | `0.0022617568947805498` |
| `weight_decay` | `6.832340393399646e-06` |
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
