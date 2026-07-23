# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_dc/aq_lsu_dc_input_univariate_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_dc_input_univariate_mlp_20260617_051942`
- Algorithm: **MLP**
- Feature selection: **univariate** (top_k=20)
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
| train | 0.297 | 0.297 | 0.00001 | 0.00001 | 0.9970 |
| val | 0.347 | 0.347 | 0.00001 | 0.00001 | 0.9935 |
| test | 6.668 | 6.859 | 0.00018 | 0.00015 | -2.8586 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `512` |
| `hidden2` | `256` |
| `dropout` | `0.21205720315428514` |
| `lr` | `0.0028708753481954683` |
| `weight_decay` | `0.00020597335357437193` |
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
