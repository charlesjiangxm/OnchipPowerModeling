# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_mcic/aq_lsu_mcic_input_univariate_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_mcic_input_univariate_mlp_20260616_022644`
- Algorithm: **MLP**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 17 |
| after_preprocess | 5461 | 1365 | 34205 | 12 |
| after_feature_selection | 5461 | 1365 | 34205 | 12 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 7.144 | 7.014 | 0.00000 | 0.00000 | 0.6007 |
| val | 5.636 | 5.626 | 0.00000 | 0.00000 | 0.6731 |
| test | 7.734 | 8.034 | 0.00000 | 0.00000 | 0.1858 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `512` |
| `hidden2` | `256` |
| `dropout` | `0.27676921780831043` |
| `lr` | `0.003550730183787484` |
| `weight_decay` | `3.5177569915392994e-05` |
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
