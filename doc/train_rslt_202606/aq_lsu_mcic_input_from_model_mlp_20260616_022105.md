# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_mcic/aq_lsu_mcic_input_from_model_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_mcic_input_from_model_mlp_20260616_022105`
- Algorithm: **MLP**
- Feature selection: **from_model** (top_k=20)
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
| train | 7.116 | 6.988 | 0.00000 | 0.00000 | 0.5770 |
| val | 5.539 | 5.516 | 0.00000 | 0.00000 | 0.6683 |
| test | 7.065 | 7.425 | 0.00000 | 0.00000 | 0.1778 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `128` |
| `hidden2` | `128` |
| `dropout` | `0.14544567020656254` |
| `lr` | `0.0025313166694651357` |
| `weight_decay` | `4.704927707037195e-05` |
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
