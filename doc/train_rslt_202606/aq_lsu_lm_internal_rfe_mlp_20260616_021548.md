# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_lm/aq_lsu_lm_internal_rfe_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_lm_internal_rfe_mlp_20260616_021548`
- Algorithm: **MLP**
- Feature selection: **rfe** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 9 |
| after_preprocess | 5461 | 1365 | 34205 | 8 |
| after_feature_selection | 5461 | 1365 | 34205 | 8 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.664 | 0.661 | 0.00000 | 0.00000 | -0.0240 |
| val | 0.563 | 0.562 | 0.00000 | 0.00000 | -0.0008 |
| test | 0.368 | 0.368 | 0.00000 | 0.00000 | -0.0033 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `64` |
| `hidden2` | `32` |
| `dropout` | `0.020463785979890174` |
| `lr` | `0.00010966322280959204` |
| `weight_decay` | `0.00040030043313706845` |
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
