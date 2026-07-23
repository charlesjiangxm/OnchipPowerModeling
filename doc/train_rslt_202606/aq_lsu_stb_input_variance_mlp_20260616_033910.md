# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_stb/aq_lsu_stb_input_variance_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_stb_input_variance_mlp_20260616_033910`
- Algorithm: **MLP**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 52 |
| after_preprocess | 5461 | 1365 | 34205 | 47 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.032 | 0.032 | 0.00000 | 0.00000 | 0.9987 |
| val | 0.055 | 0.055 | 0.00000 | 0.00000 | 0.9925 |
| test | 0.698 | 0.693 | 0.00004 | 0.00003 | 0.7580 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `512` |
| `hidden2` | `64` |
| `dropout` | `0.037610946763000086` |
| `lr` | `0.0038596250826026628` |
| `weight_decay` | `3.926830294386704e-06` |
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
