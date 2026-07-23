# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_all_sequential_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_all_sequential_mlp_20260618_132730`
- Algorithm: **MLP**
- Feature selection: **sequential** (top_k=20)
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
| train | 0.051 | 0.051 | 0.00000 | 0.00000 | 0.8814 |
| val | 0.053 | 0.053 | 0.00000 | 0.00000 | 0.8556 |
| test | 0.078 | 0.078 | 0.00000 | 0.00000 | 0.5537 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `256` |
| `hidden2` | `32` |
| `dropout` | `0.05091995773742268` |
| `lr` | `0.0034796358181524087` |
| `weight_decay` | `0.00032744954926295706` |
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
