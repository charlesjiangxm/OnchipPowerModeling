# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amr/aq_lsu_amr_internal_sequential_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amr_internal_sequential_mlp_20260615_235336`
- Algorithm: **MLP**
- Feature selection: **sequential** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 36 |
| after_preprocess | 5461 | 1365 | 34205 | 33 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.632 | 0.632 | 0.00000 | 0.00000 | 0.9744 |
| val | 0.740 | 0.739 | 0.00000 | 0.00000 | 0.9678 |
| test | 2.433 | 2.515 | 0.00002 | 0.00001 | -1.9894 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `64` |
| `hidden2` | `64` |
| `dropout` | `0.27427172265738764` |
| `lr` | `0.0029327181925704817` |
| `weight_decay` | `3.0863693904739494e-05` |
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
