# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_amr/aq_lsu_amr_internal_from_model_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_amr_internal_from_model_mlp_20260615_234800`
- Algorithm: **MLP**
- Feature selection: **from_model** (top_k=20)
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
| train | 1.113 | 1.117 | 0.00001 | 0.00000 | 0.9351 |
| val | 1.261 | 1.260 | 0.00001 | 0.00000 | 0.9232 |
| test | 2.434 | 2.556 | 0.00002 | 0.00001 | -3.6354 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `256` |
| `hidden2` | `32` |
| `dropout` | `0.24742200657301575` |
| `lr` | `0.004792795327114602` |
| `weight_decay` | `1.1038251665362905e-05` |
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
