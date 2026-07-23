# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/idu/idu_internal_variance_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/idu_internal_variance_mlp_20260620_130003`
- Algorithm: **MLP**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 132 |
| after_preprocess | 5461 | 1365 | 34205 | 122 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.150 | 0.150 | 0.00003 | 0.00002 | 0.9978 |
| val | 0.247 | 0.247 | 0.00005 | 0.00003 | 0.9947 |
| test | 0.901 | 0.893 | 0.00020 | 0.00012 | 0.8663 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `512` |
| `hidden2` | `32` |
| `dropout` | `0.0037844635930777927` |
| `lr` | `0.009635216891636898` |
| `weight_decay` | `8.476712114494962e-06` |
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
