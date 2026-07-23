# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_rdl/aq_lsu_rdl_input_variance_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_rdl_input_variance_mlp_20260616_031356`
- Algorithm: **MLP**
- Feature selection: **variance** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 34 |
| after_preprocess | 5461 | 1365 | 34205 | 26 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.012 | 0.012 | 0.00000 | 0.00000 | 0.9787 |
| val | 0.023 | 0.023 | 0.00000 | 0.00000 | 0.9652 |
| test | 0.171 | 0.170 | 0.00000 | 0.00000 | 0.9360 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `256` |
| `hidden2` | `32` |
| `dropout` | `0.2580858575119409` |
| `lr` | `0.0006791549048154131` |
| `weight_decay` | `4.007164633453806e-05` |
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
