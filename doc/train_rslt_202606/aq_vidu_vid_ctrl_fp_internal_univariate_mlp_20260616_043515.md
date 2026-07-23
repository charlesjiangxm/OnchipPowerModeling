# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_vidu_vid_ctrl_fp/aq_vidu_vid_ctrl_fp_internal_univariate_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_vidu_vid_ctrl_fp_internal_univariate_mlp_20260616_043515`
- Algorithm: **MLP**
- Feature selection: **univariate** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 19 |
| after_preprocess | 5461 | 1365 | 34205 | 19 |
| after_feature_selection | 5461 | 1365 | 34205 | 19 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 61.010 | 336.669 | 0.00000 | 0.00000 | 0.6932 |
| val | 67.538 | 367.018 | 0.00000 | 0.00000 | 0.6047 |
| test | 62.383 | 351.979 | 0.00000 | 0.00000 | 0.9366 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `256` |
| `hidden2` | `128` |
| `dropout` | `0.16747917770685367` |
| `lr` | `0.008606908807860723` |
| `weight_decay` | `0.0005981199861922954` |
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
