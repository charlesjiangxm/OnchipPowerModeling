# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_vidu_vid_wbt_fp/aq_vidu_vid_wbt_fp_internal_deep_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_vidu_vid_wbt_fp_internal_deep_mlp_20260616_054616`
- Algorithm: **MLP**
- Feature selection: **deep** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 43 |
| after_preprocess | 5461 | 1365 | 34205 | 41 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 1.524 | 1.524 | 0.00001 | 0.00001 | 0.8652 |
| val | 1.551 | 1.579 | 0.00001 | 0.00001 | 0.8628 |
| test | 96.952 | 102.151 | 0.00073 | 0.00051 | -981.9838 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `128` |
| `hidden2` | `128` |
| `dropout` | `0.0923687539161436` |
| `lr` | `0.004271167342691713` |
| `weight_decay` | `0.0007630192061438584` |
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
