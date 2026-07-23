# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_vidu_vid_ctrl_fp/aq_vidu_vid_ctrl_fp_internal_from_model_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_vidu_vid_ctrl_fp_internal_from_model_mlp_20260616_042950`
- Algorithm: **MLP**
- Feature selection: **from_model** (top_k=20)
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
| train | 59.885 | 339.692 | 0.00000 | 0.00000 | 0.6921 |
| val | 67.001 | 372.628 | 0.00000 | 0.00000 | 0.6028 |
| test | 59.763 | 349.220 | 0.00000 | 0.00000 | 0.9561 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `512` |
| `hidden2` | `64` |
| `dropout` | `0.17147225468627325` |
| `lr` | `0.004337416862919889` |
| `weight_decay` | `0.0005032454606423932` |
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
