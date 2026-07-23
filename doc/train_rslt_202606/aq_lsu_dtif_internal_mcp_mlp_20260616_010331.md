# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_dtif/aq_lsu_dtif_internal_mcp_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_dtif_internal_mcp_mlp_20260616_010331`
- Algorithm: **MLP**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 61 |
| after_preprocess | 5461 | 1365 | 34205 | 28 |
| after_feature_selection | 5461 | 1365 | 34205 | 12 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.012 | 0.012 | 0.00000 | 0.00000 | 0.9946 |
| val | 0.013 | 0.013 | 0.00000 | 0.00000 | 0.9890 |
| test | 0.114 | 0.114 | 0.00000 | 0.00000 | -0.0226 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `512` |
| `hidden2` | `64` |
| `dropout` | `0.07663806616227684` |
| `lr` | `0.0013359722334920726` |
| `weight_decay` | `5.2520290751712484e-06` |
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
