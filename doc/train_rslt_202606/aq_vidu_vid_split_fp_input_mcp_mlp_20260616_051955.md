# Run report — MLP

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_vidu_vid_split_fp/aq_vidu_vid_split_fp_input_mcp_mlp.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_vidu_vid_split_fp_input_mcp_mlp_20260616_051955`
- Algorithm: **MLP**
- Feature selection: **mcp** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 12 |
| after_preprocess | 5461 | 1365 | 34205 | 8 |
| after_feature_selection | 5461 | 1365 | 34205 | 8 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 1.212 | 1.214 | 0.00001 | 0.00001 | 0.6039 |
| val | 1.236 | 1.241 | 0.00001 | 0.00001 | 0.5579 |
| test | 1.890 | 1.886 | 0.00002 | 0.00001 | 0.8718 |

## Best HPO trial

| key | value |
|---|---|
| `hidden1` | `256` |
| `hidden2` | `256` |
| `dropout` | `0.10675896148262055` |
| `lr` | `0.005009110736587348` |
| `weight_decay` | `0.00020446625752547947` |
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
