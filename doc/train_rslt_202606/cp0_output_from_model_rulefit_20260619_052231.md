# Run report — RuleFit

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_output_from_model_rulefit.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_output_from_model_rulefit_20260619_052231`
- Algorithm: **RuleFit**
- Feature selection: **from_model** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 131 |
| after_preprocess | 5461 | 1365 | 34205 | 92 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.064 | 0.064 | 0.00000 | 0.00000 | 0.8218 |
| val | 0.062 | 0.062 | 0.00000 | 0.00000 | 0.7969 |
| test | 0.137 | 0.137 | 0.00001 | 0.00001 | -0.0341 |

## Best HPO trial

| key | value |
|---|---|
| `tree_size` | `4` |
| `max_rules` | `500` |
| `memory_par` | `0.004473636174621266` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### interaction_heatmap

![interaction_heatmap](artifacts/interaction_heatmap.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)
