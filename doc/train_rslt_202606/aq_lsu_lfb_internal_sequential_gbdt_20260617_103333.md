# Run report — GBDT

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_lfb/aq_lsu_lfb_internal_sequential_gbdt.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_lfb_internal_sequential_gbdt_20260617_103333`
- Algorithm: **GBDT**
- Feature selection: **sequential** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 185 |
| after_preprocess | 5461 | 1365 | 34205 | 171 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.023 | 0.023 | 0.00000 | 0.00000 | 0.9785 |
| val | 0.052 | 0.052 | 0.00000 | 0.00000 | 0.8686 |
| test | 0.149 | 0.149 | 0.00001 | 0.00001 | 0.4812 |

## Best HPO trial

| key | value |
|---|---|
| `n_estimators` | `752` |
| `max_depth` | `5` |
| `learning_rate` | `0.29575536828663485` |
| `subsample` | `0.7884951933693297` |
| `colsample_bytree` | `0.6587687962141313` |
| `tree_method` | `hist` |
| `n_jobs` | `-1` |
| `early_stopping_rounds` | `30` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### interaction_heatmap

![interaction_heatmap](artifacts/interaction_heatmap.png)

### convergence

![convergence](artifacts/convergence.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)
