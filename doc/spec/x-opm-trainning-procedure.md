# X-OPM Training Procedure

## Data Preperation

### Rules
- Use @~/anaconda3 as the python interpreter. 
- The datasets are stored as `.pkl.zst`, the column contains multi-bit signals, they are features. The row contains per-cycle signal values, they are samples.
- All function and classes should be implemented in the python file called `data_preprocess.py` put under @src/xopm_lib

### Procedures  
1. Calculate the variance on the raw dataset, drop 
    - the feature with 0-variance w.r.t. the target.
    - the invariant features, i.e. the features do not flip during the whole benchmark, we cannot determine the amount of power incurred from its toggle if the feature does not toggle.
    - the duplicated features, means the raw signal columns that carry identical values across every training cycle. These are normally physical net dumped at multiple hierarchy depth.
2. classify all the features with non-zero variance into the 3 categories (a,b,c) based on the signal names (matched on the full hierarchy path, lowercased). When a signal matches more than one categories, apply priority **A > C > D**. The categories are:  
    a. Control signals: signal name contains `_en` or `_vld` or `_stall` or `_req` or `_busy` or `_idle` or (`clk` and `_en`).  
    b. Data signals: signal name contains `data`.  
    c. Configuration signals: all other signals that are not control or data signals.  
3. data transformation, perform the following steps. Note: the raw `*_func.pkl.zst` cells are per-cycle signal *states* (not toggles), so toggles are computed explicitly below, per case (no carry across benchmark boundaries).
    - For control signals: If a signal is multi-bit, you should split it into single-bit signal. For example if a signal is x[14:0], you should save x[14], x[13], ... x[0] in total 15 features instead of one feature x[14:0]. 
    - For control signals: If a signal name contains `_stall` or `_idle`, inverse the value of this signal (0 to 1, 1 to 0). Because these two signal's value is negative correlated to the power.
    - For data signals: calculate the **bit-toggle** by performing bitwise XOR between the `x[t+1]` and `x[t]`, where `t` is the simulation cycle (one cycle is one sample), `x[t]` is the signal value of cycle `t`. Then, calculate the bitwise hamming distance (hd) of the bit-toggle. For example: we have a signal `data[4095:0]`, the total simulation cycle is `1000`. first calculates its toggle by XOR between `data[4095:0]` and its delayed-one-cycle signal `data_dly1[4095:0]` (append 0 at the end to make it 1000 cycle), the result is `data_toggle[4095:0]`. Then we calculate the hamming distance of the `data_toggle`, get a 1000 cycle `data_toggle_hd[12:0]`. It is 13b because the largest hamming distance of `data` is 4096, which requires 13-bit to store it. Finally, use `data_toggle_hd[12:0]` to replace `data[4095:0]` in the trainning set.
4. Scale all features in the dataset to be between 0 to 1 by dividing the original interger signal value with its maximum value. For example a signal `x[63:0]`'s maximum value is $2^{64}-1$. You should scale x as $x/(2^{64}-1)$. Single-bit features are already 0/1 (max = 1). The `data_toggle_hd` feature is scaled by the bus width $W$ (its maximum possible hamming distance) so it lands in $[0,1]$.
5. Store the training and testing dataset as pandas dataframe in `.pkl` format. Each case and each type should be stored as a separate .pkl file. All pkl files should be stored under @dataset_processed/, seperately stored to `trainset` and `testset` folder.



## Feature Selection. 
not implemented




## Feature Interaction

### Rules
- RuleFit's source code can be found in @third_party/rulefit. Feature interaction corresponds to the `rule generation` step of the RuleFit. 
- The input dataset should be @dataset_processed/. Nothing from @dataset shall be used.
- All function and classes should be implemented in the python file called `model_regression.py` put under @src/xopm_lib


### Procedures
1. Train with RuleFit, with the following constrains: 
    - Introduce a dropout mechanism to prevent overfitting.
    - Use HPO (OpTuna) to search hyperparameters. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/dart.html 
    - Add a monotonic increasing constraint to the model, since increasing a feature represents an increase in toggling, which must correspond to increased power consumption. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
    - Allow only positive coefficients of the linear model, means we only allow features that are positively correlates to the power to be used.
2. Store the following results to @analysis/x-opm/{year-month-day-hour-minute} (create a folder if not exits, remove the old folder and create a new folder if folder exists)
    - After building the tree model using RuleFit, get the "rules", calculate the gain of each rule. Record the gain, the rule's name, how the rule is built, into a `rule.csv` file. Then eliminate the rules with low gain (setup an adjusable threshold for this). Also record the rules you dropped in the `rule.csv` with `dropped: True`. Record the linear model coefficients of those rules. The csv sorts according to the gain in descending order.
    - Friedman overall H-statistic plot, Friedman pairwise H-statistic plot.
    - SHAP beeswarm plot, SHAP interaction top pairs plot
    - After you trained the model and get the final result. Plot the residual map (train, val, test) of the final training result; plot the predict value/true value VS time scatter plot (train, val, test); 
    - Write a `report.md`, include the parameter you used, the dataset you used (dataset name and dimension), the analysis results, and trian/test/val R^2, MAPE, RMSE.




## Optional steps
1. [done by human] check the dropped features to see why it is dropped. If the dropped features are supposed to be related to the target, you need to further review your dataset to see if you have omitted some benchmarks so this signal is not toggled.
2. [optional] Build extra features: type-A*type-C, type-A*type-D. Store these features in a seperate dataset, and added this dataset to the training dataset.
3. Training with RuleFit @third_party/rulefit, with the following constrains 
    - Feature interaction only allows interaction between features from type-A and type-B. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
    - Set the ElasticNet's intercept (base score) to 0, i.e., assume that when x is all zeros, y = 0. Ref:  https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html
    - Set the tree model and Linear model's intercept (base score) to 0, i.e., assume that when x is all zeros, y = 0. Ref:  https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html