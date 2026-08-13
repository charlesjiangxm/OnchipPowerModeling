/plan 

For the .pkls directly under @/home/jjiangan/disk/OnchipPowerModelingNew/c906_db_net_1cyc_20260729/aq_core/cp0, perform the training based on the following procedure. The source code should be saved to @/home/jjiangan/disk/OnchipPowerModelingNew/x-opm . The results should be saved to @/home/jjiangan/disk/OnchipPowerModelingNew/output/cp0_xopm. You should submit the computation jobs through slurm, do not use local machine to do computation. 
1. Calculate the variance on the raw dataset, drop the feature with 0-variance w.r.t. the target.
3. [done by human or agent] classify all the features with non-zero variance into the 4 types (type-A, type-B, type-C, type-D) defined by the paper @/home/jjiangan/disk/OnchipPowerModelingNew/doc/x-opm.pdf .
4. If a signal is a multi-bit data bus, calculate the hamming distance of the signal toggle. For example: we have a data signal `data[4095:0]`, the total simulation cycle is 1000. first calculates its toggle by XOR between `data[4095:0]` and its delayed-one-cycle signal `data_dly1[4095:0]` (append 0 at the end to make it 1000 cycle), the result is `data_toggle[4095:0]`. Then we calculate the hamming distance of the `data_toggle`, get a 1000 cycle `data_toggle_hamming[12:0]`. It is 13b because the largest hamming distance of `data` is 4096, which requires 13-bit to store it. Finally, use `data_toggle_hamming[12:0]` to replace `data[4095:0]` in the trainning set.
5. Standardize all features in the training and testing dataset to be between 0 to 1.
6. Store the training and testing dataset as pandas dataframe in .pkl format. Each case and each type should be stored as a separate .pkl file. All pkl files should be stored under @/home/jjiangan/disk/OnchipPowerModelingNew/output/cp0_xopm/trainset and @/home/jjiangan/disk/OnchipPowerModelingNew/output/cp0_xopm/testset .
7. Training with RuleFit: /home/jjiangan/disk/OnchipPowerModelingNew/third_party/rulefit 

The following constrain should be used during the training:
1. Introduce a dropout mechanism to prevent overfitting, and use HPO (OpTuna) to search hyperparameters. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/dart.html 
2. Add a monotonic increasing constraint to the model, since increasing a feature represents an increase in toggling, which must correspond to increased power consumption. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
3. Feature interaction constraints: specify that feature interactions between C and D should not be allowed. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
4. Use a bagged tree model to train the model. Since bagging reduces the variance.
5. Set the tree model's intercept (base score) to 0, i.e., assume that when x is all zeros, y = 0. Ref:  https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html

You should output the following analysis, save to @/home/jjiangan/disk/OnchipPowerModelingNew/output/cp0_xopm/analysis .  
1. After building the tree model using RuleFit, get the "rules", calculate the gain of each rule. Record the gain, the rule's name, how the rule is built, into a `rule.csv` file. Then eliminate the rules with low gain (setup an adjusable threshold for this). Also record the rules you dropped in the `rule.csv` with `dropped: True`.
2. 使用 Python 的 shap 库（如 shap.TreeExplainer）画出 SHAP Summary Plot，说明特征影响的大小和正负方向
3. 计算 Friedman's H-statistic 和 SHAP Interaction Value使我可以确认feature间是否存在复杂的协同效应
4. After you trained the model and get the final result. Plot the residual map (train, val, test) of the final training result; plot the predict value/true value VS time scatter plot (train, val, test); Save a `coefficient.csv` file record the trained rule, the coefficient value of that rule (sort in descend order).
4. Write a `report.md`, include the parameter you used, the dataset you used (dataset name and dimension), the analysis results, and trian/test/val R^2, MAPE, RMSE.




Optional steps
2. [done by human] check the dropped features to see why it is dropped. If the dropped features are supposed to be related to the target, you need to further review your dataset to see if you have omitted some benchmarks so this signal is not toggled.
6. [optional] Build extra features: type-A*type-C, type-A*type-D. Store these features in a seperate dataset, and added this dataset to the training dataset.
