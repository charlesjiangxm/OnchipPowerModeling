For the .pkls directly under @c906_db_net_1cyc_20260729/aq_core/cp0, perform the training based on the following procedure. The source code should be saved to @x-opm directory, the results, intermediate files, dataset, reports should be saved to @out/x-opm directory. Use @~/anaconda3 as the python interpreter. 
1. Calculate the variance on the raw dataset, drop the feature with 0-variance w.r.t. the target.
2. classify all the features with non-zero variance into the 4 types (type-A, type-B, type-C, type-D). They are classified based on the signal names. Follow the following rules:
    - type A (control signals): signal name contains `_en` or `_vld` or `_stall` or `_req` or `_busy` or `_idle`. 
    - type B (clock gating signals): signal name contains both `clk` and `en`.
    - type C (data bus): signal name contains `data`.
    - type D (control and status payload): signals that is not type A, B, D are all type C.
3. data transformation, perform the following steps: 
    - For type A and type B signals: If a signal is multi-bit, you should divide it into single-bit. For example if a signal is x[14:0], you should save x[14], x[13], ... x[0] in total 15 features instead of one feature x[14:0]. This makes all type A abd type B signals single-bit.
    - For type A signals: If a signal name contains `_stall` or `_idle`, inverse the value of this signal (0 to 1, 1 to 0). Because these two signal's value is negative correlated to the power.
    - For type D signals: calculate the hamming distance of the signal toggle. For example: we have a signal `data[4095:0]`, the total simulation cycle is 1000. first calculates its toggle by XOR between `data[4095:0]` and its delayed-one-cycle signal `data_dly1[4095:0]` (append 0 at the end to make it 1000 cycle), the result is `data_toggle[4095:0]`. Then we calculate the hamming distance of the `data_toggle`, get a 1000 cycle `data_toggle_hamming[12:0]`. It is 13b because the largest hamming distance of `data` is 4096, which requires 13-bit to store it. Finally, use `data_toggle_hamming[12:0]` to replace `data[4095:0]` in the trainning set.
4. Scale all features in the training and testing dataset to be between 0 to 1 by dividing the original interger signal value with its maximum value. For example a signal x[63:0]'s maximum value is $2^{64}-1$. You should scale x as $x/(2^{64}-1)$.
5. save the intermediate files and reports
    - store a `.csv` table recording the features processed by the former steps. Including the feature name, feature type, the rule you follow to determine that signal belongs to that type, the data width, the min, max, mean value of this feature.
    - Store the training and testing dataset as pandas dataframe in .pkl format. Each case and each type should be stored as a separate .pkl file. All pkl files should be stored under @out/x-opm/dataset/, seperately stored to `trainset` and `testset` folder.

7. Training with RuleFit @third_party/rulefit, with the following constrains 
    - Add a monotonic increasing constraint to the model, since increasing a feature represents an increase in toggling, which must correspond to increased power consumption. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
    - Feature interaction only allows interaction between features from type-A and type-B. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
   
    - Introduce a dropout mechanism to prevent overfitting, and use HPO (OpTuna) to search hyperparameters. Ref: https://xgboost.readthedocs.io/en/stable/tutorials/dart.html 
    - Use a bagged tree model to train the model. Since bagging reduces the variance.
    - Set the tree model and Linear model's intercept (base score) to 0, i.e., assume that when x is all zeros, y = 0. Ref:  https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html
    - Allow only positive coefficients

8. Train the original dataset with @cobit as a reference. 
    

You should output the following analysis, save to @/home/jjiangan/disk/OnchipPowerModelingNew/output/cp0_xopm/analysis .  
1. After building the tree model using RuleFit, get the "rules", calculate the gain of each rule. Record the gain, the rule's name, how the rule is built, into a `rule.csv` file. Then eliminate the rules with low gain (setup an adjusable threshold for this). Also record the rules you dropped in the `rule.csv` with `dropped: True`.
2. Friedman overall H-statistic plot, Friedman pairwise H-statistic plot.
3. SHAP beeswarm plot, SHAP interaction top pairs plot
4. After you trained the model and get the final result. Plot the residual map (train, val, test) of the final training result; plot the predict value/true value VS time scatter plot (train, val, test); Save a `coefficient.csv` file record the trained rule, the coefficient value of that rule (sort in descend order).
4. Write a `report.md`, include the parameter you used, the dataset you used (dataset name and dimension), the analysis results, and trian/test/val R^2, MAPE, RMSE.




Optional steps
2. [done by human] check the dropped features to see why it is dropped. If the dropped features are supposed to be related to the target, you need to further review your dataset to see if you have omitted some benchmarks so this signal is not toggled.
6. [optional] Build extra features: type-A*type-C, type-A*type-D. Store these features in a seperate dataset, and added this dataset to the training dataset.
