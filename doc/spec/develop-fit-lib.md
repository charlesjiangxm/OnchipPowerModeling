``### Requirements

Refactor the code in @src folder, create a python library for fitting regression models based on the .pkl files . All the source file goes to the @src folder, all the input parameters should be stored as a YAML file in @config folder. Create an user interface script under @script for user to pass the path of YAML configurations they want to use and kickstart the model fitting main procedure, suggest `python script/run_fit.py config configs/<name>.yaml` 

### User Interfaces

Below lists the parameters needs to be supported by the YAML configuration file. The parameters should be grouped into multiple sections to be easily decoupled.

#### Section 1. General parameters:

1. trainset_x_path
   - type: a list of string
   - meaning: The list of .pkl files used as the training feature (x) for fitting the model . Each .pkl file corresponds to a power simulation benchmark case. 

2. testset_x_path
   - type: a list of string
   - meaning: The list of .pkl files used as the testing feature (x) for fitting the model . Each .pkl file corresponds to a power simulation benchmark case.

3. y_path
   - type: a list of string
   - meaning: The list of .pkl files used as the label (y) for fitting the model . Each .pkl file corresponds to a power simulation benchmark case.
   
4. y_label
   - type: string
   - meaning: .pkl in each y pkl may contain multiple columns, but only one column is used as the label for fitting the model. y_label is the name of the column used as the label.

5. seed
   - type: int
   - meaning: The random seed for reproducibility.

#### Section 2. data preprocessing related parameters

5. drop_zero_var  
   - type: bool
   - meaning: if this is true, drop all the columns of x dataframe that has zero variance. Implement this feature with sklearn VarianceThreshold.

6. train_val_test_ratio 
   - type: (float, float, float), each float is from 0-1, three float should adds up to 1.
   - meaning: If the test_x_path list is empty, this ratio means the split ratio of train set, validation set and the test set, e.g. 0.8:0.1:0.1, based on the trainset_x. If the test_x_path is not empty, the train set : validation set ratio is e.g. 0.8:(0.1+0.1). The test set comes from test_x_path. The validation set is used for early-stopping for FT-Transformer/MLP/GBDT, or per-algorithm HPO. The val set is not included in the final standardizer fit. for the random split, cut one continuous data out starting from a seed-controlled random position per-benchmark, then concatenate them. 

7. avg_wsize 
   - type: int, > 0
   - meaning: the window size for performing non-overlapped averaging on the sample data. drop the tail bin if its dimension does not match with others.

#### Section 3. The feature selection  

8. fea_sel_alg
   - type: str|None
   - meaning: the feature selection algorithm supported in @feature_selectors.py. If None, bypass the feature selection.

9. top_k
   - type: int, > 0
   - meaning: The number of features selected. 

10hyper-parameters for each feature selection algorithms, group each algorithm's hyperparameters into a dictionary for clean interface.

#### Section 4. The model fitting   

10. rgr_alg
    - type: str
    - meaning: the regression algorithms supported, can be chosed from FT-Transformer, RuleFit, GBDT, MLP and ElasticNetCV.

11. intercept_on
    - type: bool
    - meaning: whether do we allow to fit the intercept. Only RuleFit, ElastiveNetCV supports it. 

12. non_negative_coef_only
    - type: bool
    - meaning: whether do we only allows the model to use non-negative model coefficients. Only RuleFit and ElastiveNetCV supports it. For RuleFit, if this parameter is True, than you should let the Lasso algorithm in RuleFit to only allow non-negative coefficients.

13. hpo_timeout
    - type: int, >0
    - meaning: the maximum time for performing hyper-parameter tunning for the model fitting.

13. hyper-parameters for each regression algorithms, group each algorithm's hyperparameters into a dictionary for clean interface.

### Main Procedure

Data preprocessing:
1. Log all parameters and hyperparameters input by the user as fit.log to `output/<config_stem>_<timestamp>/`.
2. List and log the matched x and y pair. Match files based on the filename stem (e.g. MMU_func.pkl <-> MMU_pwr.pkl) - strip suffix _func from x stems and _pwr from y stems and match by remainder. Concatenate the pkl files in the training set, and testing set, respectively, to form the training x, y and the testing x, y dataset.  
3. Get the list of zero-variance columns named in training x. Remove those columns in both training x and testing x dataset.
4. Perform non-overlapped averaging on the sampling dimension of training x, testing x and y.
5. Perform standardization on the training set x and y. Record the mean/std, and applies the standardization with same parameters to the testing set. compute metrics in the original y scale — inverse-transform ŷ and y before sMAPE/RMSE.   
6. Save the `train_x.pkl, train_y.pkl, val_x.pkl, val_y.pkl, test_x.pkl, test_y.pkl` after data preprocessing as .pkl files. Saved to `output/<config_stem>_<timestamp>/`.  

Feature selection:  
1. Apply the feature selection algorithm to the training x. After you get the selection rule, apply this rule to the testing x.
2. Save the `train_x.pkl, train_y.pkl, val_x.pkl, val_y.pkl, test_x.pkl, test_y.pkl` after the feature selecion as .pkl files. Saved to `output/<config_stem>_<timestamp>/`.

Model regression:
1. Apply the model regression algorithm the user picks. If cuda is available, use CUDA for MLP and FT-Transformer, else if mps is available, use MPS, else use CPU.
2. Perform automatically hyperparameter searching with sklearn `*SearchCV` (RuleFit / GBDT / ElasticNetCV) or `Optuna` (MLP & FT-Transformer) based on maximizing R^2. Calculate single trial-score on the held-out val set. The timeout is defined by user, and you should use n_jobs=-1 for sklearn SearchCV; for Optuna on GPU/MPS, run trials sequentially (n_jobs=1). Help me visualize the trend of hyperparameter searching: for *SearchCV, plot cv_results_["mean_test_score"] vs trial; for Optuna, dump plot_optimization_history + plot_param_importances.
3. Save the top 20 rules (only RuleFit), features, the feature interaction heat map (rule co-occurrence for RuleFit, attention for FT-Transformer, SHAP and Friedman H for GBDT, N/A for MLP and ElasticNetCV), the prediction vs groud truth residual map, the training convergence curve (for GBDT, MLP, FT-Transformer). Saved to `output/<config_stem>_<timestamp>/`.

Outputs:
1. Write a markdown file to summarize the training results. Includes the number of samples and features of the training x, testing x and y after the data preprocessing and the feature selection. Includes the sMAPE,RMSE,R2 of the training and testing dataset, and visualizing the results (use link to the generated figures). 

### Exception Handling
1. If train/test/val (x) pkls have different features, report a Fatal and stop the training.
2. If the y label dimension does not match with any x, it's a fatal error, report it and stop the training.
3. If y-label cannot be found in y, it's a fatal error.

### Migrate from the Old Code
1. The old FeatureSelector have code to re-standardize the data, you should avoid this re-standardization.
2. The old code supports time ordered training, don't support it in the new code. New code only supports concatenated training.
3. The old code implements Ridge, but I want ElasticNetCV
