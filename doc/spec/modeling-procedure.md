Procedure of power modeling:

step 1. build a power model foreach each level-1 module: 
1. calculate the variance on the raw dataset, drop the feature with 0-variance to the target.
2. classify all remaining features into four types described by the x-opm paper section III. according to the port-meaning document.
3. transform the type-D (data bus) features as the binary summation. For example if a type-D feature's value equals to 4'b1010 (binary) in a sample, you should transform it to 2 (unsigned int), because there are two 1s in the binary representation of that feature.
4. combine type-A and type-B as type-AB, type-C and type-D as type-CD. Loop each features in type-AB and each features in type-CD, construct type-AB * type-CD as new features and add to the dataset.
5. construct power model using GBDT based ont the dataset
6. construct power model using rulefit based on the dataset. For rulefit, negative coefficient is not allowed for the linear model, and the linear model does not have intercept.

step 2. build a GAM model 
1. The GAM model is built based on all the module level power model. The module level power model's coeffcients and parameters are fixed. One learnable coefficient is used to scale each module level power model's prediction magnitude. GAM has intercept. The GAM model is fit based on the top level module's power label
