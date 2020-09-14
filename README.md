Introduction:
In this assignment, I am going to build a model to predict whether a customer will respond to the marketing campaign based on the training dataset. I'll use the model to predict the customer responses using the features provided in the test dataset.

In the training dataset, there are 7414 samples and 21 predictor feature. The target variable "responded" has two values, "yes" and "no". Thus, this project is a binary classification problem. In the exploratory analysis, the 21 predictors were separated into three categories: personal, bank activity ,and macro-economic data, and their statistics were examined to get some intuitive between the predictors and target.

In the data selection, the three highly correlated variables were dropped to improve the model (linear models) stability. The categorical features were converted to numerical data as the input for the model training. Before model training, the number of samples in the minority class was balanced using Synthetic Minority Oversampling Technique.

Three different algorithms were adopted for model training, logistic regression, random forest, and gradient boosting tree (XGboost). The model performance was evaluated based on the ROC_AUC score. The random forest and gradient boosting tree models' average performance were similar with ROC_AUC score near 0.8, but on the class responded with 'Yes' value, the gradient boosting tree model is slightly better. In the three models, the model performance was not so great in class responded with 'Yes'. The best recall score is only 0.23, which means only 23% of potential customers were correctly predicted, even though the weighted average recall is 0.88. There's still some potential to fine-tune the model to increase the prediction result in class 'Yes'.

The top important factors of predicting customer response are also identified based on the importance analysis. The strong indicators of the customer response are previous campaign outcome, previous contact, macro-economic, campaign month, and contact type.".

The assignment was intended to build a model that can be used to create some useful results rather than to obtain the finest model. To improve the prediction results, some experiments are worth testing:

Feature engineering to extract useful information from current predictors.
Test class weight parameters to improve the results in class responded "Yes".
Other models can also be tested, for example, Neural network.
Deliverables:
Provide the following:

The source code you used to build the model and make predictions.
The source code would be provided following the introduction and discussion.
A .csv file containing the predictions of the test data. You can add the target column (‘responded’) to the test data or simply provide it alone with the ‘id’ column.
The .csv file is sent along with this report by email.
Briefly answer the following questions:
a. Describe your model and why did you choose this model over other types of models?
I choose the gradient boosting tree model as the final model to predict based on the test data. The gradient boosting tree model is tree-based model and was build based on an ensemble of many individual trees. Each tree estimator was built sequentially that the next tree was build to minimize the residual after the current tree. The model and residual was updated with a weighted factor after each step.
In this assignment, the gradient boosting tree provided the best score in both training and validation data. The way gradient boosting tree builds its final model was able to extract useful information from the weak predictors. Though a lot of the time, the training of gradient boosting tree model is slow. The data size is small in this case, thus the training cost is not an issue.

b. Describe any other models you have tried and why do you think this model preforms better?
I have tried another two models, logistic regression and random forest. The logistic regression training is pretty fast, but the result score is the lowest among the three. logistic regression is more stable than tree based, but the trade off is the accuracy is not as good as the tree based model in this case. The predictors in the training dataset also have a lot of missing or unknown value which may cause problems in linear regression models but not in tree based models.

The random forest model actually provided very close result as the gradient boosting tree model. Both are tree based models, but in random forest, the trees are built simultaneously with subsample of the training sample and predictors, while the trees are built sequentially to target minimizing the residuals. Usually, the gradient boosting tree result in training dataset is better, but likely overfitting. Proper parameter tunning is needed to avoid the problem. In the end, in class responded "yes", the gradient boosting tree result is slightly better.

c. How did you handle missing data?
In both train and test datasets, three features contain missing data: custAge, schooling, and day_of_week. Among these three features, custAge is a numeric feature, while schooling and day_of_week are categorical features. By plotting the statistics of the custAge, I did not observe a strong relationship between the feature and the target value, thus it's acceptable to impute the missing data with the median value of custAge, which is 38. For the categorical features with missing data, I replaced the missing data with an additional category "missing_value", which is similar to the "unknown" value in those features.

d. How did you handle categorical (string) data?
I used one-hot-encoding to convert categorical data to numerical data which can be used in machine learning algorithms. In this assignment, almost all categorical data has no specific order related to the categorical values, thus it's reasonable to use one-hot-encoding. If there's inherent order associated with values in the categorical feature, label encoding may be used.

e. How did you handle unbalanced data?
In this assignment, the number of data with target value 0 is about 8 times the data with target value 1. I used oversampling based on Synthetic Minority Oversampling Technique (SMOTE). A random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.

f. How did you test your model?
The training data was split into two parts: training and validation with a ratio of 8:2. The models are training using K-fold methods to avoid overfitting. The parameters were tuned based on grid search or a Bayesian-based hyper-parameter tunning method. The ROC AUC score was used to evaluate the model since the imbalance of the sample numbers in a different class.
