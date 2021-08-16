# Challenge Model Evaluation - US Income

## Mission objectives

Create a ML model that predicts whether a US citizen earns more than 50k/year based on several features (age, sex, native country, ...).
More information about the features can be found [here](https://archive.ics.uci.edu/ml/datasets/Census+Income).

# Installation

## Python version
* Python 3.9

## Packages used

* pandas
* numpy
* matplotlib.pyplot
* seaborn
* plotly.express
* sklearn

# Usage

| Filename                             | Usage                                                     |
|--------------------------------------|-----------------------------------------------------------|
| us_income_master.ipynb | Jupyter Notebook file containing Python code.<br>Used to create the RandomForestClassifier Model.<br>Used to finetune our model's hyperparameters. <br>Used to create charts.
| data_train.csv | csv file with training data.|
| data_test.csv | csv file with test data.|
| Visuals | Folder containing visuals.|

# First steps

<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_distribution_income_train.png" width=50% height=50%>

* Training and test datasets have already been cleaned and divided for us.
* The distribution of our target variable 'Income' is distributed the same way in our training and test datasets. It is evenly unbalanced in both the training and the test sets. 
* Income '0' (<=50k) accounts for +/- 76% and Income '1' (>50k) for +/- 24%.


# Machine Learning
### Base model
First, we simply create a RandomForestClassifer model with the default parameters (random_state = 42). The results are already pretty good.

<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_0_roc.png" width=60% height=60%>

| Metric            | Score |  
| ----------------- | ----- | 
| Accuracy          | .8529 |  
| Mean CV           | .8581 | 
| Matthew's coef    | .57   | 


<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_0_confusion_matrix.png" width=50% height=50%>

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .88       | .93    | .91      | 12435   |
| 1 | .73       | .61    | .66      | 3846    |

As we can see, this first model is great at identifying income '0' (income <=50k) but not that great at identifying income '1'. Case of overfitting. There are different metrics we can use. If only using the accuracy, we might think that out model is doing a great job, but using [Matthews correlation coefficient](
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) as well as the confusion matrix, we can better understand the performance of our model (whether or not the model is badly identifying '0' and '1').

### Features importance
![](/Visuals/visual_features_importance.png)

In the previous chart, we see which features are most important for our model: finalweight, age and capital gain.

## Improving the model
### Hyperparameters fine-tuning

To increase the performance of our model, we'll fine-tune its hyperparameters using GridSearch CV (with KFold = 3). The hyperparameters of the RandomForestClassifier we'll focus on are:
* n_estimators
* max_features
* min_samples_leaf

After fine-tuning, we find the following parameters, increasing the model's performance by +/- 1%.
### Best parameters 
* n_estimators: 90
* max_features: 3
* min_samples_leaf: 4

| Metric            | Score |  
| ----------------- | ----- | 
| Accuracy          | .8636 |  
| Mean CV           | .8638 | 
| Matthew's coef    | .579 | 

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .89       | .94    | .91      | 12435   |
| 1 | .77       | .61    | .68      | 3846    |

<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_1_confusion_matrix.png" width=50% height=50%>

### One-Hot Encoding & Hyperparameters fine-tuning

We'll One-Hot encode the features of our dataset which are categorical:     
* 'workclass'
* 'education'
* 'marital-status'
* 'occupation'
* 'relationship'
* 'race'
* 'sex'
* 'native-country'

Using GridSearch CV, we create a new RFC model which yields the following results. A small increase in performance.
* n_estimators: 110
* max_features: 10
* min_samples_leaf: 2

| Metric            | Score |  
| ----------------- | ----- | 
| Accuracy          | .8646 |  
| Mean CV           | .8643 | 
| Matthew's coef    | .603  | 

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .89       | .94    | .91      | 12435   |
| 1 | .76       | .62    | .68      | 3846    |

<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_one_hot_confusion_matrix.png" width=50% height=50%>

## Score evolution

<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_score_evolution.png" width=100% height=100%>

# Conclusion 
RandomForestClassifier already reaches a good accuracy score with default parameters. Using GridSearch CV to fine-tune the model did increase our score by a little. One-Hot encoding did also increase our model's performance by a little bit.
It's important to also look at the Confusion Matrix and Matthew score to see if any class (in this case '1') is badly identified.

# Timeline
13/08/2021 - 16/08/2021
