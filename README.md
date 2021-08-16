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
| Balanced Accuracy | .7674 | 
| Mean CV           | .8581 | 
| AUC               | .90 | 

<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_0_confusion_matrix.png" width=50% height=50%>

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .88       | .93    | .91      | 12435   |
| 1 | .73       | .61    | .66      | 3846    |

As we can see, this first model is great at identifying income '0' (income <=50k) but not that great at identifying income '1'. It's also important to use the metric 'Balanced Accuracy' as the target variable is unevenly distributed.

![](/Visuals/visual_features_importance.png)

In the previous chart, we see which features are most important for our model: finalweight, age and capital gain.

## Improving the model
### Hyperparameters fine-tuning

To increase the performance of our model, we'll fine-tune its hyperparamaters using GridSearch CV (with KFold = 3). The hyperparameters of the RandomForestClassifier we'll focus on are:
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
| Balanced Accuracy | .7753 | 
| Mean CV           | .8638 | 
| AUC               | .91   | 

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .89       | .94    | .91      | 12435   |
| 1 | .77       | .61    | .68      | 3846    |


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

| Metric            | Score |  
| ----------------- | ----- | 
| Accuracy          | .8655 |  
| Balanced Accuracy | .78 | 
| Mean CV           | .8641 | 
| AUC               | .92   | 

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .89       | .94    | .91      | 12435   |
| 1 | .77       | .62    | .68      | 3846    |

# Conclusion 
As we can see, the best scores are always with a cluster of size 2, and the silhouette score keeps decreasing as we increase the number of features used to cluster.
It seems that we have 3 possible outliers that create their own cluster. By having values very dissimilar to the rest, it creates a nice separation between clusters. However, they are only 3 bearings in that cluster which is very unbalanced. We could also remove those from the dataset and redo the clustering search with the updated dataset.

# Further investigation
* Try other clustering methods and see if we have a better score
* Investigate possible 'outliers' affecting the performance of KMeans

# Timeline
09/08/2021 - 11/08/2021
