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

![](/Visuals/visual_distribution_income_train.png)
* Training and test datasets have already been cleaned and divided for us.
* The distribution of our target variable 'Income' is distributed the same way in our training and test datasets. It is evenly unbalanced in both the training and the test sets. 
* Income '0' (<=50k) accounts for +/- 76% and Income '1' (>50k) for +/- 24%.


# Machine Learning
### Base model
First, we simply create a RandomForestClassifer model with the default parameters (random_state = 42). The results are already pretty good.

![](/Visuals/visual_0_roc.png)
<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_0_roc.png" width=50% height=50%>

| Metrics           | Score |  
| ----------------- | ----- | 
| Accuracy          | .8529 |  
| Balanced Accuracy | .7674 | 
| Mean CV           | .8581 | 
| AUC               | .90 | 

![](/Visuals/visual_0_confusion_matrix.png)
<img src="https://github.com/lvendrix/us-income/blob/main/Visuals/visual_0_confusion_matrix.png" width=50% height=50%>

|   | Precision | Recall | F1-Score | Support |
| - | --------- | ------ | -------- | ------- |
| 0 | .88       | .93    | .91      | 12435   |
| 1 | .73       | .61    | .66      | 3846    |

As we can see, this first model is great at identifying income '0' (income <=50k) but not that great at identifying income '1'. It's also important to use the metric 'Balanced Accuracy' as the target variable is unevenly distributed.

![](/Visuals/visual_features_importance.png)

In the previous chart, we see which features are most important for our model: finalweight, age and capital gain.

### Base model


![](/Visuals/Visual_3_features_original_gif.gif)

Now, we create 3 functions to iterate over all possible combinations:
* combination_features(df, number_features): Takes a dataframe and the desired number of features to be combined. Outputs a list
* score_features(df, list_combinations, max_number_clusters, random_state): Takes a dataframe, a list of combinations, a maximum number of cluster and a random state. Outputs a dictionary of the top-10 combination of features, based on the silhouette score.
* plot_features(df, dictionary): Takes a dataframe and a dictionary. Outputs a 2d scatter-plot if 2 features, a 3d scatter-plot if 3 features, and nothing if more than 3 features. Will also plot a silhouette plot. 

![](/Visuals/Visual_3_features_best_gif.gif)

# Results
Best silhouette scors sfor n features (KMeans++) using all the dataset
| # features | Clusters | Score |  
| ---------- | -------- | ----- |
| 2          | 2        | 0.956 | 
| 3          | 2        | 0.917 |
| 4          | 2        | 0.882 |
| 5          | 2        | 0.846 |
| 6          | 2        | 0.82  |

![](/Visuals/Visual_evolution_score.png)

# Conclusion 
As we can see, the best scores are always with a cluster of size 2, and the silhouette score keeps decreasing as we increase the number of features used to cluster.
It seems that we have 3 possible outliers that create their own cluster. By having values very dissimilar to the rest, it creates a nice separation between clusters. However, they are only 3 bearings in that cluster which is very unbalanced. We could also remove those from the dataset and redo the clustering search with the updated dataset.

# Further investigation
* Try other clustering methods and see if we have a better score
* Investigate possible 'outliers' affecting the performance of KMeans

# Timeline
09/08/2021 - 11/08/2021
