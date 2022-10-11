#!/usr/bin/env python
# coding: utf-8

# 
# ## Data Exploration and Machine Learning
# 
# _Using ML & Stats Models as a Scoring Methods | North Star **Order Completed to Subsequent Order Completed** Case Study_
# 
# ***
# 
# ## Table of Contents
# 
# * [Introduction](#intro)
# * [General Outline](#outline)
# * [Import Required Libraries](#import)
# * [Load Data](#load)
# * [Explore Data - EDA](#eda)
#     * [Check Data](#check)
#     * [Select Features & Clean Data](#select)
#     * [Descriptive Statistics](#stats)
#     * [Check Target for Imbalanced Classes](#imbalance)
#     * [Univariate EDA](#uni)
# * [Prepare Data for Machine Learning](#ml)
#     * [Create Dummy Variables for the Categorical Features](#dummy)
#     * [Shuffle](#shuffle)
#     * [Split to Train/Test](#split)
#     * [Scale Numerical Features](#scale)
# * [Train Models](#train)
#     * [Model Evaluation](#metrics)
#     * [Select Features with Chi_Squared](#chi2)
#     * [Select Features with RFECV](#rfecv)
#     * [Hyper-Parameter Tuning with Grid Search](#gridsearch)
#     * [Select Best Model with Grid Search and Pipeline](#best)
#     * [Examine Grid Search Results](#grid-results)
#     * [Train Best Combinations & Plot Results](#plot-best)
#     * [Tune Best Model](#tune-best)
# * [Evaluate Best Model with Confusion Matrix](#cm)
# * [Predict Probabilities and Evaluate with ROC-AUC Curve](#roc-auc)
# * [Cut-off Analysis](#cut-off)
# * [Export Data](#export)
# 
# 
# The objective of this study is to apply statistical and machine learning (ML) models like logistic regression classifier in a LRX dataset to indicate which attributes are most important. Specifically, we want to evaluate the timeline of activities that took place to understand what should be a good starting point (as a baseline assumption) to build a statistically sound **Order Completed to Subsequent Order Completed** scoring model. The dataset is already filtered to only show people who had this specific conversion (investment account to order completed to subsequent order completed conversions). There are 3 types of values that we are interested:
# 
# 1. **non-conversion** - this activity took place during a session on a given day but this did not take at the same time the conversion took place (a session is a group of user interactions with a website that take place within a given time frame. For example, a single session can contain multiple page views, events, social interactions, and transactions. A single user can open multiple sessions).
# 2. **touchpoint** - this activity took place before the conversion event (order completed) took place.
# 3. **conversion** * these are the activities that took place at the time of the conversion itself.
# 
# 
# For the implementation of the solution the following steps will be followed:
# * Data exploration (EDA) to identify problems (missing values, types, imbalances, etc.), reveal hidden patterns and relationships.
# * Investigate various statistical and machine learning models - preferably logistic regression and tree-based ones, which can be interpreted easily.
# * Create a scoring system from the model's probabilities by
#     * user-level
#     * profile level
# * Examine the most important factors that make a user to proceed to a purchase using various advance ML techniques, such as the Wrapper, Filter, and Embedded methods.
# 
# The general outline of this study can be summarised as follows:
# 
# 1. Load data
# 2. Explore data
# 3. Preprocess data
# 4. Test and validate logistic regression
# 5. Evaluate models
# 6. Create scoring system from model's probabilities
# 7. Find most significant features.
# 



# Import the required libraries.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

# Set Plotly theme.
pio.templates.default = "gridon"

# Set global variables.
RANDOM_STATE = 5 # set a seed so as to have reproducibility in the analysis.

get_ipython().run_line_magic('matplotlib', 'inline')


# Read the dataset from CSV and load it into a Pandas dataframe. Import only the fields we need for this analysis:
# 
# 1. email
# 2. event
# 3. campaignName
# 4. group_type
# 5. conversion_name - the target (dependent) variable



# Read data.
data = pd.read_csv('order_completed_to_subsequent_order_completed.csv',
                    parse_dates=['touchpointDateTimes'],
                    infer_datetime_format=True,
                    usecols=['touchpointDateTimes', 'email', 'event',
                             'campaignName', 'group_type', 'conversion_name'])

# Show first rows.
data.head(2)


# 
# The Exploratory Data Analysis or EDA include the following steps:
# 
# * Review the available data and select specific variables of interest.
# * Check the quality of data.
# * Check for imbalances and create charts.
# 
# The dataframe will be examined for the quality of the data. The types and shape of the data will be checked, as well as if there are any missing or duplicated records.



# Create a function to check the data.
def check_data(df): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()]).T.squeeze()
    duplicates = df.duplicated().sum()
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = round((df.isnull().sum()/ obs) * 100, 2)
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape: ', df.shape)
    print('Duplicates: ', duplicates)
    display(df.describe())
    frame = {'types': types, 'counts': counts, 'uniques': uniques, 'nulls': nulls, 'distincts': distincts,
             'missing_ratio': missing_ratio, 'skewness': skewness, 'kurtosis': kurtosis}
    checks = pd.DataFrame(frame)
    display(checks)

check_data(data)


# 
# * The selection of the features has already implemented during the CSV data reading.
# * Filter out the touchpoint events and keep only the conversions.
# * Remove the `Impression catcher` and `Direct Visits` from campaign name.
# * Group by user and timestamp. The first timestamp of each user will be the first order, the next timestamps will be the subsequent orders.
# * The target (dependent) binary variable will be:
#     * `first_order` (mapped to 0) and
#     * `subsequent_orders` (mapped to 1).



# Drop any rows with at least one blank.
data.dropna(inplace=True)

# Filter out the touchpoint events and keep only the conversions.
data = data[data['event'] == 'conversion']

# Remove the "Impression catcher" and "Direct Visits" from campaign name.
data = data[data['campaignName'] != 'Impression catcher']
data = data[data['campaignName'] != 'Direct Visits']

# Sort values.
data.sort_values(['email', 'touchpointDateTimes'], inplace=True)

# Create a new df with the first orders only - get the earliest date with argmin().
first = data.groupby(['email']).agg(lambda x: x.iloc[x.touchpointDateTimes.argmin()])

# Init a new field "Orders" for the engineered target variable.
# This should have the value 0 - this is for the first order.
first['Orders'] = 0

# Reset index and keep only the 'email', 'touchpointDateTimes', and 'Orders'.
first.reset_index(inplace=True)
first = first[['email', 'touchpointDateTimes', 'Orders']]

# Join the two dfs.
df = pd.merge(data, first,  how='left',
              left_on=['email', 'touchpointDateTimes'],
              right_on = ['email', 'touchpointDateTimes'])

# Fill "Orders" NaNs with 1. These are the subsequent orders.
df.fillna(1, inplace=True)

# Convert target to category variable.
df['Orders'] = df['Orders'].astype('category')

# Select the features we need.
df = df[['email', 'campaignName', 'group_type', 'Orders']]

# Let's check again.
check_data(df)


# **Inference**
# 
# * Most of the data were removed, but now we have a clean dataset with no missing values.
# * We have about ~9K rows - the training examples.
# * We will use 2 features/attributes (`campaignName` and `group_type`), and
# * 1 target variable (`Orders`).
# 
# Descriptive statistics of the numerical continuous variables are useful to get a better feeling of our data, e.g. check the measures of central tendencies mean, median, or mode, check min and max values and quantiles of each feature applying the Pandas describe function.
# 
# In our case , we didn't keep any numerical attribute. However, we can group the various categorical features and plot the counts for each category.



# Create a list of features.
features = ['campaignName', 'group_type']

for feat in features:
    # Show counts for each feature.
    fig = px.bar(df.groupby(feat).count().reset_index().sort_values(by=['email']),
                 x='email', y=feat, text='email',
                 opacity=0.6, orientation='h', height=900)
    fig.update_layout(title_text="Distribution of " + feat)
    fig.update_xaxes(showgrid=False, title_text=None)
    fig.update_yaxes(showgrid=False, title_text=None)
    fig.update_yaxes(showticklabels=True, automargin=True)
    fig.show()


# **Inference**
# 
# * We can see the most common categories of each feature.
# 
# Imbalanced classes put **accuracy** out of business. This is a surprisingly common problem in machine learning classification, occurring in datasets with a disproportionate ratio of observations in each class. Standard accuracy no longer reliably measures performance, which makes model training much trickier. Suppose we have two classes - A and B. Class A is 90% of your data-set and class B is the other 10%, but we are most interested in identifying instances of class B. We can reach an accuracy of 90% by simply predicting class A every time, but this provides a useless classifier for our intended use case. Instead, a properly calibrated method may achieve a lower accuracy, but would have a substantially higher true positive rate (or recall), which is really the metric we should have been optimizing for.



# Check visually target for imbalance.
fig = px.bar(df.groupby('Orders').count().reset_index(),
             x='Orders', y='email', text='Orders', opacity=0.6)
fig.update_layout(title_text="Distribution of Orders")
fig.update_xaxes(showgrid=False, title_text=None)
fig.update_yaxes(showgrid=False, title_text=None)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext = ['First Orders', 'Subsequent Orders']
    )
)
fig.show()

print(str(round((sum(df['Orders'])/len(df['Orders'].index)) * 100, 2)) + '% conversion')


# **Inference**
# 
# * The dataset is imbalanced.
# 
# Let's inspect where converted users come from.



for feat in features:
    # Count converted users by feature.
    feat_df = df.groupby([feat, 'Orders']).count().reset_index()

    # Plot converted users by feature.
    fig = px.bar(feat_df, x='email', y=feat, color='Orders',
                 opacity=0.6, barmode='relative', orientation='h', height=900)
    fig.update_layout(title_text="Distribution of Converted Users by " + feat,
                      yaxis={'categoryorder':'total ascending'})
    fig.update_xaxes(showgrid=False, title_text=None)
    fig.update_yaxes(showgrid=False, title_text=None)
    fig.update_yaxes(showticklabels=True, automargin=True)
    fig.show()


# **Inference**
# 
# * We can see the most common categories of each feature by the target variable.
# 
# 
# Now, let’s move on to attempting to predict the target variable with ML techniques. For this case, we will be using the train dataset and then we will test the algorithms on the test data. The appropriate order is:
# 
# 1. Transform categorical features to binary indicator variables.
# 2. Split to train and test datasets.
# 3. Scale numerical features.
# 
# **Important Note:** Applying the feature scaling before train/test splitting arises a problem; if we scale features for all the dataset and use the scaled features for cross-validation, then the test fold already contains the info about training set as the whole training set (X_train) was used for scaling.
# 
# Most ML algorithms cannot operate on label/categorical data directly. They require all input variables and output variables to be numeric. Hopefully the `Pandas` library offers the `get_dummies` function, which creates a new dataFrame with [binary indicator variables](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) for each category in the columns specified. When `get_dummies` is called, the result is a dataframe with $n$ columns, each of which describes one of the levels/values of the initial column. Once that's done, the new dummy columns are merged (concatenated) into the original dataset and the initial columns, which are no longer needed, are removed. However, in our case we will keep for now the initial columns, so as to use it later to revert back to the original dataset, but we will remove them during the training stage.
# 
# We create dummies only for categorical variables to avoid order. We also enable `drop_first=True` to avoid dummy variable [trap](https://medium.com/datadriveninvestor/dummy-variable-trap-c6d4a387f10a) and then concatenate the dataframes.



# Encode categorical data with Pandas "get_dummies".
df = pd.concat([df, pd.get_dummies(df[features], drop_first=True)], axis=1)

# Show first rows.
# 
# We shuffle the data to make sure that our training/test/validation sets are representative of the overall distribution of the data. Shuffling data serves the purpose of reducing variance and making sure that models remain general and overfit less.



# Shuffle dataframe rows and reset index.
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# Divide data into X and y variables and then split to a train and test set. The test dataset will be used to evaluate the machine learning algorithms.
# 
# * **Note:** According to Andrew Ng, when working with big datasets (over a million training samples) the dev/test sets can be very small compared to the train data, for example 10%, 1% or even 0.1% of all the data. Let's use a 10% split in our case.



# Divide data into X and y variables.
y = df['Orders']

# Remove the initial cols used in get_dummies, useless cols, and the target variable.
X = df.drop(labels=['email', 'Orders'] + features, axis=1)

# Split to train and test datasets in a stratified fashion.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=df.Orders,
                                                    random_state = RANDOM_STATE)

print("Number of training set: %d\nNumber of testing set: %d\nTotal: %d"      % (len(X_train), len(X_test), (len(X_train)+len(X_test))))



# Features that are measured at different scales do not contribute equally to the model fitting and might end up creating a bias. Thus, to deal with this potential problem feature normalization or standardization as min-max scaling is usually used prior to model fitting. We don't have any numerical variables in our case, so there is no need to do anything here.



# # Build the scaler model.
# norm = MinMaxScaler()

# # Fit and transform only the train set to prevent data leakage to the test set.
# df_train[['']] = norm.fit_transform(df_train[['']])

# # Transform the test test.
# df_test[['']] = norm.transform(df_test[['']])


# Now, we are going to use various statistical models from `statsmodels` and scikit-learn. A common mistake is to learn the parameters of a prediction algorithm and testing it on the same data. What's wrong with that? We cannot fit the model to our training data and hope it would accurately work for the real data it has never seen before.
# 
# To avoid this to happen, there are several techniques: we could remove a part of the training data and using it to get predictions from the model trained on rest of the data (= Holdout Method). But, by reducing the training data, we risk losing patterns in data set and increase the error. The k-fold cross validation will help us to solve this problem.
# 
# In k-fold [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html), we split our data into k separated "folds". Then, the Holdout Method is repeated k times, such as each time, one of the k folds will be the test subset and the (k-1) other folds will be used together as the training set. In our multi-class case, we will use a variation of k-fold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
# 
# Note that this method does not depend on the model. In this example, we will use it on classification algorithms, such as logistic regression, but we could use it on any methods we want (Linear Regression). The general workflow to apply the Cross Validation is always the same:
# 
# 1. Instanciate the model from scikit-learn.
# 2. Instanciate the `StratifiedKFold` class with the parameters we want.
# 3. Use the `cross_val_score()` function to measure the performance of the model.
#     1. Use the cross_val_score() function to do the cross validation of k-folds.
#     2. Use the LogisticRegression instance we just created.
#     3. Use the feature columns `X_train` for the training and the target column as the `y_train`.
#     4. Use the scoring parameter with various metrics.
#     5. Return an array with the score values (one for each fold).
#     6. Assign the result to the variable `scores`.
#     
# ### Model Evaluation <a id='metrics'></a>
# 
# Evaluating a machine learning algorithm is an essential part of any project. A model may give satisfying results when evaluated using accuracy, but may give poor results when evaluated against other metrics such as recall or precision. Some of the different types of evaluation metrics available are:
# 
# * Accuracy: This is the ratio of number of correct predictions to the total number of input samples. However, the fraction of correct predictions is typically not enough information to evaluate a model. Although it is a starting point, it can lead to invalid decisions. Models with high accuracy may have inadequate precision or recall scores.
# 
# * Precision: This is the number of correct positive results divided by the number of positive results predicted by the classifier. In other words, this is the ability of the classifier not to label as positive a sample that is negative. The best value is $1$ and the worst value is $0$.
# 
# * Recall: This is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). In other words, this is the ability of the classifier to find all the positive samples. The best value is $1$ and the worst value is $0$. In context to our study, recall shows how well our identifier can find the correct email that opens the link. For example, a low recall score of $0.4$ indicates that our identifier predicts only $40\%$ of the real. The rest $60\%$ cannot not be found by our model.
# 
# * F1 Score: This is the Harmonic Mean (a weighted average) between precision and recall. The range for F1 Score is [0, 1]. This scores tells how precise a classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances). High precision but lower recall, gives an extremely accurate prediction, but it then misses a large number of instances that are difficult to classify. The greater the F1 Score, the better is the performance of our model. The formula for the F1 score is:
# 
# $$F1 = 2 \times \frac{precision \times recall}{precision + recall}$$
# 
# **Note:** Some metrics are essentially defined for binary classification tasks (e.g. f1_score, roc_auc_score). In extending a binary metric to multiclass or multilabel problems, the data is treated as a collection of binary problems, one for each class. For more look at sklearn [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#average)
# 
# We use `SelectKBest` to select the features with best chi-square. We pass two parameters one is the scoring metric that is `chi2` and the other is the value of `K` which signifies the number of features we want in final dataset. 


# Select 50 features.
select_feature = SelectKBest(chi2, k=50).fit(X_train, y_train)

# Save the selected features.
X_train_chi = select_feature.transform(X_train)
X_test_chi = select_feature.transform(X_test)

# Print the features and scores.
selected_features_df = pd.DataFrame({'Feature':list(X_train.columns), 'Scores':select_feature.scores_})




# Create a function to generate scores.
def generate_scores(model, X, y):
    y_pred = model.predict(X)
    confusion = metrics.confusion_matrix(y, y_pred)
    print(confusion)
    print('F1: ', metrics.f1_score(y, y_pred))
    print('Recall: ', metrics.recall_score(y, y_pred))
    print('Precision: ', metrics.precision_score(y, y_pred))
    print('ROC-AUC: ', metrics.roc_auc_score(y, y_pred))




# Fit logistic regression with selected features from chi2.
lr = LogisticRegression()     
model = lr.fit(X_train_chi, y_train)

# Print results.
generate_scores(model, X_test_chi, y_test)


# **Inference**
# 
# * Let's try a more advanced ML technique like RFE.
# 
# Recursive Feature Elimination (or RFE) is a wrapper method, which works by recursively removing attributes and building a model on those attributes that remain. For example, for data with n features, the process of RFE is the following:
# 
# 1. n−1 models are created with combination of all features except 1. The least performing feature is removed
# 2. n−2 models are created by removing another feature and so on.
# 
# There is also the Recursive Feature Elimination with Cross-Validated election (RFECV) that performs RFE in a cross-validation loop to find the optimal number of features. Let's use this more advanced method, which gives better quality results.


# Create the RFECV object and use an appropriate metric for imbalanced datasets.
metric = 'precision'

# Create a model classifier for the RFECV.
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Create a RFECV object.
rfecv = RFECV(estimator=dt, step=1, cv=5, scoring=metric, n_jobs=-1)

# Train.
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])



# Create a df with the features selected by RFECV.
RFECV_features = pd.DataFrame({'feature' : X_train.columns, 'rank' : rfecv.ranking_,
                               'support' : rfecv.support_})
condition = RFECV_features['support'] == True 
rfecv_features = RFECV_features[condition].sort_values(by='rank', ascending=True)



# Save the selected features to X train and test in order to fit a model.
cols = rfecv_features['feature'].values
X_train_rfecv = X_train[cols]
X_test_rfecv = rfecv.transform(X_test)

# Fit logistic regression with selected features from RFECV.
model = dt.fit(X_train_rfecv, y_train)

# Print results.
generate_scores(model, X_test_rfecv, y_test)


# **Inference**
# 
# * Next, we will try several classification models from Sklearn using `Pipeline` and `GridSearchCV`.
# 
# ### Hyper-Parameter Tuning with Grid Search <a id='gridsearch'></a>
# 
# One of the most common reasons to do cross validation is to fine tune hyper-parameters. Hyper-parameters are set by the programmer whereas parameters are generated by the model. Most of the learning algorithms require some parameters tuning. It could be the number of trees in Gradient Boosting classifier, hidden layer size or activation functions in a Neural Network, type of kernel in an SVM and many more. We want to find the best parameters for our problem. We do it by trying different values and choosing the best ones. There are many methods to do this. It could be a manual search, a grid search or some more sophisticated optimization. However, in all those cases we can’t do it on our training test and not on our test set of course. We have to use a third set, a validation set.
# 
# Grid searching is the process of testing different parameter values for a model and selecting the ones that produce the best results. The steps are:
# 
# 1. Load data.
# 2. Make a parameter dictionary.
# 3. Initiate a GridSearch algorithm.
# 4. GridSearch the data.
# 5. Print the results.
# 
# Hyper-parameters can be tuned using grid search and pipeline. In our study we will also use the pipeline method, which allows the searching of multiple algorithms with many hyper-parameters each. It is a very code efficient way of testing many models in order to select the best possible one. Additionally, pipelining can handle pre-processing tasks as well, allowing for further control of the process.
# 
# **Notes:**
# 
# - Since, this task requires a vast amount of time, let's keep it simple and search only a few hyper-parameters. Later, when we find the best classifier, we can search more thorougly the hyper-parameters.
# - We prefer to use decision trees with limited number of trees, i.e. not more than 100. An `ExtraTreesClassifier` is an ensemble method that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. This technique can be really slow in our dataset.


# Set an appropriate metric for imbalanced datasets.
metric = 'precision'

results = dict()
for train_data in [X_train, X_train_rfecv]:
    n_features = train_data.shape[1]
    print("{}_Features".format(n_features))
    # Create the pipeline object.
    pipe = Pipeline([('clf', LogisticRegression())])

    # Create a list of candidate classifiers with their hyper-parameters for grid search.
    search_space = [{'clf': [LogisticRegression(n_jobs=-1, class_weight='balanced',
                                                random_state=RANDOM_STATE)],
                     'clf__C': [1.0, 5.0]
                    },
                    {'clf': [RandomForestClassifier(n_jobs=-1, class_weight='balanced',
                                                    random_state=RANDOM_STATE)],
                     'clf__n_estimators': [10, 100],
                     'clf__max_depth': [2, 4]
                    },
                    {'clf': [DecisionTreeClassifier(class_weight='balanced',
                                                    random_state=RANDOM_STATE)],
                     'clf__min_samples_leaf': [1, 5],
                     'clf__criterion': ["gini", "entropy"]
                    },
                    {'clf': [ExtraTreesClassifier(n_jobs=-1, class_weight='balanced',
                                                  random_state=RANDOM_STATE)],
                     'clf__n_estimators': [10, 100],
                     'clf__max_depth': [2, 4]
                    },
                    {'clf': [AdaBoostClassifier(random_state=RANDOM_STATE)],
                     'clf__n_estimators': [10, 100]
                    }
                   ]

    # Create the grid search object.
    gridsearch = GridSearchCV(pipe, param_grid=search_space, scoring=metric, cv=5, verbose=5, n_jobs=-1)

    # Fit grid search.
    best_model = gridsearch.fit(train_data, y_train)

    # Cache results.
    results[str(n_features)] = best_model

    print(best_model.best_estimator_)
    print('The ' + metric + ' score of the best model is:', best_model.score(train_data, y_train))
    print()


# **Inference**
# 
# - Decision Tree gave the best score with all features.
# 
# Let's visualize the scores of all the combinations tested previously using `GridSearchCV`. We will sort the models by the best score. We can get valuable information with statistics of training times and scores.



# Convert the dictionary of all the combinations from the gridsearch to a Pandas df to visualize it properly.
score_results = pd.DataFrame(results['277'].cv_results_)

# Sort by best model first.
score_results.sort_values(by='rank_test_score')


# We will train again the best combinations of the previous models and then plot the scores as box plots to check visually the variance between k-fold results.



# Set the models to search.
models = [LogisticRegression(n_jobs=-1, class_weight='balanced', random_state=RANDOM_STATE, C=1.0),
          RandomForestClassifier(n_estimators=100, max_depth=2, class_weight='balanced',
                                 n_jobs=-1, random_state=RANDOM_STATE),
          DecisionTreeClassifier(min_samples_leaf=5, class_weight='balanced',
                                 criterion='gini', random_state=RANDOM_STATE),
          ExtraTreesClassifier(class_weight='balanced', n_estimators=100, max_depth=2,
                               n_jobs=-1, random_state=RANDOM_STATE),
          AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=100)]

# Set the number of folds for cross validation.
CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring=metric, cv=CV, n_jobs=-1)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['Classifier', 'fold_idx', metric])

# Plot the results.
cv_df = cv_df[['Classifier', metric]]
fig = px.box(cv_df, x='Classifier', y=metric)
fig.show()


# **Inference**
# 
# * We observe similar scores for all models, but let's select Extra Trees, which presented high scores with small range and is also a good model for tweaking with a lot hyper-parameters.
# 
# From the above results, it can be seen that `ExtraTreesClassifier` is one of the best classifiers for our case. Next, we will grid search the hyper-parameters to tune the model.



n_features = X_train.shape[1]
n_samples = X_train.shape[0]

# # Create a list of hyper-parameters for grid search.
# parameters = {'criterion': ['gini', 'entropy'],
#               'max_depth': [None,1,2,3,4,5,6,7],
#               'max_features': [None, 'sqrt', 'auto', 'log2', 0.3,0.5,0.7, n_features//2, n_features//3],
#               'min_samples_split': [2,0.3,0.5, n_samples//2, n_samples//3, n_samples//5],
#               'min_samples_leaf':[1, 0.3,0.5, n_samples//2, n_samples//3, n_samples//5]}

# Create a list of hyper-parameters for grid search.
parameters = {
    'n_estimators': [10, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4],
    'max_features': [None, 'sqrt', 'auto', 'log2'],
    'min_samples_split': [2, 0.5],
    'min_samples_leaf':[1, 0.5]
}

# Create the decision tree model object.
clf = ExtraTreesClassifier(class_weight='balanced', random_state=RANDOM_STATE)

# Create the grid search object.
gridsearch = GridSearchCV(estimator=clf, param_grid=parameters, scoring=metric, cv=5, verbose=0, n_jobs=-1)

# Fit grid search.
best_model = gridsearch.fit(X_train, y_train)

print(best_model.best_estimator_)
print('\nThe ' + metric + ' score of the best model is:', best_model.score(X_train, y_train))

# Convert the dictionary of all the combinations from the gridsearch to a Pandas df to visualize it properly.
score_results = pd.DataFrame(best_model.cv_results_)

# Sort by best model first and show first rows only.




fi = pd.DataFrame({'Features':X_train.columns,'FI':best_model.best_estimator_.feature_importances_})



# Now that we have fit and tuned our model, we will evaluate its performance by predicting off the test values. Test values were kept out of the initial dataset before scaling or any other preprocessing and can be considered safely as unknown values to the model.
# 
# Before feeding the model and proceeding with the predictions, we need to transform/scale the numerical features as before.
# 
# The model will be evaluated by constructing a confusion matrix, which is a table used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualisation of the performance of an algorithm and the easy identification of confusion between classes, e.g. one class is commonly mislabeled as the other. Most performance measures can be computed from the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).


# Create the classifier.
clf = ExtraTreesClassifier(class_weight='balanced', criterion='entropy', max_depth=2,
                           max_features='log2', n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1)

# Fit the classifier.
clf.fit(X_train, y_train)

# Make predictions.
y_pred = clf.predict(X_test)

# Print metrics.
print(classification_report(y_test, y_pred))

# Plot confusion matrix.
plot_confusion_matrix(clf, X_test, y_test)
plt.show()


# **Inference**
# 
# * The model cannot predict the minority class very well.


# Generate a no skill prediction for the majority class "1".
ns_probs = [0 for _ in range(len(y_test))]

# Predict probabilities.
clf_probs = clf.predict_proba(X_test)

# Keep probabilities of the positive outcome only.
clf_probs_1 = clf_probs[:, 1]

# Calculate scores.
ns_auc = roc_auc_score(y_test, ns_probs)
clf_auc = roc_auc_score(y_test, clf_probs_1)

# Summarize scores.
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (clf_auc))

# Calculate roc curves.
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
clf_fpr, clf_tpr, _ = roc_curve(y_test, clf_probs_1)

# Plot the roc curve for the model.
fig = go.Figure()
fig.add_trace(go.Scatter(x=ns_fpr, y=ns_tpr,
                    mode='lines',
                    name='No Skill'))
fig.add_trace(go.Scatter(x=clf_fpr, y=clf_tpr,
                    mode='lines+markers',
                    name='Decision Tree'))

fig.update_layout(width=550,
                  height=450,
                  title='Receiver Operating Characteristic (ROC)',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  xaxis = dict(
                      tickmode = 'linear',
                      tick0 = 0,
                      dtick = 0.2))

fig.show()


# **Inference**
# 
# * The model cannot predict, but this is not an appropriate score for an imbalanced dataset.
# 
# The optimal cut-off point is where "true positive rate" or TPR is high and the "false positive rate" or FPR is low. In other words, optimal cut-off probability is that probability where we have [balance](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) between **sensitivity** (or Recall or TPR) and **specificity** (or "true negative rate" or TNR). Let's apply this analysis in the train data set.



# Predict probabilities.
probabilities = clf.predict_proba(X_train)

# Keep probabilities for the positive outcome only.
probabilities = probabilities[:, 1]

# Create a df to keep the data.
y_train_predictions = pd.DataFrame({'conversion_name':y_train.values, 'Probabilities':probabilities})

# Create a col for predicted values using 0.5 as default threshold probability.
y_train_predictions['Predicted'] = y_train_predictions.Probabilities.map(lambda x: 1 if x > 0.5 else 0)

# Create columns with different probability cutoffs.
numbers = [float(x)/10 for x in range(10)]

for i in numbers:
    y_train_predictions[i]= y_train_predictions.Probabilities.map(lambda x: 1 if x > i else 0)

# Display first rows.

# Calculate accuracy sensitivity and specificity for the various probability cutoffs.
cutoffs = pd.DataFrame(columns = ['Probability', 'Accuracy', 'Sensitivity', 'Specificity'])

for i in numbers:
    cm = metrics.confusion_matrix(y_train_predictions.conversion_name, y_train_predictions[i])
    total = sum(sum(cm))
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1]/(cm[1, 0] + cm[1, 1])
    cutoffs.loc[i] =[i, accuracy, sensitivity, specificity]

# Plot accuracy, sensitivity and specificity for the various probabilities.
fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=cutoffs.Probability, y=cutoffs.Accuracy,
                         mode='lines',
                         name='Accuracy'))
fig.add_trace(go.Scatter(x=cutoffs.Probability, y=cutoffs.Sensitivity,
                         mode='lines',
                         name='Sensitivity'))
fig.add_trace(go.Scatter(x=cutoffs.Probability, y=cutoffs.Specificity,
                         mode='lines',
                         name='Specificity'))
fig.update_layout(title='Cut-off point in relation to specificity, sensitivity and accuracy of the model')
fig.show()

# * The curves of the three metrics intersect at \$0.45\$ and thus, we choose this as the **cut-off point**.
# 
# We can calculate the metrics again, using this point value and make predictions on the test set following this approach.



# Set cut-off point.
cutoff = 0.45

# Convert probabilities to predicted classes.
clf_probs_1[clf_probs_1 > cutoff] = 1
clf_probs_1[clf_probs_1 <= cutoff] = 0

# Print metrics.
print(classification_report(y_test, clf_probs_1))


# **Inference**
# 
# - Generally, playing around with the cut-off value, we can tweak a model to our needs.
# 
# Finally, we will fit the best model to the whole dataset, predict all the probabilities for each id, and export the results to a CSV file.



pd.set_option('display.max_rows', 50)

# Shuffle data.
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Divide data into X and y variables.
y = df['Orders']
X = df[X_train.columns]

# Create the classifier.
clf = ExtraTreesClassifier(class_weight='balanced', criterion='entropy', max_depth=2,
                           max_features='log2', n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1)

# Fit the classifier.
clf.fit(X, y)

# Predict probabilities using the cut-off point.
# probabilities = (clf.predict_proba(X)[:, 1] >= cutoff).astype(bool)
probabilities = clf.predict_proba(X)

# Keep probabilities for the positive outcome only.
pos_probabilities = probabilities[:, 1]

# Add probabilities to a new column.
df['Probabilities'] = pos_probabilities

# Keep the id/email and the predicted probabilities.
results = df[features + ['email', 'Orders', 'Probabilities']]

# Keep the features and the probabilities.
results2 = df[features + ['Orders', 'Probabilities']]

# Take the average of probs.
results_mean = results.groupby('email')['Probabilities'].agg([('Mean_Probabilities', 'mean'),
                                                              ('Count_Cases', 'count')])
results2_mean = results2.groupby(features)['Probabilities'].agg([('Mean_Probabilities', 'mean'),
                                                                 ('Count_Cases', 'count')])

# Sort by higher probability first.
results_mean = results_mean.sort_values(by=['Mean_Probabilities', 'Count_Cases'], ascending=False)
results2_mean = results2_mean.sort_values(by=['Mean_Probabilities', 'Count_Cases'], ascending=False)

# Show first columns.

# Save to CSV.
results_mean.to_csv('results.csv')
results2_mean.to_csv('results2.csv')


