#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# 
# # Automatic selection from 20 classifier models on the example of competition "Titanic: Machine Learning from Disaster"

# **Automatic selection of binary classification models** works as follows (methodology worked out on the my kernel with 15 regression models - [Suspended substances prediction in river](https://www.kaggle.com/vbmokin/suspended-substances-prediction-in-river)) - a training dataset (on the given *path* to files) with a target column is divided into training and "test" (*test_train_split_part* = 0.2: train/test => 80/20% default). Models are built on the training set and are tested on a "test" set - which they give a match on given metrics (default metrics: *r2-score*, *relative error* and *rmse* but you can only select some from them). The user chooses which metric is the main one (*metric_main*) - it selects the *N_best_models* amount of most accurate models for this metric. 
# Next, kernel automatic rebuild these the best of models on a full (100%) training dataset and apply to a true test file and generate *N_best_models* submission files.
# 
# Each model is built using cross-validation (except LGBM). The parameters of the model are selected to ensure the maximum matching of accuracy on the training and validation data. A plot is being built for this purpose with **learning_curve** from sklearn library.
# 
# **NEW: I created additional double features in different combinations to find new patterns**

# This kernel is based on the kernels:
# 
# * [Suspended substances prediction in river](https://www.kaggle.com/vbmokin/suspended-substances-prediction-in-river)
# 
# * [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models)
# 
# * [Titanic: random forest](https://www.kaggle.com/morenovanton/titanic-random-forest)

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [FE&EDA](#3)
# 1. [Preparing to modeling](#4)
# 1. [Tuning models with GridSearchCV](#5)
#     -  [Linear Regression](#5.1)
#     -  [Support Vector Machines](#5.2)
#     -  [Linear SVC](#5.3)
#     -  [MLP Classifier](#5.4)
#     -  [Stochastic Gradient Descent](#5.5)
#     -  [Decision Tree Classifier](#5.6)
#     -  [Random Forest](#5.7)
#     -  [XGB Classifier](#5.8)
#     -  [LGBM Classifier](#5.9)
#     -  [Gradient Boosting Classifier](#5.10)
#     -  [Ridge Classifier](#5.11)
#     -  [Bagging Classifier](#5.12)
#     -  [Extra Trees Classifier](#5.13)
#     -  [AdaBoost Classifier](#5.14)
#     -  [Logistic Regression](#5.15)
#     -  [k-Nearest Neighbors (KNN)](#5.16)
#     -  [Naive Bayes](#5.17)
#     -  [Perceptron](#5.18)
#     -  [Gaussian Process Classification](#5.19)
#     -  [Voting Classifier](#5.20)    
# 1. [Models comparison](#6)
# 1. [Prediction](#7)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pp
import math
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# preprocessing
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

# models
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


path = "../input/titanic/"


# In[ ]:


cv_n_split = 2
random_state = 0
test_train_split_part = 0.15


# In[ ]:


metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4] # you can only select some numbers of metrics from metrics_all


# In[ ]:


traindf = pd.read_csv(path + 'train.csv').set_index('PassengerId')
testdf = pd.read_csv(path + 'test.csv').set_index('PassengerId')
submission = pd.read_csv(path + 'gender_submission.csv')


# In[ ]:


traindf.head(3)


# In[ ]:


target_name = 'Survived'


# In[ ]:


traindf.info()


# In[ ]:


testdf.info()


# ## 3. FE&EDA <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# **FE based on the my kernel [Autoselection from 20 classifier models & L_curves](https://www.kaggle.com/vbmokin/autoselection-from-20-classifier-models-l-curves)**

# In[ ]:


#Thanks to:
# https://www.kaggle.com/mauricef/titanic
# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code

df = pd.concat([traindf, testdf], axis=0, sort=False)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]
family = df.groupby(df.LastName).Survived
df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())
df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)
df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())
df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount -                                     df.Survived.fillna(0), axis=0)
df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)
df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)
df['Alone'] = (df.WomanOrBoyCount == 0)


# In[ ]:


df.head(3)


# ### FE from the notebook https://www.kaggle.com/morenovanton/titanic-random-forest

# In[ ]:


#Thanks to https://www.kaggle.com/morenovanton/titanic-random-forest
# Title !
df['Title'] = df['Title'].replace('Ms','Miss')
df['Title'] = df['Title'].replace('Mlle','Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')

# Embarked, Fare !
df['Embarked'] = df['Embarked'].fillna('S')
med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df['Fare'] = df['Fare'].fillna(med_fare)

# Cabin, Deck, famous_cabin !
df['famous_cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'

# Family_Size !
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Name_length !
df['Name_length'] = df['Name'].apply(len)

df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)
df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)
df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)
df.Alone = df.Alone.fillna(0)


# In[ ]:


cols_to_drop = ['Name','Ticket','Cabin']   
df = df.drop(cols_to_drop, axis=1)


# In[ ]:


Y = df.Survived.loc[traindf.index].astype(int)
X_train, X_test = df.loc[traindf.index], df.loc[testdf.index]
X_test = X_test.drop(['Survived'], axis = 1)


# In[ ]:


print(X_train.isnull().sum())
#print(X_test.isnull().sum())


# In[ ]:


# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = X_train.columns.values.tolist()
for col in features:
    if X_train[col].dtype in numerics: continue
    categorical_columns.append(col)
categorical_columns


# In[ ]:


# Encoding categorical features
for col in categorical_columns:
    if col in X_train.columns:
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        X_test[col] = le.transform(list(X_test[col].astype(str).values))


# In[ ]:


X_train = X_train.reset_index()
X_test = X_test.reset_index()
X_dropna_categor = X_train.dropna().astype(int)
Xtest_dropna_categor = X_test.dropna().astype(int)


# In[ ]:


# Surviving girls:
Sex_female_Survived = X_dropna_categor.loc[(X_dropna_categor.Sex == 0) & (X_dropna_categor.Survived == 1)]
# Dead girls:
Sex_female_NoSurvived = X_dropna_categor.loc[(X_dropna_categor.Sex == 0) & (X_dropna_categor.Survived == 0)]
# Surviving guys:
X_Sex_male_Survived = X_dropna_categor.loc[(X_dropna_categor.Sex == 1) & (X_dropna_categor.Survived == 1)] 
# Dead guys:
X_Sex_male_NoSurvived = X_dropna_categor.loc[(X_dropna_categor.Sex == 1) & (X_dropna_categor.Survived == 0)]

X_test_male = Xtest_dropna_categor.loc[Xtest_dropna_categor.Sex == 1]
X_test_female = Xtest_dropna_categor.loc[Xtest_dropna_categor.Sex == 0]

female_Survived_mean, female_NoSurvived_mean = Sex_female_Survived['Age'].mean(), Sex_female_NoSurvived['Age'].mean()
male_Survived_mean, male_NoSurvived_mean = X_Sex_male_Survived['Age'].mean(), X_Sex_male_NoSurvived['Age'].mean()

female_Survived_std, female_NoSurvived_std = Sex_female_Survived['Age'].std(), Sex_female_NoSurvived['Age'].std()
male_Survived_std, male_NoSurvived_std = X_Sex_male_Survived['Age'].std(), X_Sex_male_NoSurvived['Age'].std()

female_std, female_mean = X_test_female['Age'].std(), X_test_female['Age'].mean()
male_std, male_mean = X_test_male['Age'].std(), X_test_male['Age'].mean()

X_train['Survived'] = X_train['Survived'].astype(int)


# In[ ]:


# Confidence interval calculation function: 
def derf(sample, mean, std):
    age_shape = sample['Age'].shape[0] # sample size
    if age_shape > 0:
        standard_error_ofthe_mean = std / math.sqrt(age_shape)
        random_mean = round(random.uniform(mean-(1.96*standard_error_ofthe_mean), mean+(1.96*standard_error_ofthe_mean)), 2)
    else: random_mean = 0
    
    return random_mean


# In[ ]:


for i in X_train.loc[(X_train['Sex']==0) & (X_train['Survived']==1) & (X_train['Age'].isnull())].index:
    X_train.at[i, 'Age'] = derf(Sex_female_Survived, female_Survived_mean, female_Survived_std)

for h in X_train.loc[(X_train['Sex']==0) & (X_train['Survived']==0) & (X_train['Age'].isnull())].index:
    X_train.at[h, 'Age'] = derf(Sex_female_NoSurvived, female_NoSurvived_mean, female_NoSurvived_std)
    
for l in X_train.loc[(X_train['Sex']==1) & (X_train['Survived']==1) & (X_train['Age'].isnull())].index:
    X_train.at[l, 'Age'] = derf(X_Sex_male_Survived, male_Survived_mean, male_Survived_std)
    
for b in X_train.loc[(X_train['Sex']==1) & (X_train['Survived']==0) & (X_train['Age'].isnull())].index:
    X_train.at[b, 'Age'] = derf(X_Sex_male_NoSurvived, male_NoSurvived_mean, male_NoSurvived_std)
    
for p in X_test.loc[(X_test['Sex']==1) & (X_test['Age'].isnull())].index:
    X_test.at[p, 'Age'] = derf(X_test_male, male_mean, male_std)

for y in X_test.loc[(X_test['Sex']==0) & (X_test['Age'].isnull())].index:
    X_test.at[y, 'Age'] = derf(X_test_female, female_mean, female_std)


# In[ ]:


X_train


# In[ ]:


X_train.describe()


# In[ ]:


X_train = X_train.drop(['Survived'], axis = 1)


# In[ ]:


print(X_train.isnull().sum())
print(X_test.isnull().sum())


# ### My upgrade - creation new features

# In[ ]:


def fe_creation(df):
    df['Age2'] = df['Age']//10
    df['Fare2'] = df['Fare']//10
    for i in ['Sex', 'Family_Size', 'Fare2','Alone', 'famous_cabin']:
        for j in ['Age2','Title', 'Embarked', 'Deck']:
            df[i + "_" + j] = df[i].astype('str') + "_" + df[j].astype('str')
    return df

X_train = fe_creation(X_train)
X_test = fe_creation(X_test)


# In[ ]:


# Determination categorical features
categorical_columns = []
features = X_train.columns.values.tolist()
for col in features:
    if X_train[col].dtype in numerics: continue
    categorical_columns.append(col)
categorical_columns


# In[ ]:


# Encoding categorical features
for col in categorical_columns:
    if col in X_train.columns:
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        X_test[col] = le.transform(list(X_test[col].astype(str).values))


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


train0, test0 = X_train, X_test
target0 = Y


# In[ ]:


train0.head(3)


# In[ ]:


train0.info()


# In[ ]:


train0.describe()


# **EDA based on the my kernel [FE & EDA with Pandas Profiling](https://www.kaggle.com/vbmokin/fe-eda-with-pandas-profiling)**

# In[ ]:


#pp.ProfileReport(train0)


# In[ ]:


#pp.ProfileReport(test0)


# ## 4. Preparing to modeling <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


train0.info()


# In[ ]:


# Standartization
scaler = StandardScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)
test0 = pd.DataFrame(scaler.transform(test0), columns = test0.columns)

# For boosting model
train0b = train0.copy()
test0b = test0.copy()
# Synthesis valid as "test" for selection models
trainb, testb, targetb, target_testb = train_test_split(train0b, target0, test_size=test_train_split_part, random_state=random_state)


# In[ ]:


# For models from Sklearn
# Normalization
scaler = MinMaxScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)
test0 = pd.DataFrame(scaler.fit_transform(test0), columns = test0.columns)


# In[ ]:


train0.head(3)


# In[ ]:


# Synthesis valid as test for selection models
train, test, target, target_test = train_test_split(train0, target0, test_size=test_train_split_part, random_state=random_state)


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


# list of accuracy of all model - amount of metrics_now * 2 (train & test datasets)
num_models = 20
acc_train = []
acc_test = []
acc_all = np.empty((len(metrics_now)*2, 0)).tolist()
acc_all


# In[ ]:


acc_all_pred = np.empty((len(metrics_now), 0)).tolist()
acc_all_pred


# In[ ]:


# Splitting train data for model tuning with cross-validation
cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)


# In[ ]:


def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5


# In[ ]:


def acc_metrics_calc(num,model,train,test,target,target_test):
    # The models selection stage
    # Calculation of accuracy of model by different metrics
    global acc_all

    ytrain = model.predict(train).astype(int)
    ytest = model.predict(test).astype(int)
    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])
    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])

    num_acc = 0
    for x in metrics_now:
        if x == 1:
            #r2_score criterion
            acc_train = round(r2_score(target, ytrain) * 100, 2)
            acc_test = round(r2_score(target_test, ytest) * 100, 2)
        elif x == 2:
            #accuracy_score criterion
            acc_train = round(metrics.accuracy_score(target, ytrain) * 100, 2)
            acc_test = round(metrics.accuracy_score(target_test, ytest) * 100, 2)
        elif x == 3:
            #rmse criterion
            acc_train = round(acc_rmse(target, ytrain) * 100, 2)
            acc_test = round(acc_rmse(target_test, ytest) * 100, 2)
        elif x == 4:
            #relative error criterion
            acc_train = round(acc_d(target, ytrain) * 100, 2)
            acc_test = round(acc_d(target_test, ytest) * 100, 2)
        
        print('acc of', metrics_all[x], 'for train =', acc_train)
        print('acc of', metrics_all[x], 'for test =', acc_test)
        acc_all[num_acc].append(acc_train) #train
        acc_all[num_acc+1].append(acc_test) #test
        num_acc += 2


# In[ ]:


def acc_metrics_calc_pred(num,model,name_model,train,test,target):
    # The prediction stage
    # Calculation of accuracy of model for all different metrics and creates of the main submission file for the best model (num=0)
    global acc_all_pred

    ytrain = model.predict(train).astype(int)
    ytest = model.predict(test).astype(int)

    print('**********')
    print(name_model)
    print('target = ', target[:15].values)
    print('ytrain = ', ytrain[:15])
    print('ytest =', ytest[:15])
    
    num_acc = 0
    for x in metrics_now:
        if x == 1:
            #r2_score criterion
            acc_train = round(r2_score(target, ytrain) * 100, 2)
        elif x == 2:
            #accuracy_score criterion
            acc_train = round(metrics.accuracy_score(target, ytrain) * 100, 2)
        elif x == 3:
            #rmse criterion
            acc_train = round(acc_rmse(target, ytrain) * 100, 2)
        elif x == 4:
            #relative error criterion
            acc_train = round(acc_d(target, ytrain) * 100, 2)

        print('acc of', metrics_all[x], 'for train =', acc_train)
        acc_all_pred[num_acc].append(acc_train) #train
        num_acc += 1
    
    # Save the submission file
    submission[target_name] = ytest
    submission.to_csv('submission_' + name_model + '.csv', index=False)    


# ## 5. Tuning models and test for all features <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning, we can narrow down our choice of models to a few. These include:
# 
# - Linear Regression, Logistic Regression
# - Naive Bayes 
# - k-Nearest Neighbors algorithm
# - Perceptron
# - Support Vector Machines and Linear SVR
# - Stochastic Gradient Descent, GradientBoostingRegressor, RidgeCV, BaggingRegressor
# - Decision Tree Classifier, Random Forest, AdaBoostClassifier, XGBRegressor, LGBM, ExtraTreesRegressor 
# - Gaussian Process Classification
# - MLPRegressor (Deep Learning)
# - Voting Classifier
# 
# Each model is built using cross-validation (except LGBM). The parameters of the model are selected to ensure the maximum matching of accuracy on the training and validation data. A plot is being built for this purpose with [learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html?highlight=learning_curve#sklearn.model_selection.learning_curve) from sklearn library.

# In[ ]:


# Thanks to https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, cv=None, axes=None, ylim=None, 
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), random_state=0):
    """
    Generate 2 plots: 
    - the test and training learning curve, 
    - the training samples vs fit times curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    
    random_state : random_state
    
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)
    
    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator=estimator, X=X, y=y, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    plt.show()
    return


# ### 5.1 Linear Regression <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# **Linear Regression** is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression. Reference [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression).
# 
# Note the confidence score generated by the model based on our training dataset.

# In[ ]:


# Linear Regression
linreg = LinearRegression()
linreg_CV = GridSearchCV(linreg, param_grid={}, cv=cv_train, verbose=False)
linreg_CV.fit(train, target)
print(linreg_CV.best_params_)
acc_metrics_calc(0,linreg_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(linreg, "Linear Regression", train, target, cv=cv_train)


# ### 5.2 Support Vector Machines <a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# **Support Vector Machines** are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).

# In[ ]:


# Support Vector Machines

svr = SVC()
svr_CV = GridSearchCV(svr, param_grid={'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                       'tol': [1e-4]}, 
                      cv=cv_train, verbose=False)
svr_CV.fit(train, target)
print(svr_CV.best_params_)
acc_metrics_calc(1,svr_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(svr, "Support Vector Machines", train, target, cv=cv_train)


# ### 5.3 Linear SVC <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# **Linear SVC** is a similar to SVM method. Its also builds on kernel functions but is appropriate for unsupervised learning. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Support-vector_clustering_(svr).

# In[ ]:


# Linear SVR

linear_svc = LinearSVC()
param_grid = {'dual':[False],
              'C': np.linspace(1, 15, 15)}
linear_svc_CV = GridSearchCV(linear_svc, param_grid=param_grid, cv=cv_train, verbose=False)
linear_svc_CV.fit(train, target)
print(linear_svc_CV.best_params_)
acc_metrics_calc(2,linear_svc_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(linear_svc, "Linear SVR", train, target, cv=cv_train)


# ### 5.4 MLP Classifier<a class="anchor" id="5.4"></a>
# 
# [Back to Table of Contents](#0.1)

# The **MLPClassifier** optimizes the squared-loss using LBFGS or stochastic gradient descent by the Multi-layer Perceptron regressor. Reference [Sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor).

# Thanks to:
# * https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
# * https://stackoverflow.com/questions/44803596/scikit-learn-mlpregressor-performance-cap

# In[ ]:


get_ipython().run_cell_magic('time', '', "# MLPClassifier\n\nmlp = MLPClassifier()\nparam_grid = {'hidden_layer_sizes': [i for i in range(2,5)],\n              'solver': ['sgd'],\n              'learning_rate': ['adaptive'],\n              'max_iter': [1000]\n              }\nmlp_GS = GridSearchCV(mlp, param_grid=param_grid, cv=cv_train, verbose=False)\nmlp_GS.fit(train, target)\nprint(mlp_GS.best_params_)\nacc_metrics_calc(3,mlp_GS,train,test,target,target_test)")


# In[ ]:


# Building learning curve of model
plot_learning_curve(mlp, "MLP Classifier", train, target, cv=cv_train)


# ### 5.5 Stochastic Gradient Descent <a class="anchor" id="5.5"></a>
# 
# [Back to Table of Contents](#0.1)

# **Stochastic gradient descent** (often abbreviated **SGD**) is an iterative method for optimizing an objective function with suitable smoothness properties (e.g. differentiable or subdifferentiable). It can be regarded as a stochastic approximation of gradient descent optimization, since it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data). Especially in big data applications this reduces the computational burden, achieving faster iterations in trade for a slightly lower convergence rate. Reference [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier(early_stopping=True)
param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
sgd_CV = GridSearchCV(sgd, param_grid=param_grid, cv=cv_train, verbose=False)
sgd_CV.fit(train, target)
print(sgd_CV.best_params_)
acc_metrics_calc(4,sgd_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(sgd, "Stochastic Gradient Descent", train, target, cv=cv_train)


# ### 5.6 Decision Tree Classifier<a class="anchor" id="5.6"></a>
# 
# [Back to Table of Contents](#0.1)

# This model uses a **Decision Tree** as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).

# In[ ]:


# Decision Tree Classifier

decision_tree = DecisionTreeClassifier()
param_grid = {'min_samples_leaf': [i for i in range(2,10)]}
decision_tree_CV = GridSearchCV(decision_tree, param_grid=param_grid, cv=cv_train, verbose=False)
decision_tree_CV.fit(train, target)
print(decision_tree_CV.best_params_)
acc_metrics_calc(5,decision_tree_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(decision_tree, "Decision Tree", train, target, cv=cv_train)


# ### 5.7 Random Forest <a class="anchor" id="5.7"></a>
# 
# [Back to Table of Contents](#0.1)

# **Random Forest** is one of the most popular model. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators= [100, 300]) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Random Forest\n# Parameters of model (param_grid) taken from the notebook https://www.kaggle.com/morenovanton/titanic-random-forest\n\nrandom_forest = RandomForestClassifier()\nparam_grid = {'n_estimators': [300, 400, 500, 600], 'min_samples_split': [60], 'min_samples_leaf': [20, 25, 30, 35, 40], \n              'max_features': ['auto'], 'max_depth': [5, 6, 7, 8, 9, 10], 'criterion': ['gini'], 'bootstrap': [False]}\nrandom_forest_CV = GridSearchCV(estimator=random_forest, param_grid=param_grid, \n                             cv=cv_train, verbose=False)\nrandom_forest_CV.fit(train, target)\nprint(random_forest_CV.best_params_)\nacc_metrics_calc(6,random_forest_CV,train,test,target,target_test)")


# In[ ]:


# Building learning curve of model
plot_learning_curve(random_forest, "Random Forest", train, target, cv=cv_train)


# ### 5.8 XGB Classifier<a class="anchor" id="5.8"></a>
# 
# [Back to Table of Contents](#0.1)

# **XGBoost** is an ensemble tree method that apply the principle of boosting weak learners (CARTs generally) using the gradient descent architecture. XGBoost improves upon the base Gradient Boosting Machines (GBM) framework through systems optimization and algorithmic enhancements. Reference [Towards Data Science.](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)

# In[ ]:


get_ipython().run_cell_magic('time', '', '# XGBoost Classifier\nxgb_clf = xgb.XGBClassifier(objective=\'reg:squarederror\') \nparameters = {\'n_estimators\': [200, 300, 400], \n              \'learning_rate\': [0.001, 0.003, 0.005, 0.006, 0.01],\n              \'max_depth\': [4, 5, 6]}\nxgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=cv_train).fit(trainb, targetb)\nprint("Best score: %0.3f" % xgb_reg.best_score_)\nprint("Best parameters set:", xgb_reg.best_params_)\nacc_metrics_calc(7,xgb_reg,trainb,testb,targetb,target_testb)')


# In[ ]:


# Building learning curve of model
plot_learning_curve(xgb_clf, "XGBoost Classifier", trainb, targetb, cv=cv_train)


# ### 5.9 LGBM Classifier <a class="anchor" id="5.9"></a>
# 
# [Back to Table of Contents](#0.1)

# **Light GBM** is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithms. It splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’. Reference [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/).

# In[ ]:


#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=test_train_split_part, random_state=random_state)
modelL = lgb.LGBMClassifier(n_estimators=1000, num_leaves=50)
modelL.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], early_stopping_rounds=50, verbose=True)


# In[ ]:


acc_metrics_calc(8,modelL,trainb,testb,targetb,target_testb)


# In[ ]:


fig =  plt.figure(figsize = (5,5))
axes = fig.add_subplot(111)
lgb.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();
plt.close()


# ### 5.10 Gradient Boosting Classifier<a class="anchor" id="5.10"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/kabure/titanic-eda-model-pipeline-keras-nn

# **Gradient Boosting** builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced. The features are always randomly permuted at each split. Therefore, the best found split may vary, even with the same training data and max_features=n_features, if the improvement of the criterion is identical for several splits enumerated during the search of the best split. To obtain a deterministic behaviour during fitting, random_state has to be fixed. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

# In[ ]:


# Gradient Boosting Classifier

gradient_boosting = GradientBoostingClassifier()
param_grid = {'learning_rate' : [0.001, 0.01, 0.1],
              'max_depth': [i for i in range(2,5)],
              'min_samples_leaf': [i for i in range(2,5)]}
gradient_boosting_CV = GridSearchCV(estimator=gradient_boosting, param_grid=param_grid, 
                                    cv=cv_train, verbose=False)
gradient_boosting_CV.fit(train, target)
print(gradient_boosting_CV.best_params_)
acc_metrics_calc(9,gradient_boosting_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(gradient_boosting_CV, "Gradient Boosting Classifier", train, target, cv=cv_train)


# ### 5.11 Ridge Classifier <a class="anchor" id="5.11"></a>
# 
# [Back to Table of Contents](#0.1)

# Tikhonov Regularization, colloquially known as **Ridge Classifier**, is the most commonly used regression algorithm to approximate an answer for an equation with no unique solution. This type of problem is very common in machine learning tasks, where the "best" solution must be chosen using limited data. If a unique solution exists, algorithm will return the optimal value. However, if multiple solutions exist, it may choose any of them. Reference [Brilliant.org](https://brilliant.org/wiki/ridge-regression/).

# In[ ]:


# Ridge Classifier

ridge = RidgeClassifier()
ridge_CV = GridSearchCV(estimator=ridge, param_grid={'alpha': np.linspace(.1, 1.5, 15)}, cv=cv_train, verbose=False)
ridge_CV.fit(train, target)
print(ridge_CV.best_params_)
acc_metrics_calc(10,ridge_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(ridge_CV, "Ridge Classifier", train, target, cv=cv_train)


# ### 5.12 BaggingClassifier <a class="anchor" id="5.12"></a>
# 
# [Back to Table of Contents](#0.1)

# Bootstrap aggregating, also called **Bagging**, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach. Bagging leads to "improvements for unstable procedures", which include, for example, artificial neural networks, classification and regression trees, and subset selection in linear regression. On the other hand, it can mildly degrade the performance of stable methods such as K-nearest neighbors. Reference [Wikipedia](https://en.wikipedia.org/wiki/Bootstrap_aggregating).

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Bagging Classifier\n\nbagging = BaggingClassifier(base_estimator=linear_svc_CV)\nparam_grid={'max_features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],\n            'n_estimators': [3, 5, 10],\n            'warm_start' : [True],\n            'random_state': [random_state]}\nbagging_CV = GridSearchCV(estimator=bagging, param_grid=param_grid, cv=cv_train, verbose=False)\nbagging_CV.fit(train, target)\nprint(bagging_CV.best_params_)\nacc_metrics_calc(11,bagging_CV,train,test,target,target_test)")


# In[ ]:


# Building learning curve of model
plot_learning_curve(bagging_CV, "Bagging Classifier", train, target, cv=cv_train)


# ### 5.13 Extra Trees Classifier <a class="anchor" id="5.13"></a>
# 
# [Back to Table of Contents](#0.1)

# **ExtraTreesClassifier** implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The default values for the parameters controlling the size of the trees (e.g. max_depth, min_samples_leaf, etc.) lead to fully grown and unpruned trees which can potentially be very large on some data sets. To reduce memory consumption, the complexity and size of the trees should be controlled by setting those parameter values. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html). 
# 
# In extremely randomized trees, randomness goes one step further in the way splits are computed. As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/ensemble.html#Extremely%20Randomized%20Trees).

# In[ ]:


# Extra Trees Classifier

etr = ExtraTreesClassifier()
etr_CV = GridSearchCV(estimator=etr, param_grid={'min_samples_leaf' : [10, 20, 30, 40, 50]}, cv=cv_train, verbose=False)
etr_CV.fit(train, target)
acc_metrics_calc(12,etr_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(etr, "Extra Trees Classifier", train, target, cv=cv_train)


# ### 5.14 AdaBoost Classifier <a class="anchor" id="5.14"></a>
# 
# [Back to Table of Contents](#0.1)

# The core principle of **AdaBoost** ("Adaptive Boosting") is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction. The data modifications at each so-called boosting iteration consist of applying N weights to each of the training samples. Initially, those weights are all set to 1/N, so that the first step simply trains a weak learner on the original data. For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data. At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly. As iterations proceed, examples that are difficult to predict receive ever-increasing influence. Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/ensemble.html#adaboost).

# In[ ]:


# AdaBoost Classifier

Ada_Boost = AdaBoostClassifier()
Ada_Boost_CV = GridSearchCV(estimator=Ada_Boost, param_grid={'learning_rate' : [.01, .1, .5, 1]}, cv=cv_train, verbose=False)
Ada_Boost_CV.fit(train, target)
acc_metrics_calc(13,Ada_Boost_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(Ada_Boost, "AdaBoost Classifier", train, target, cv=cv_train)


# ### 5.15 Logistic Regression <a class="anchor" id="5.15"></a>
# 
# [Back to Table of Contents](#0.1)

# **Logistic Regression** is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg_CV = GridSearchCV(estimator=logreg, param_grid={'C' : [.1, .3, .5, .7, 1]}, cv=cv_train, verbose=False)
logreg_CV.fit(train, target)
acc_metrics_calc(14,logreg_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(logreg, "Logistic Regression", train, target, cv=cv_train)


# ### 5.16 k-Nearest Neighbors (KNN)<a class="anchor" id="5.16"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/startupsci/titanic-data-science-solutions

# In pattern recognition, the **k-Nearest Neighbors algorithm** (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

# In[ ]:


# KNN - k-Nearest Neighbors algorithm

knn = KNeighborsClassifier()
param_grid={'n_neighbors': range(2, 7)}
knn_CV = GridSearchCV(estimator=knn, param_grid=param_grid, 
                      cv=cv_train, verbose=False).fit(train, target)
print(knn_CV.best_params_)
acc_metrics_calc(15,knn_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(knn, "KNN", train, target, cv=cv_train)


# ### 5.17 Naive Bayes <a class="anchor" id="5.17"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/startupsci/titanic-data-science-solutions

# In machine learning, **Naive Bayes classifiers** are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
param_grid={'var_smoothing': [1e-8, 1e-9, 1e-10]}
gaussian_CV = GridSearchCV(estimator=gaussian, param_grid=param_grid, cv=cv_train, verbose=False)
gaussian_CV.fit(train, target)
print(gaussian_CV.best_params_)
acc_metrics_calc(16,gaussian_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(gaussian, "Gaussian Naive Bayes", train, target, cv=cv_train)


# ### 5.18 Perceptron <a class="anchor" id="5.18"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/startupsci/titanic-data-science-solutions

# The **Perceptron** is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. Reference [Wikipedia](https://en.wikipedia.org/wiki/Perceptron).

# In[ ]:


# Perceptron

perceptron = Perceptron()
param_grid = {'penalty': [None, 'l2', 'l1', 'elasticnet']}
perceptron_CV = GridSearchCV(estimator=perceptron, param_grid=param_grid, cv=cv_train, verbose=False)
perceptron_CV.fit(train, target)
print(perceptron_CV.best_params_)
acc_metrics_calc(17,perceptron_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(perceptron, "Perceptron", train, target, cv=cv_train)


# ### 5.19 Gaussian Process Classification <a class="anchor" id="5.19"></a>
# 
# [Back to Table of Contents](#0.1)

# The **GaussianProcessClassifier** implements Gaussian processes (GP) for classification purposes, more specifically for probabilistic classification, where test predictions take the form of class probabilities. GaussianProcessClassifier places a GP prior on a latent function, which is then squashed through a link function to obtain the probabilistic classification. The latent function is a so-called nuisance function, whose values are not observed and are not relevant by themselves. Its purpose is to allow a convenient formulation of the model. GaussianProcessClassifier implements the logistic link function, for which the integral cannot be computed analytically but is easily approximated in the binary case.
# 
# In contrast to the regression setting, the posterior of the latent function is not Gaussian even for a GP prior since a Gaussian likelihood is inappropriate for discrete class labels. Rather, a non-Gaussian likelihood corresponding to the logistic link function (logit) is used. GaussianProcessClassifier approximates the non-Gaussian posterior with a Gaussian based on the Laplace approximation. Reference [Sklearn documentation](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc).

# In[ ]:


# Gaussian Process Classification

gpc = GaussianProcessClassifier()
param_grid = {'max_iter_predict': [100, 200],
              'warm_start': [True, False],
              'n_restarts_optimizer': range(3)}
gpc_CV = GridSearchCV(estimator=gpc, param_grid=param_grid, cv=cv_train, verbose=False)
gpc_CV.fit(train, target)
print(gpc_CV.best_params_)
acc_metrics_calc(18,gpc_CV,train,test,target,target_test)


# In[ ]:


# Building learning curve of model
plot_learning_curve(gpc, "Gaussian Process Classification", train, target, cv=cv_train)


# ### 5.20 Voting Classifier <a class="anchor" id="5.20"></a>
# 
# [Back to Table of Contents](#0.1)

# There is **VotingClassifier**. The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a majority vote (hard vote) or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses. Reference [sklearn documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier).

# In[ ]:


# Voting Classifier

Voting_ens = VotingClassifier(estimators=[('log', logreg_CV), ('mlp', mlp_GS), ('svc', linear_svc_CV)])
Voting_ens.fit(train, target)
acc_metrics_calc(19,Voting_ens,train,test,target,target_test)


# ## 6. Models comparison <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# We can now compare our models and to choose the best one for our problem.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Machines', 'Linear SVC', 
              'MLPClassifier', 'Stochastic Gradient Decent', 
              'Decision Tree Classifier', 'Random Forest',  'XGBClassifier', 'LGBMClassifier',
              'GradientBoostingClassifier', 'RidgeClassifier', 'BaggingClassifier', 'ExtraTreesClassifier', 
              'AdaBoostClassifier', 'Logistic Regression',
              'KNN', 'Naive Bayes', 'Perceptron', 'Gaussian Process Classification',
              'VotingClassifier']})


# In[ ]:


for x in metrics_now:
    xs = metrics_all[x]
    models[xs + '_train'] = acc_all[(x-1)*2]
    models[xs + '_test'] = acc_all[(x-1)*2+1]
    if xs == "acc":
        models[xs + '_diff'] = models[xs + '_train'] - models[xs + '_test']
models


# In[ ]:


print('Prediction accuracy for models')
ms = metrics_all[metrics_now[1]] # the accuracy
models.sort_values(by=[(ms + '_test'), (ms + '_train')], ascending=False)


# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format


# In[ ]:


for x in metrics_now:   
    # Plot
    xs = metrics_all[x]
    xs_train = metrics_all[x] + '_train'
    xs_test = metrics_all[x] + '_test'
    plt.figure(figsize=[25,6])
    xx = models['Model']
    plt.tick_params(labelsize=14)
    plt.plot(xx, models[xs_train], label = xs_train)
    plt.plot(xx, models[xs_test], label = xs_test)
    plt.legend()
    plt.title(str(xs) + ' criterion for ' + str(num_models) + ' popular models for train and test datasets')
    plt.xlabel('Models')
    plt.ylabel(xs + ', %')
    plt.xticks(xx, rotation='vertical')
    plt.show()


# In[ ]:


# Choose the number of metric by which the best models will be determined =>  {1 : 'r2_score', 2 : 'relative_error', 3 : 'rmse'}
metrics_main = 2
xs = metrics_all[metrics_main]
xs_train = metrics_all[metrics_main] + '_train'
xs_test = metrics_all[metrics_main] + '_test'
print('The best models by the',xs,'criterion:')
direct_sort = False if (metrics_main >= 2) else True
models_sort = models.sort_values(by=[xs_test, xs_train], ascending=direct_sort)


# In[ ]:


# Selection the best models except VotingClassifier
models_sort = models_sort[models_sort.Model != 'VotingClassifier']
models_best = models_sort[(models_sort.acc_diff < 5) & (models_sort.acc_train > 90)]
models_best[['Model', ms + '_train', ms + '_test', 'acc_diff']].sort_values(by=['acc_test'], ascending=False)


# ## 7. Prediction <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


models_pred = pd.DataFrame(models_best.Model, columns = ['Model'])
N_best_models = len(models_best.Model)


# You should copy the code here from the appropriate section to prepare one of the best models on the entire training dataset train0

# In[ ]:


def model_fit(name_model,train,target):
    # Fitting name_model (from 20 options) for giver train and target
    # You can optionally add hyperparameters optimization in any model
    if name_model == 'LGBMClassifier':
        Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=test_train_split_part, random_state=random_state)
        model = lgb.LGBMClassifier(n_estimators=1000)
        model.fit(Xtrain, Ztrain, eval_set=[(Xval, Zval)], early_stopping_rounds=50, verbose=False)
                
    else:
        param_grid={}
        
        if name_model == 'Linear Regression':
            model_clf = LinearRegression()
            
        elif name_model == 'Support Vector Machines':
            model_clf = SVC()
            param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'tol': [1e-4]}
            
        elif name_model == 'Linear SVC':
            model_clf = LinearSVC()
            param_grid = {'dual':[False],
                          'C': np.linspace(1, 15, 15)}
            
        elif name_model == 'MLPClassifier':
            model_clf = MLPClassifier()
            param_grid = {'hidden_layer_sizes': [i for i in range(2,5)],
                          'solver': ['sgd'],
                          'learning_rate': ['adaptive'],
                          'max_iter': [1000]
                          }

        elif name_model == 'Stochastic Gradient Decent':
            model_clf = SGDClassifier(early_stopping=True)
            param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}

        elif name_model == 'Decision Tree Classifier':
            model_clf = DecisionTreeClassifier()
            param_grid = {'min_samples_leaf': [i for i in range(2,10)]}

        elif name_model == 'Random Forest':
            model_clf = RandomForestClassifier()
            param_grid = {'n_estimators': [300, 400, 500, 600], 'min_samples_split': [60], 'min_samples_leaf': [20, 25, 30, 35, 40], 
                          'max_features': ['auto'], 'max_depth': [5, 6, 7, 8, 9, 10], 'criterion': ['gini'], 'bootstrap': [False]}

        elif name_model == 'XGBClassifier':
            model_clf = xgb.XGBClassifier(objective='reg:squarederror') 
            param_grid = {'n_estimators': [200, 300, 400], 
                          'learning_rate': [0.001, 0.003, 0.005, 0.006, 0.01],
                          'max_depth': [4, 5, 6]}
                        
        elif name_model == 'GradientBoostingClassifier':
            model_clf = GradientBoostingClassifier()
            param_grid = {'learning_rate' : [0.001, 0.01, 0.1],
                          'max_depth': [i for i in range(2,5)],
                          'min_samples_leaf': [i for i in range(2,5)]}

        elif name_model == 'RidgeClassifier':
            model_clf = RidgeClassifier()
            param_grid={'alpha': np.linspace(.1, 1.5, 15)}

        elif name_model == 'BaggingClassifier':
            model_base_estimator = GridSearchCV(LinearSVC(), param_grid={'dual':[False], 'C': np.linspace(1, 15, 15)}, 
                                            cv=cv_train, verbose=False)
            model_base_estimator.fit(train, target)
            model_clf = BaggingClassifier(base_estimator=model_base_estimator)
            param_grid={'max_features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'n_estimators': [3, 5, 10],
                        'warm_start' : [True],
                        'random_state': [random_state]}

        elif name_model == 'ExtraTreesClassifier':
            model_clf = ExtraTreesClassifier()
            param_grid={'min_samples_leaf' : [10, 20, 30, 40, 50]}

        elif name_model == 'AdaBoostClassifier':
            model_clf = AdaBoostClassifier()
            param_grid={'learning_rate' : [.01, .1, .5, 1]}

        elif name_model == 'Logistic Regression':
            model_clf = LogisticRegression()
            param_grid={'C' : [.1, .3, .5, .7, 1]}

        elif name_model == 'KNN':
            model_clf = KNeighborsClassifier()
            param_grid={'n_neighbors': range(2, 7)}

        elif name_model == 'Naive Bayes':
            model_clf = GaussianNB()
            param_grid={'var_smoothing': [1e-8, 1e-9, 1e-10]}

        elif name_model == 'Perceptron':
            model_clf = Perceptron()
            param_grid = {'penalty': [None, 'l2', 'l1', 'elasticnet']}
            
        elif name_model == 'Gaussian Process Classification':
            model_clf = GaussianProcessClassifier()
            param_grid = {'max_iter_predict': [100, 200],
                          'warm_start': [True, False],
                          'n_restarts_optimizer': range(3)}
            
        model = GridSearchCV(model_clf, param_grid=param_grid, cv=cv_train, verbose=False)
        model.fit(train, target)
        
    return model


# In[ ]:


for i in range(N_best_models):
    name_model = models_best.iloc[i]['Model']
    if (name_model == 'LGBMClassifier') or (name_model == 'XGBClassifier'):
        # lgboosting model
        model = model_fit(name_model,train0b,target0)
        acc_metrics_calc_pred(i,model,name_model,train0b,test0b,target0)
    else:
        # model from Sklearn
        model = model_fit(name_model,train0,target0)
        acc_metrics_calc_pred(i,model,name_model,train0,test0,target0)


# In[ ]:


for x in metrics_now:
    xs = metrics_all[x]
    models_pred[xs + '_train'] = acc_all_pred[(x-1)]
models_pred[['Model', 'acc_train']].sort_values(by=['acc_train'], ascending=False)


# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
