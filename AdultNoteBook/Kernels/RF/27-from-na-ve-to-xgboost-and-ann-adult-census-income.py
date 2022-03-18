#!/usr/bin/env python
# coding: utf-8

# # **From Naïve to XGBoost and ANN: Adult Census Income**
# 
# In this kernel, we will use the Adult Census Income dataset from UCI Machine Learning in order to predict if a person earns more than 50k per year or not. Therefore, this will be a binary classification problem where input data is both continous and categorical. In order to make predictions we will develop the following models:
# 
# - Logistic Regression
# - Categorical Naïve Bayes
# - Gaussian Naïve Bayes
# - K-Nearest Neighbors
# - Support Vector Machines
# - Decision Trees
# - Random Forest
# - XGBoost
# - Artificial Neural Networks
# 
# 
# As usual, first we will have a look at the data in order to understand the different variables, then we will clean it in order to use it for prediction purposes, and finally we will develop the different models. These models will be tested with cross validation in order to pick the best ones and use them to develop an ensemble model with the purpose of achieving more accuracy than each member of it. Then, all the models will be tested in a test set in order to determine the best one.
# 

# # **Table of Contents**
# 
# - [1. Descriptive analysis](#1)
#     - [1.1 Target variable](#1.1)
#     - [1.2 Categorical variables](#1.2)
#     - [1.3 Continous variables](#1.3)
#         - [1.3.1 Correlation matrix](#1.3.1)
# - [2. Cleaning data](#2)
#     - [2.1 Drop useless variables](#2.1)
#     - [2.2 Deal with missing data](#2.2)
# - [3. Split data and get dummies](#3)
# - [4. Proposed models](#4)
#     - [4.1 Logistic Regression](#4.1)
#     - [4.2 Categorical Naïve Bayes](#4.2)
#     - [4.3 Gaussian Naïve Bayes](#4.3)
#     - [4.4 K-Nearest Neighbors](#4.4)
#     - [4.5 Support Vector Machines](#4.5)
#     - [4.6 Decision Trees](#4.6)
#     - [4.7 Random Forest](#4.7)
#     - [4.8 XGBoost](#4.8)
#     - [4.9 Artificial Neural Networks](#4.9)
#     - [4.10 Ensembling](#4.10)
# 

# ## **Initial Set-Up**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing

# libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns 

# split into train and test, do grid search for hyperparameter optimization and
# do cross validation with stratified data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

# data scaling
from sklearn.preprocessing import scale, MinMaxScaler
# test data
from sklearn.metrics import accuracy_score

# models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# libraries for ANN
from tensorflow import keras
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier


# # **1. Descriptive analysis** <a class="anchor" id="1"></a>
# 
# In this section we will have a first look at the data and try to understand each variable in the dataset. First of all we read the data:

# In[ ]:


df = pd.read_csv('../input/adult-census-income/adult.csv', sep=",")
df.head()


# In[ ]:


df.shape


# We can see that we have both continous and categorical data. Now we are going to study each of the 15 variables in the dataset.

# # **1.1 Target variable** <a class="anchor" id="1.1"></a>

# ## **Income**

# In[ ]:


# Income is the target variable. 
df['income'].unique() # show unique values


# We substitute the values for ones and zeros and count the number of samples for each value.

# In[ ]:


df['income'].replace(['<=50K','>50K'],[0,1], inplace=True) # replace for 0 and 1
df['income'].value_counts() # show number of samples for each value


# In[ ]:


# we can see the percentage of people that has >50k:
np.mean(df['income'])


# We can see that we have an imbalanced target variable, so we will use stratification when splitting data into train and test sets and when doing cross-validation.

# # **1.2 Categorical variables** <a class="anchor" id="1.2"></a>

# ## **Workclass**

# In[ ]:


df['workclass'].value_counts() 


# As wee see there are 1836 missing values.

# In[ ]:


# probability of belonging to the group with the highest income
workclass_income = df.groupby('workclass')['income'].mean() # there is correlation as spected

plt.rcParams['axes.axisbelow'] = True # grid behind graphs bars
plt.figure(figsize=(20, 8))
plt.ylim(0,1) # values from 0 to 1 as there are probabilities
plt.bar(workclass_income.index.astype(str), workclass_income,
       color = 'SkyBlue' , edgecolor='black' )
plt.ylabel('Probability', size=20)
plt.xlabel('Workclass', size=20)
plt.grid(axis='y')


# Probabilities are what we would expect.

# ## **Education**

# In[ ]:


df['education'].unique() 


# There are not null values.

# In[ ]:


# probability of belonging to the group with the highest income
education_income = df.groupby('education')['income'].mean() # there is correlation as spected

plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.xticks(rotation=30) # rotate axis text
plt.bar(education_income.index.astype(str), education_income,
       color = 'SkyBlue', edgecolor='black' )
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Education', size=20)
plt.grid(axis='y')


# Probabilities are what we would expect: it increases with education.

# ## **Marital.status**

# In[ ]:


df['marital.status'].unique() 


# There are not null values.

# In[ ]:


# probability of belonging to the group with the highest income
marital_income = df.groupby('marital.status')['income'].mean()

plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.bar(marital_income.index.astype(str), marital_income,
       color = 'SkyBlue', edgecolor='black' )
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Marital status', size=20)
plt.grid(axis='y')


# Probabilities are what we would expect. Married people has more probability than the rest.

# ## **Occupation**

# In[ ]:


df['occupation'].value_counts() 


# There are 1843 missing values. It must be correlated with workclass

# In[ ]:


# Show null values in common
work_ocupation = df.loc[df['workclass'] == df['occupation'],'workclass']
work_ocupation.value_counts()


# As we see there are 1836 null values in common.

# In[ ]:


# probability of belonging to the group with the highest income
occupation_income = df.groupby('occupation')['income'].mean()

plt.figure(figsize=(25, 8))
plt.ylim(0,1)
plt.xticks(rotation=30) # rotate axis text
plt.bar(occupation_income.index.astype(str), occupation_income,
       color = 'SkyBlue', edgecolor='black' )
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Occupation', size=20)
plt.grid(axis='y')


# Probabilities are what we would expect.

# ## **Relationship**

# In[ ]:


df['relationship'].value_counts() 


# There are not null values.

# In[ ]:


# probability of belonging to the group with the highest income
relationship_income = df.groupby('relationship')['income'].mean()

plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.bar(relationship_income.index.astype(str), relationship_income,
       color = 'SkyBlue', edgecolor='black')
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Relationship', size=20)
plt.grid(axis='y')


# As we see being married increases the probability of earning more than 50k.

# ## **Race**

# In[ ]:


df['race'].value_counts()


# In[ ]:


race_income = df.groupby('race')['income'].mean()

plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.bar(race_income.index.astype(str), race_income,
       color = 'SkyBlue', edgecolor='black')
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Race', size=20)
plt.grid(axis='y')


# We can see that white people and Asian-Pac-Islander have the higher probabilities.

# ## **Sex**

# In[ ]:


df['sex'].value_counts()


# In[ ]:


sex_income = df.groupby('sex')['income'].mean()

plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.bar(sex_income.index.astype(str), sex_income,
       color = 'SkyBlue', edgecolor='black')
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Sex', size=20)
plt.grid(axis='y')


# As we see male have higher probabilities than female.

# ## **native.country**

# In[ ]:


df['native.country'].unique() 


# We can see that there are missing data.

# In[ ]:


# Show number of missing values
df.loc[df['native.country'] == '?', 'native.country'].count() 


# In[ ]:


# Show if missing values have something to do with occupation missing data
df.loc[df['native.country'] == 'occupation','occupation' ].count()


# In[ ]:


# Show if missing values have something to do with workclass missing data
df.loc[df['native.country'] == 'workclass','workclass' ].count()


# As we see missing data from native.country as nothing to do with missing data from occupation and workclass.  

# # **1.3 Continous variables** <a class="anchor" id="1.3"></a>

#  ## **1.3.1 Correlation matrix** <a class="anchor" id="1.3.1"></a>

# In[ ]:


df.info() # Show continoues variables 


# In[ ]:


# Group all continous variables 
df_continous = df[['income', 'age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']]
# Correlation matrix
plt.figure(figsize=(15, 8))
sns.heatmap(data=df_continous.corr(), annot=True, vmin=-1, vmax=1)


# As we see fnlwgt does not have a high correlation with income so we will drop it.

# ## **Age**

# In[ ]:


df['age'].unique()


# In[ ]:


# plot histogram
plt.figure(figsize=(20, 8))
plt.hist(df['age'],density=True, bins=20, color = 'SkyBlue')
plt.ylabel('Probability', size=20)
plt.xlabel('Age', size=20)
plt.grid(axis='y')


# In[ ]:


# Show average age by income
df.groupby("income")["age"].mean() 


# People with more than 50k are on average older.

# In[ ]:



# divide age into groups
age_range = pd.cut(df['age'], bins = [20,30,40,50,60,70,80,90])

# show probability of belonging to the group with the highest income
age_income = df.groupby(age_range)['income'].mean()

# barplot showing probability of belonging to the group with the highest income per age range
plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.bar(age_income.index.astype(str), age_income, color = 'SkyBlue',
       edgecolor='black')
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Age range', size=20)
plt.grid(axis='y')


# Results match with what we would expect.

# ## **fnlwgt**

# In[ ]:


df.loc[df['fnlwgt'] == '?'] 


# It doesn't have null values.

# ## **education.num**
# This is an ordinal variable for education.

# In[ ]:


df['education.num'].value_counts()


# In[ ]:


df['education'].value_counts() # we can see that it has the same number of values


# ## **capital.gain**

# In[ ]:


df['capital.gain'].unique()


# ## **capital.loss**

# In[ ]:


df['capital.loss'].unique()


# ## **hours.per.week**

# In[ ]:


df['hours.per.week'].unique()


# In[ ]:


# plot histogram
plt.figure(figsize=(15, 8))
plt.hist(df['hours.per.week'],density=True, bins=10,  color = 'SkyBlue')
plt.ylabel('Probability', size=20)
plt.xlabel('hours per week', size=20)
plt.grid(axis='y')


# # **2. Cleaning data** <a class="anchor" id="2"></a>

# ## **2.1 Drop useless variables** <a class="anchor" id="2.1"></a>
# 
# We have seen that fnlwgt variable has a really small correlation with the target variable so we can drop it. We can also drop education.num as if we don't do that it will be a multicollinearity problem with education.
# 

# In[ ]:


df = df.drop('fnlwgt', axis=1)
df = df.drop('education.num', axis=1)
df.shape


# ## **2.2 Deal with missing data** <a class="anchor" id="2.2"></a>
# As we have a lot of data and the missing values is just a small part of the dataset, we drop rows with missing values.

# In[ ]:


df = df.loc[ (df['workclass'] != '?') & (df['occupation'] != '?') & (df['native.country']!= '?')]
df.shape


# # **3. Split data and get dummies** <a class="anchor" id="3"></a>
# First we will split the data into dependent and independent variables, and then we will split the dependent variables into continous variables and categorical variables.

# In[ ]:


# Split into dependend and independent variables
X = df.drop('income', axis=1)
y = df['income']


# In[ ]:


# Split X into continous variables and categorical variables

X_continous  = X[['age', 'capital.gain', 'capital.loss', 'hours.per.week']]

X_categorical = X[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race',
                   'sex', 'native.country']]


# Now we can get the dummies from the categorical variables and concatenate both continous and categorical datasets.

# In[ ]:


# Get the dummies
X_encoded = pd.get_dummies(X_categorical)
# Concatenate both continous and encoded sets:
X = pd.concat([X_continous, X_encoded],axis=1)
X


# # **4. Proposed models** <a class="anchor" id="4"></a>
# 
# In this section we will develop the predictive models. We will stratify the data and use a specific random state so all the models have the same target values.
# It is worth mentioning that in most of the models (the complex ones) I have done hyperparameter optimization with GridSearchCV. As this process takes a lot of time, I have commented the lines where I fit the GridSearch and I comment the results that I had when running it. 

# # **4.1 Logistic Regression** <a class="anchor" id="4.1"></a>

# In[ ]:


# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
                                                    stratify=y,random_state=10 )

# MODEL
logit = LogisticRegression(max_iter=10000)
logit = logit.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3) # we make 3 splits
val_logit = cross_val_score(logit, X_train, y_train, cv=cv).mean()
val_logit # show validation set score


# In[ ]:


# PREDICTIONS
logit_predictions = logit.predict(X_test)
acc_logit = accuracy_score(y_test,logit_predictions)
acc_logit # show test set score


# # **4.2 Categorical Naïve Bayes** <a class="anchor" id="4.2"></a>

# In[ ]:


# Prepare the data. We only use categorical independent variables 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size= 1/3, 
                                                    stratify=y, random_state=10 )

# MODEL
cnb = CategoricalNB()
cnb = cnb.fit(X_train, y_train)

# PREDICTIONS
cnb_predictions = cnb.predict(X_test)
acc_cnb = accuracy_score(y_test,cnb_predictions)
acc_cnb # show test set score


# # **4.3 Gaussian Naïve Bayes** <a class="anchor" id="4.3"></a>

# In[ ]:


# Prepare the data. We only use continous independent variables 
X_train, X_test, y_train, y_test = train_test_split(X_continous, y, test_size= 1/3,
                                                    stratify=y, random_state=10)

# MODEL
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_cnb = cross_val_score(gnb, X_train, y_train, cv=cv).mean()
val_cnb # validation set score


# In[ ]:


# PREDICTIONS
gnb_predictions = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test,gnb_predictions)
acc_gnb # test set score


# # **4.4 K-Nearest Neighbors** <a class="anchor" id="4.4"></a>

# In[ ]:


# Prepare the data. We scale the data as this algorithm is distance-based

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, 
                                                    stratify=y, random_state=10)
# scale data in a range of (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# HYPERPARAMETERS OPTIMIZATION

# set the hyperparameters we want to test
param_grid = {'n_neighbors' : [40, 60, 70]}

cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator = KNeighborsClassifier(),
    param_grid = param_grid,
    scoring = 'accuracy',
    verbose = 2,
    cv = cv,
)

#>>> optimal_params.fit(X_train, y_train)

#>>> optimal_params.best_params_
# {'n_neighbors': 40}


# In[ ]:


# MODEL

knn = KNeighborsClassifier(n_neighbors=40)
knn = knn.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_knn = cross_val_score(knn, X_train, y_train, cv=cv).mean()
val_knn # validation score


# In[ ]:


# PREDICTIONS
knn_predictions = knn.predict(X_test)
acc_knn = accuracy_score(y_test,knn_predictions)
acc_knn # test score


# # **4.5 Support Vector Machines** <a class="anchor" id="4.5"></a>

# In[ ]:


# Prepare the data. We scale the data as this algorithm is distance-based

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
                                                    stratify=y, random_state=10)
# scale the data (mean=0 and sd=1)
X_train = scale(X_train)
X_test = scale(X_test)


# In[ ]:



# HYPERPARAMETERS OPTIMIZATION

# 1 ROUND
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator = svm.SVC(),
    param_grid = param_grid,
    scoring = 'accuracy',
    verbose = 2,
    cv = cv
)

#>>> optimal_params.fit(X_train, y_train,)


#>>> optimal_params.best_params_
# {'kernel': 'linear'}

# As linear kernels are the simplest ones the result is unexpected.
# wW can see how better linear kernels are with respect to the rest.

#>>> opt = optimal_params.cv_results_
#>>> opt = pd.DataFrame.from_dict(opt)
#>>> opt[['params', 'mean_test_score']]
#                   params  mean_test_score
# 0   {'kernel': 'linear'}         0.843346
# 1     {'kernel': 'poly'}         0.818580
# 2      {'kernel': 'rbf'}         0.839865
# 3  {'kernel': 'sigmoid'}         0.830316


# As we can tune parameters from the rbf kernel, we will try to improve
# its performance


# 2 ROUND

param_grid = {
    'kernel': ['rbf'],
    'C' : [1, 2, 3, 4, 5],
}

cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator = svm.SVC(),
    param_grid = param_grid,
    scoring = 'accuracy',
    verbose = 2,
    cv = cv
)

#>>> optimal_params.fit(X_train, y_train,)

#>>> optimal_params.best_params_
#{'C': 2, 'kernel': 'rbf'}

#>>> optimal_params.best_score_
# 0.8398648211769877

# as we see rbf is worse than linear, so linear will be used


# In[ ]:


# MODEL
suppvm = svm.SVC(kernel='linear')
suppvm = suppvm.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_suppvm = cross_val_score(suppvm, X_train, y_train, cv=cv).mean()
val_suppvm # validation score


# In[ ]:


# PREDICTIONS
suppvm_predictions = suppvm.predict(X_test)
acc_suppvm = accuracy_score(y_test,suppvm_predictions)
acc_suppvm # test score


# # **4.6 Decision Trees** <a class="anchor" id="4.6"></a>

# In[ ]:


# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, 
                                                    stratify=y, random_state=10)


# In[ ]:


# HYPERPARAMETERS OPTIMIZATION
param_grid = {
'max_depth' : [2,4,6,7,8,9,10,11,12,16,20]
}

cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator = DecisionTreeClassifier(),
    param_grid = param_grid,
    scoring = 'accuracy',
    verbose = 2,
    cv = cv
)

#>>> optimal_params.fit(X_train, y_train,)

#>>> optimal_params.best_params_
# {'max_depth': 11}


# In[ ]:


# MODEL
tree = DecisionTreeClassifier(max_depth=11)
tree = tree.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_tree = cross_val_score(tree, X_train, y_train, cv=cv).mean()
val_tree # validation score


# In[ ]:


# PREDICTIONS
tree_predictions = tree.predict(X_test)
acc_tree = accuracy_score(y_test,tree_predictions)
acc_tree # test score


# # **4.7 Random Forest** <a class="anchor" id="4.7"></a>

# In[ ]:


# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
                                                    stratify=y, random_state=10)


# In[ ]:


# HYPERPARAMETERS OPTIMIZATION
param_grid = {
'max_depth' : [8,10,12,16,18,20],
'n_estimators': [50,100,200],
'max_samples': [1,0.8,0.6]
}

cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid = param_grid,
    scoring = 'accuracy',
    verbose = 2,
    cv = cv
)

#>>> optimal_params.fit(X_train, y_train,)

#>>> optimal_params.best_params_
#{'max_depth': 16, 'max_samples': 0.6, 'n_estimators': 200}


# In[ ]:


# MODEL
Rforest = RandomForestClassifier(max_depth=16,max_samples=0.6, n_estimators=200)
Rforest = Rforest.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_Rforest = cross_val_score(Rforest, X_train, y_train, cv=cv).mean()
val_Rforest # validation score


# In[ ]:


# PREDICTIONS
Rforest_predictions = Rforest.predict(X_test)
acc_Rforest = accuracy_score(y_test,Rforest_predictions)
acc_Rforest # test score


# # **4.8 XGBoost** <a class="anchor" id="4.8"></a>

# In[ ]:


# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
                                                    stratify=y, random_state=10)


# In[ ]:


# HYPERPARAMETER OPTIMIZATION

# ROUND 1

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.3, 0.1, 0.05],
    'gamma': [0, 1, 10],
    'reg_lambda': [0, 1, 10]
}


cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', #for binary classification
                                eval_metric="logloss",
                                use_label_encoder=False), #avoid warning (since we have done encoding)
    param_grid=param_grid,
    scoring='accuracy',
    verbose=2,
    cv = cv
)

#>>> optimal_params.fit(X_train, y_train,)

#>>> optimal_params.best_params_
#{'gamma': 0, 'learning_rate': 0.3, 'max_depth': 5, 'reg_lambda': 10}



# ROUND 2


param_grid = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.3, 0.5],
    'subsample': [1, 0.8, 0.6, 0.4],
    'gamma' : [10, 50, 100]
}



cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', #for binary classification
                                eval_metric="logloss",
                                learning_rate= 0.1,
                                reg_lambda=0,
                                use_label_encoder=False), #avoid warning (since we have done encoding)
    param_grid=param_grid,
    scoring='accuracy',
    verbose=2,
    cv = cv
)

#>>> optimal_params.fit(X_train, y_train,)

#>>> optimal_params.best_params_
#{'learning_rate': 0.3, 'max_depth': 5, 'reg_lambda': 10, 'subsample': 1}



# In[ ]:


# MODEL
xgbm = xgb.XGBClassifier(eval_metric="logloss",
                        learning_rate= 0.3,
                        reg_lambda=10,
                        use_label_encoder=False, # as we have done encoding
                        max_depth=5,
                        subsample=1)

xgbm = xgbm.fit(X_train, y_train)

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_xgbm = cross_val_score(xgbm, X_train, y_train, cv=cv).mean()
val_xgbm


# In[ ]:


# PREDICTIONS
xgbm_predictions = xgbm.predict(X_test)
acc_xgbm = accuracy_score(y_test,xgbm_predictions)
acc_xgbm


# In[ ]:


# Save predictions with probabilities in order to later make the ensembling
xgbm_predictions_prob = xgbm.predict_proba(X_test)
xgbm_predictions_prob = xgbm_predictions_prob[:,1]


# # **4.9 Artifical Neural Networks** <a class="anchor" id="4.9"></a>

# In[ ]:


# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, 
                                                    stratify=y, random_state=10)
# scale the data (mean=0, sd=1)
X_train = scale(X_train)
X_test = scale(X_test)


# In[ ]:



# HYPERPARAMETERS OPTIMIZATION

# ROUND 1

# first we need to define the model 
def ANN_1(neurons=10, hidden_layers=0, dropout_rate=0, learn_rate= 0.1):
    # model
    model = keras.Sequential()
    model.add(keras.layers.Dense(neurons, input_shape = (X_train.shape[1], ), activation='relu'))
    for i in range(hidden_layers):
        # Add one hidden layer
        model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum = 0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# we will do the grid search with KerasClassifier
ann = KerasClassifier(build_fn=ANN_1, batch_size=30)


param_grid = {
    'neurons': [10, 30, 60, 100, 200],
    'hidden_layers': [0, 1, 2],
    'dropout_rate': [0, 0.1, 0.2, 0.4],
    'epochs': [8,15],
    'learn_rate': [0.1, 0.03]
}

cv = StratifiedKFold(n_splits=3)

optimal_params = GridSearchCV(estimator=ann, param_grid=param_grid, verbose=2, cv=cv)

#>>> optimal_params.fit(X_train, y_train,)

#>>> optimal_params.best_params_
# {'dropout_rate': 0.2, 'epochs': 15, 'hidden_layers': 1, 'learn_rate': 0.1, 'neurons': 10}



# ROUND 2

def ANN_2(init_mode='uniform', activation='relu'):
    # model
    model = keras.Sequential()
    model.add(keras.layers.Dense(10,kernel_initializer=init_mode,
                                 input_shape = (X_train.shape[1], ), activation=activation))
    model.add(keras.layers.Dense(10, kernel_initializer=init_mode,activation=activation))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1,kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


ann = KerasClassifier(build_fn=ANN_2, epochs= 15,  batch_size=30)


param_grid = {
    'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                  'glorot_uniform', 'he_normal', 'he_uniform'],
    'activation': ['softmax','relu', 'tanh', 'sigmoid']
}


cv = StratifiedKFold(n_splits=3)


optimal_params = GridSearchCV(estimator=ann, param_grid=param_grid, verbose=2, cv=cv)

#>>> optimal_params.fit(X_train, y_train)

#>>> optimal_params.best_params_
# {'activation': 'relu', 'init_mode': 'uniform'}


# In[ ]:


# MODEL

def ANN_():
    model = keras.Sequential()
    model.add(keras.layers.Dense(10,kernel_initializer='uniform',
                                 input_shape = (X_train.shape[1], ), activation='relu'))
    model.add(keras.layers.Dense(10, kernel_initializer='uniform',activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    optimizer = keras.optimizers.SGD(lr=0.1, momentum = 0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# we define a learning rate schedule in order to decrease the learning rate
# as we epoch increases.
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  if epoch < 15:
	  return 0.05
  else:
      return 0.01


# Early stopping: stop the learning when it has 3 consecutive epoch without improvement
callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# Learning rate schedule
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

ann = KerasClassifier(build_fn=ANN_, epochs= 15,  batch_size=30, verbose=0)


# CROSS VALIDATION
cv = StratifiedKFold(n_splits=3)
val_ann= cross_val_score(ann, X_train, y_train,
                         cv=cv, fit_params={'callbacks': [callback,callback2]}).mean()
val_ann # validation score


# In[ ]:


# Neural Networks Ensembling
# we will make 10 Neural Networks and then join its predictions by averaging. 


n_members = 10
ann_dict = {} # dictionary where we will store the predictions

for i in range(n_members):
    ann = ANN_()
    ann.fit(X_train, y_train, epochs=15, batch_size=30, 
            verbose=0, callbacks=[callback, callback2])
    ann_predictions = ann.predict(X_test)
    ann_predictions = ann_predictions.reshape(ann_predictions.shape[0], )
    ann_dict["ann%s" %i] = ann_predictions

# create a pandas DataFrame from the dictionary
ann_dataframe = pd.DataFrame.from_dict(ann_dict)

ann_mean_prob = ann_dataframe.mean(axis=1) #averaging all the ANN predictions for each row
ann_mean = np.where(ann_mean_prob > 0.5, 1, 0) # transforem probabilities to a binary variable
acc_ann = accuracy_score(y_test, ann_mean)
acc_ann # test score


# # **4.10 Ensembling** <a class="anchor" id="4.10"></a>

# In[ ]:


# Select the models by looking at the validation score

np.array([val_logit, val_cnb, val_knn, val_suppvm, val_tree, val_Rforest, val_xgbm, val_ann])


# As we see XGBoost outperforms the rest of the models. It it worth mentioning that the validation score for the ANN is for just one ANN, so when doing the Neural Networks Ensemble we expect it to be higher. Thus, we can do the ensembling with XGBoost and ANN. As XGBoost seems to perform better, we will give it a higher weight.

# In[ ]:


# Create dataset with the best predictions
best_predictions = pd.DataFrame(data= {'ann':ann_mean_prob, 
                                       'xgb':xgbm_predictions_prob})

# We give a higher weight to XGBoost
ensembling = best_predictions['ann']* 0.4 + best_predictions['xgb']*0.6

# Probabilities to binary
ensembling_binary = np.where(ensembling > 0.5, 1, 0)
acc_ensembling = accuracy_score(y_test, ensembling_binary)
acc_ensembling # test score


# # **5. Conclusion**
# 
# We can represent the different accuracy results:
# 
# 

# In[ ]:


# make a dictionary with all the results
results = {'Logistic Regression': acc_logit, 
           'Categorical Naive Bayes': acc_cnb,
           'Gaussian Naive Bayes': acc_gnb,
           'K-Nearest Neighbors': acc_knn ,
           'Support Vector Machines': acc_suppvm,
           'Decision Trees':acc_tree ,
           'Random Forest': acc_Rforest,
           'XGBoost':acc_xgbm ,
           'Artificial Neural Networks':acc_ann ,
           'XGBoost-ANN Ensembling': acc_ensembling,         
          }

results_dataframe = pd.DataFrame.from_dict(results, orient='index', 
                                           columns=['Accuracy'])
results_dataframe


# As we have seen, the XGBoost-ANN ensembling model has the best accuracy, but it is not remarkably better than the single XGBoost model. We can conclude that XGBoost model outperforms all the other models. 
