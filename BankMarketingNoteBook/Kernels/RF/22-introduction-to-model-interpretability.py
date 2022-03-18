#!/usr/bin/env python
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:



import pandas as pd

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# algorithms, ranging from easiest to the hardest to intepret.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


# This notebook is to learn about the intepretable ML models like SHAP and LIME. The data set used is the Bank marketing UCI dataset. 
# The reference to this project is the PyData NY conference 2018([Open the Black Box: an Introduction to Model Interpretability with LIME and SHAP - Kevin Lemagnen](https://www.youtube.com/watch?v=C80SQe16Rao&ab_channel=PyData))

# - Models are opinions embedded in mathematics, if there is a bias in your dataset then it will flow into the model
# - Classify images - Wolf and Husky dog- The model worked well, the problem is that the model looked at the snow and then identified Husky. So basically a snow detector!! 

# Why to intepret the models:
# * how the decisions are made?
# * convert to white box models
# * Reduce the bias in the model data
# * harder models like ensembles, boosting, Deep NNs are difficult to interpret
# * if the feature is effecting positively or negatively?

# We are going to look at the following methods:
# - ELI5
# - SHAP
# - LIME

# In[ ]:


data = pd.read_csv('../input/bank-marketing/bank-additional-full.csv', sep=';')
data.head()
# Output variable (desired target):if the client subscribed a term deposit? (binary: "yes","no")


# # Data Preprocessing

# In[ ]:


data.info()


# In[ ]:


# Lets check the target variable to see if it is balanced or not
data['y'].value_counts()
# the dataset is imbalanced


# In[ ]:


#Lets built the attribute set and the target data set
data_y= data['y'].map({'yes':1,'no':0})
data_X= data.drop('y',axis=1)


# **Attribute information:**
# *Input variables:*
# 
# *bank client data:*
# 
# 1 - age (numeric)
# 
# 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services")
# 
# 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
# 
# 4 - education (categorical: "unknown","secondary","primary","tertiary")
# 
# 5 - default: has credit in default? (binary: "yes","no")
# 
# 6 - balance: average yearly balance, in euros (numeric)
# 
# 7 - housing: has housing loan? (binary: "yes","no")
# 
# 8 - loan: has personal loan? (binary: "yes","no")
# 
# *related with the last contact of the current campaign:*
# 9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
# 
# 10 - day: last contact day of the month (numeric)
# 
# 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 
# 12 - duration: last contact duration, in seconds (numeric)
# 
# other attributes:
# 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
# 
# 15 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

# In[ ]:


# Lets check if there is a case of missing data in the columns
#check for the missing values in the train data
data_X.isnull().sum()


# In[ ]:


data_X.drop('duration',inplace=True,axis=1)


# In[ ]:


data_X.dtypes # to look at the data types of the features


# In[ ]:


data_features_cat =  data_X.dtypes[data_X.dtypes == 'object'].index.to_list()
data_features_num =  (data_X.dtypes[data_X.dtypes == 'int64'].index |  data_X.dtypes[data_X.dtypes == 'float64'].index).to_list()


# In[ ]:


print ('categorical features are: \n', data_features_cat)
print('----'*25)
print ('numerical features are: \n', data_features_num)


# * To process these two lists seperately, we can use  a column transformer for that we have to pass in the respective variables as Tuples
# * Each transformer is a three-element tuple that defines the name of the transformer, the transform to apply, and the column indices to 
#   apply it to. For example: (Name, Object, Columns)
# *  https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/

# In[ ]:




preprocessor = ColumnTransformer(transformers=[('numerical','passthrough',data_features_num),('categorical',OneHotEncoder(sparse=False,handle_unknown='ignore'),
                                   data_features_cat)])


# In[ ]:


# The classification models that we are going to try out are:
# The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to 
# class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

# 1. Logistic regression
LGmodel =  LogisticRegression(class_weight='balanced',solver='liblinear',random_state=42)

# 2.Decision tree
DTmodel = DecisionTreeClassifier(class_weight='balanced')

# 3.Random forest
RFmodel = RandomForestClassifier(class_weight='balanced',n_estimators=100, n_jobs=-1)

# 4.XG Boost
XGBmodel =  XGBClassifier(scale_pos_weight=(1 - data_y.mean()), n_jobs=-1)


# * Lets built a pipeline so that we can chain the steps sequentially
# * The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters
# * Gridsearch, CVs etc are all possible on a pipeline
# * Paras → List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last    object an estimator.
# * https://machinelearningmastery.com/machine-learning-modeling-pipelines/

# In[ ]:


# defining the model pipelines

log_reg_model = Pipeline([('preprocessor',preprocessor),('model',LGmodel)])
dt_model = Pipeline([('preprocessor',preprocessor),('model',DTmodel)])
rf_model = Pipeline([('preprocessor',preprocessor),('model',RFmodel)])
xgb_model = Pipeline([('preprocessor',preprocessor),('model',XGBmodel)])


# Train Test Data split

# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(data_X,data_y,stratify=data_y,test_size=0.3,random_state=42)


# # ELI5
# - useful to debug scikit learn models
# - provides global interpretation of white box models
# - show feature importances and explain predictions
# -  Two functions - show weights and show observations

# ## Logistic Regression

# In[ ]:


# fit the log reg model first 
# Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of 
# algorithm parameters specified in a grid.
gs_logreg = GridSearchCV(estimator=log_reg_model, param_grid={"model__C": [1,1.2,1.2, 1.3, 1.4, 1.5]}, n_jobs=-1, cv=5, scoring="accuracy")
gs_logreg.fit(X_train,y_train)


# In[ ]:


# check the best parameters and best score
print(gs_logreg.best_params_)
print(gs_logreg.best_score_)


# In[ ]:


log_reg_model.set_params(**gs_logreg.best_params_)


# In[ ]:


log_reg_model.get_params('model')


# In[ ]:


# fit the best values on the training set again
log_reg_model.fit(X_train,y_train)


# In[ ]:


# Now generate the predictions
y_pred = log_reg_model.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))
# bad recall and precision for the minority class


# In[ ]:


# Let's use eli5 to visualise the weights associated to each feature:


import eli5
eli5.show_weights(log_reg_model.named_steps["model"]) # going into the pipeline and getting the model
# can't get much enough from this as the features are encoded


# In[ ]:


# we need to get back the name of the features from the post one hot encoding
# so the steps are go to the model → preprocessor → categorical feats

preprocessor = log_reg_model.named_steps['preprocessor'] # to get the preprocessor from the model
ohe_feats = preprocessor.named_transformers_['categorical'].categories_ # to get the categories from the transformation inside preprocessor


# In[ ]:


print(data_features_cat)
print('****'*25)
print(ohe_feats)


# In[ ]:


#build a list of featurs by combining the cat_feats with the sub categories
new_ohe_feats =  [f'{col}_{val}' for col,vals in zip(data_features_cat,ohe_feats) for val in vals]


# In[ ]:


print(new_ohe_feats)


# In[ ]:


data_all_features = data_features_num + new_ohe_feats


# In[ ]:


pd.DataFrame(log_reg_model.named_steps["preprocessor"].transform(X_train), columns=data_all_features).head()


# In[ ]:


eli5.show_weights(log_reg_model.named_steps['model'],feature_names = data_all_features)
# Notes
# whether campaign happened in march ? 


# In[ ]:


# show observations
# Take row 4 from the data
test_row = 4
X_test.iloc[[test_row]]


# In[ ]:



y_test.iloc[[test_row]] # =1 was able to get it


# In[ ]:


eli5.show_prediction(log_reg_model.named_steps['model'],
                     log_reg_model.named_steps['preprocessor'].transform(X_test)[test_row],
                     feature_names=data_all_features,show_feature_values = True)


# In[ ]:


# We can see that the modt imp feature that eli5 looked at consumer price index, but that feature is not linked to client but to the company
# it means it is not a good model
# again the feature that negatively effecting the model is the no: employees, again dependent on the company not the client!


# ## Decision trees

# In[ ]:


gs_dt =GridSearchCV(dt_model, {"model__max_depth": [3, 5, 7], 
                             "model__min_samples_split": [2, 5]},n_jobs=1,cv=5,scoring='accuracy')
gs_dt.fit(X_train,y_train)


# In[ ]:


from sklearn import set_config

set_config(display='diagram')
gs_dt


# In[ ]:


#To check the parameters in the pipeline
#sorted(dt_model.get_params().keys())


# In[ ]:


print(gs_dt.best_params_)
print(gs_dt.best_score_)


# In[ ]:


dt_model.set_params(**gs_dt.best_params_)


# In[ ]:


dt_model.fit(X_train,y_train)
y_pred = dt_model.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred)) # still not a good model


# In[ ]:


eli5.show_weights(dt_model.named_steps['model'],feature_names= data_all_features)
#its picking on the no:employees again!! Not a good model!


# In[ ]:


eli5.show_prediction(dt_model.named_steps['model'],
                     dt_model.named_steps['preprocessor'].transform(X_test)[test_row],
                     feature_names = data_all_features, show_feature_values = True)


# # SHAP

# 

# ## Random forests
# 

# In[ ]:


gs_rf =GridSearchCV(rf_model, {"model__max_depth": [10,12,15], 
                             "model__min_samples_split": [2,7,10]},n_jobs=1,cv=5,scoring='accuracy')
gs_rf.fit(X_train,y_train)


# In[ ]:


print(gs_rf.best_params_)
print(gs_rf.best_score_)


# In[ ]:


rf_model.set_params(**gs_rf.best_params_)


# In[ ]:


rf_model.fit(X_train,y_train)
y_pred= rf_model.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# look the features with eli5 first of all
eli5.show_weights(rf_model.named_steps["model"], 
                  feature_names=data_all_features)


# ## Gradient Boosting 

# In[ ]:


gs_xb = GridSearchCV(xgb_model, {"model__max_depth": [5, 10],
                              "model__min_child_weight": [5, 10],
                              "model__n_estimators": [25]},
                  n_jobs=-1, cv=5, scoring="accuracy")

gs_xb.fit(X_train, y_train)


# In[ ]:


# fit model and create Predictions

print(gs_xb.best_params_)
print(gs_xb.best_score_)
xgb_model.set_params(**gs_xb.best_params_)
xgb_model.fit(X_train, y_train)

#predictions
y_pred = xgb_model.predict(X_test)


# In[ ]:


#check accuracy and classification report
print('Accuracy is :',accuracy_score(y_test,y_pred))
print('--'*25)
print(classification_report(y_test,y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# # SHAP

# - Explanation model is a simpler model that is agood approximation of a complex model
# - Local explanation is a linear combination of the features - using shapely values from game theory
# - How a feature is impacting a decision when it is added to the subset
# - Tree explainer (for tree based models) and kernelexplainer for other models
# 

# **Steps**
# - Create a new explainer
# - Calculate shap values
# - Use visualization

# In[ ]:


import shap
shap.initjs()


# In[ ]:


explainer =shap.TreeExplainer(xgb_model.named_steps['model'])


# In[ ]:


observations = xgb_model.named_steps['preprocessor'].transform(X_train.sample(1000,random_state=42))
shap_values= explainer.shap_values(observations)


# In[ ]:


# visualizations
shap_i = 0
shap.force_plot(explainer.expected_value,shap_values[shap_i],features=observations[shap_i],feature_names=data_all_features)


# **Inferences**
# - how diff features influence my decision
# - red features pushes the features to 1 and blue push the decions to class 0

# In[ ]:


shap.force_plot(explainer.expected_value, shap_values,
                features=observations, feature_names=data_all_features)


# - Here we look at all observations at once
# - can look at individual features in particular

# In[ ]:


shap.summary_plot(shap_values, features=observations, feature_names=data_all_features)


# - gloabal explanation of the model
# - sorted on the order of importance

# In[ ]:


shap.dependence_plot("nr.employed", shap_values, 
                     pd.DataFrame(observations, columns=data_all_features))


# In[ ]:




