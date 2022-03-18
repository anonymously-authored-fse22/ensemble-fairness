#!/usr/bin/env python
# coding: utf-8

# # Titanic - Hyperparameter tuning with GridSearchCV
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/3136/logos/header.png)

# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:#7ca4cd; border:0' role="tab" aria-controls="home"><center>Quick Navigation</center></h3>
# 
# * [1. Data loading and feature engineering](#1)
# * [2. Decision Tree](#2)
# * [3. Random Forest](#3)
# * [4. AdaBoost](#4)
# * [5. XGBoost](#5)
# * [6. LightGBM](#6)
# * [7. CatBoost](#7)
# * [8. Logistic Regression](#8)
# * [9. SVC](#9)
# * [10. K-Nearest Neighbors](#10)
# * [Submission](#100)

# In[ ]:


import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    

SEED = 42
set_seed(SEED)


# <a id="1"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>Data loading and feature engineering<center><h2>

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# Check train samples

# In[ ]:


print(f"Train shape: {train_df.shape}")
train_df.sample(3)


# Check test samples

# In[ ]:


print(f"Test shape: {test_df.shape}")
test_df.sample(3)


# Concatenate train and test data together to exploratory analysis

# In[ ]:


full_df = pd.concat(
    [
        train_df.drop(["PassengerId", "Survived"], axis=1), 
        test_df.drop(["PassengerId"], axis=1),
    ]
)
y_train = train_df["Survived"].values


# Lets check missed values

# In[ ]:


full_df.isna().sum()


# Age and Cabin have a lot of NULL values - we can ignore them.   

# In[ ]:


full_df = full_df.drop(["Age", "Cabin"], axis=1)


# Check the distribution of features below to try to fill not so big NULL valued columns

# In[ ]:


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(full_df["Fare"], bins=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Fare distribution", fontsize=16)

plt.subplot(1, 2, 2)
embarked_info = full_df["Embarked"].value_counts()
plt.bar(embarked_info.index, embarked_info.values)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Embarked distribution", fontsize=16);


# Let's fill the Embarked column with more frequently value "S".   
# The column Fare fill with a mean value

# In[ ]:


full_df["Embarked"].fillna("S", inplace=True)
full_df["Fare"].fillna(full_df["Fare"].mean(), inplace=True)


# Extract titles of people from their names

# In[ ]:


full_df["Title"] = full_df["Name"].str.extract(" ([A-Za-z]+)\.")
full_df["Title"] = full_df["Title"].replace(["Ms", "Mlle"], "Miss")
full_df["Title"] = full_df["Title"].replace(["Mme", "Countess", "Lady", "Dona"], "Mrs")
full_df["Title"] = full_df["Title"].replace(["Dr", "Major", "Col", "Sir", "Rev", "Jonkheer", "Capt", "Don"], "Mr")
full_df = full_df.drop(["Name"], axis=1)


# Encode categories as numbers

# In[ ]:


full_df["Sex"] = full_df["Sex"].map({"male": 1, "female": 0}).astype(int)    
full_df["Embarked"] = full_df["Embarked"].map({"S": 1, "C": 2, "Q": 3}).astype(int)    
full_df['Title'] = full_df['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}).astype(int)   


# Extract ticket numbers from ticket column (some tickets have the same number - it can be useful)

# In[ ]:


full_df["TicketNumber"] = full_df["Ticket"].str.split()
full_df["TicketNumber"] = full_df["TicketNumber"].str[-1]
full_df["TicketNumber"] = LabelEncoder().fit_transform(full_df["TicketNumber"])
full_df = full_df.drop(["Ticket"], axis=1)


# Create new features:
# - FamilySize - total number of family members in the ship
# - IsAlone - the person has some family or traveled alone?

# In[ ]:


full_df["FamilySize"] = full_df["SibSp"] + full_df["Parch"] + 1
full_df["IsAlone"] = full_df["FamilySize"].apply(lambda x: 1 if x == 1 else 0)


# In[ ]:


full_df.head()


# Let's split the data back into training and test data

# In[ ]:


X_train = full_df[:y_train.shape[0]]
X_test = full_df[y_train.shape[0]:]

print(f"Train X shape: {X_train.shape}")
print(f"Train y shape: {y_train.shape}")
print(f"Test X shape: {X_test.shape}")


# Let's create one-hot categorical representations and normalize numerical columns for the gradient-based models

# In[ ]:


one_hot_cols = ["Embarked", "Title"]
for col in one_hot_cols:
    full_df = pd.concat(
        [full_df, pd.get_dummies(full_df[col], prefix=col)], 
        axis=1, 
        join="inner",
    )
full_df = full_df.drop(one_hot_cols, axis=1)


# In[ ]:


scaler = StandardScaler()
full_df.loc[:] = scaler.fit_transform(full_df)


# In[ ]:


full_df


# In[ ]:


X_train_norm = full_df[:y_train.shape[0]]
X_test_norm = full_df[y_train.shape[0]:]

print(f"Train norm X shape: {X_train_norm.shape}")
print(f"Train y shape: {y_train.shape}")
print(f"Test norm X shape: {X_test_norm.shape}")


# Let's highlight categorical features in one list, since they may be needed for some models

# In[ ]:


categorical_columns = ['Sex', 'Embarked', 'Title', 'TicketNumber', 'IsAlone']


# Save mean cross-validated accuracy scores of best models

# In[ ]:


cross_valid_scores = {}


# <a id="2"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>Decision Tree<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "max_depth": [3, 5, 7, 9, 11, 13],\n}\n\nmodel_desicion_tree = DecisionTreeClassifier(\n    random_state=SEED,\n    class_weight=\'balanced\',\n)\n\nmodel_desicion_tree = GridSearchCV(\n    model_desicion_tree, \n    parameters, \n    cv=5,\n    scoring=\'accuracy\',\n)\n\nmodel_desicion_tree.fit(X_train, y_train)\n\nprint(\'-----\')\nprint(f\'Best parameters {model_desicion_tree.best_params_}\')\nprint(\n    f\'Mean cross-validated accuracy score of the best_estimator: \' + \\\n    f\'{model_desicion_tree.best_score_:.3f}\'\n)\ncross_valid_scores[\'desicion_tree\'] = model_desicion_tree.best_score_\nprint(\'-----\')')


# <a id="3"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>Random Forest<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "n_estimators": [5, 10, 15, 20, 25], \n    "max_depth": [3, 5, 7, 9, 11, 13],\n}\n\nmodel_random_forest = RandomForestClassifier(\n    random_state=SEED,\n    class_weight=\'balanced\',\n)\n\nmodel_random_forest = GridSearchCV(\n    model_random_forest, \n    parameters, \n    cv=5,\n    scoring=\'accuracy\',\n)\n\nmodel_random_forest.fit(X_train, y_train)\n\nprint(\'-----\')\nprint(f\'Best parameters {model_random_forest.best_params_}\')\nprint(\n    f\'Mean cross-validated accuracy score of the best_estimator: \'+ \\\n    f\'{model_random_forest.best_score_:.3f}\'\n)\ncross_valid_scores[\'random_forest\'] = model_random_forest.best_score_\nprint(\'-----\')')


# <a id="4"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>AdaBoost<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "n_estimators": [5, 10, 15, 20, 25, 50, 75, 100], \n    "learning_rate": [0.001, 0.01, 0.1, 1.],\n}\n\nmodel_adaboost = AdaBoostClassifier(\n    random_state=SEED,\n)\n\nmodel_adaboost = GridSearchCV(\n    model_adaboost, \n    parameters, \n    cv=5,\n    scoring=\'accuracy\',\n)\n\nmodel_adaboost.fit(X_train, y_train)\n\nprint(\'-----\')\nprint(f\'Best parameters {model_adaboost.best_params_}\')\nprint(\n    f\'Mean cross-validated accuracy score of the best_estimator: \'+ \\\n    f\'{model_adaboost.best_score_:.3f}\'\n)\ncross_valid_scores[\'ada_boost\'] = model_adaboost.best_score_\nprint(\'-----\')')


# <a id="5"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>XGBoost<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'max_depth': [3, 5, 7, 9], \n    'n_estimators': [5, 10, 15, 20, 25, 50, 100],\n    'learning_rate': [0.01, 0.05, 0.1]\n}\n\nmodel_xgb = xgb.XGBClassifier(\n    random_state=SEED,\n)\n\nmodel_xgb = GridSearchCV(\n    model_xgb, \n    parameters, \n    cv=5,\n    scoring='accuracy',\n)\n\nmodel_xgb.fit(X_train, y_train)\n\nprint('-----')\nprint(f'Best parameters {model_xgb.best_params_}')\nprint(\n    f'Mean cross-validated accuracy score of the best_estimator: ' + \n    f'{model_xgb.best_score_:.3f}'\n)\ncross_valid_scores['xgboost'] = model_xgb.best_score_\nprint('-----')")


# <a id="6"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>LightGBM<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'n_estimators': [5, 10, 15, 20, 25, 50, 100],\n    'learning_rate': [0.01, 0.05, 0.1],\n    'num_leaves': [7, 15, 31],\n}\n\nmodel_lgbm = lgbm.LGBMClassifier(\n    random_state=SEED,\n    class_weight='balanced',\n)\n\nmodel_lgbm = GridSearchCV(\n    model_lgbm, \n    parameters, \n    cv=5,\n    scoring='accuracy',\n)\n\nmodel_lgbm.fit(\n    X_train, \n    y_train, \n    categorical_feature=categorical_columns\n)\n\nprint('-----')\nprint(f'Best parameters {model_lgbm.best_params_}')\nprint(\n    f'Mean cross-validated accuracy score of the best_estimator: ' + \n    f'{model_lgbm.best_score_:.3f}'\n)\ncross_valid_scores['lightgbm'] = model_lgbm.best_score_\nprint('-----')")


# <a id="7"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>CatBoost<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {\n    'iterations': [5, 10, 15, 20, 25, 50, 100],\n    'learning_rate': [0.01, 0.05, 0.1],\n    'depth': [3, 5, 7, 9, 11, 13],\n}\n\nmodel_catboost = cb.CatBoostClassifier(\n    verbose=False,\n)\n\nmodel_catboost = GridSearchCV(\n    model_catboost, \n    parameters, \n    cv=5,\n    scoring='accuracy',\n)\n\nmodel_catboost.fit(X_train, y_train)\n\nprint('-----')\nprint(f'Best parameters {model_catboost.best_params_}')\nprint(\n    f'Mean cross-validated accuracy score of the best_estimator: ' + \n    f'{model_catboost.best_score_:.3f}'\n)\ncross_valid_scores['catboost'] = model_catboost.best_score_\nprint('-----')")


# <a id="8"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>Logistic Regression<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "C": [0.001, 0.01, 0.1, 1.],\n    "penalty": ["l1", "l2"]\n}\n\nmodel_logistic_regression = LogisticRegression(\n    random_state=SEED,\n    class_weight="balanced",\n    solver="liblinear",\n)\n\nmodel_logistic_regression = GridSearchCV(\n    model_logistic_regression, \n    parameters, \n    cv=5,\n    scoring=\'accuracy\',\n)\n\nmodel_logistic_regression.fit(X_train_norm, y_train)\n\nprint(\'-----\')\nprint(f\'Best parameters {model_logistic_regression.best_params_}\')\nprint(\n    f\'Mean cross-validated accuracy score of the best_estimator: \' + \n    f\'{model_logistic_regression.best_score_:.3f}\'\n)\ncross_valid_scores[\'logistic_regression\'] = model_logistic_regression.best_score_\nprint(\'-----\')')


# <a id="9"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>SVC<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "C": [0.001, 0.01, 0.1, 1.],\n    "kernel": ["linear", "poly", "rbf", "sigmoid"],\n    "gamma": ["scale", "auto"],\n}\n\nmodel_svc = SVC(\n    random_state=SEED,\n    class_weight="balanced",\n    probability=True,\n)\n\nmodel_svc = GridSearchCV(\n    model_svc, \n    parameters, \n    cv=5,\n    scoring=\'accuracy\',\n)\n\nmodel_svc.fit(X_train_norm, y_train)\n\nprint(\'-----\')\nprint(f\'Best parameters {model_svc.best_params_}\')\nprint(\n    f\'Mean cross-validated accuracy score of the best_estimator: \' + \n    f\'{model_svc.best_score_:.3f}\'\n)\ncross_valid_scores[\'svc\'] = model_svc.best_score_\nprint(\'-----\')')


# <a id="10"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>K-Nearest Neighbors<center><h2>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {\n    "weights": ["uniform", "distance"],\n}\n\nmodel_k_neighbors = KNeighborsClassifier(\n)\n\nmodel_k_neighbors = GridSearchCV(\n    model_k_neighbors, \n    parameters, \n    cv=5,\n    scoring=\'accuracy\',\n)\n\nmodel_k_neighbors.fit(X_train_norm, y_train)\n\nprint(\'-----\')\nprint(f\'Best parameters {model_k_neighbors.best_params_}\')\nprint(\n    f\'Mean cross-validated accuracy score of the best_estimator: \' + \n    f\'{model_k_neighbors.best_score_:.3f}\'\n)\ncross_valid_scores[\'k_neighbors\'] = model_k_neighbors.best_score_\nprint(\'-----\')')


# <a id="100"></a>
# <h2 style='background:#7ca4cd; border:0; color:white'><center>Submission<center><h2>

# In[ ]:


pd.DataFrame(cross_valid_scores, index=['cross_valid_score']).T


# In[ ]:


def create_submission(model, X_test, test_passenger_id, model_name):
    y_pred_test = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {
            'PassengerId': test_passenger_id, 
            'Survived': (y_pred_test >= 0.5).astype(int),
        }
    )
    submission.to_csv(f"submission_{model_name}.csv", index=False)
    
    return y_pred_test


# In[ ]:


test_pred_decision_tree = create_submission(
    model_desicion_tree, X_test, test_df["PassengerId"], "decision_tree"
)
test_pred_random_forest = create_submission(
    model_random_forest, X_test, test_df["PassengerId"], "random_forest"
)
test_pred_adaboost = create_submission(
    model_adaboost, X_test, test_df["PassengerId"], "adaboost"
)
test_pred_xgboost = create_submission(
    model_xgb, X_test, test_df["PassengerId"], "xgboost"
)
test_pred_lightgbm = create_submission(
    model_lgbm, X_test, test_df["PassengerId"], "lightgbm"
)
test_pred_catboost = create_submission(
    model_catboost, X_test, test_df["PassengerId"], "catboost"
)
test_pred_logistic_regression = create_submission(
    model_logistic_regression, X_test_norm, test_df["PassengerId"], "logistic_regression"
)
test_pred_svc = create_submission(
    model_svc, X_test_norm, test_df["PassengerId"], "svc"
)
test_pred_k_neighbors = create_submission(
    model_k_neighbors, X_test_norm, test_df["PassengerId"], "k_neighbors"
)


# In[ ]:


test_pred_merged = (
    test_pred_decision_tree + 
    test_pred_random_forest + 
    test_pred_adaboost +
    test_pred_xgboost + 
    test_pred_lightgbm + 
    test_pred_catboost +
    test_pred_logistic_regression + 
    test_pred_svc +
    test_pred_k_neighbors
)
test_pred_merged = np.round(test_pred_merged / 9)


# In[ ]:


submission = pd.DataFrame(
    {
        'PassengerId': test_df["PassengerId"], 
        'Survived': test_pred_merged.astype(int),
    }
)
submission.to_csv(f"submission_merged.csv", index=False)


# In[ ]:




