#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from warnings import filterwarnings as filt
from scipy.stats import skew, norm 
pd.options.display.max_columns = None

filt('ignore')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.shape


# In[ ]:


df.columns = [c.replace('.','_') for c in df.columns]
df.head()


# In[ ]:


df.info()


# In[ ]:


df.nunique()


# ### handling null values 

# In[ ]:


df.isnull().values.sum()


# instead of Nan they are ? in this datasets 

# In[ ]:


df.workclass.unique()


# In[ ]:


df.occupation.unique()


# In[ ]:


def null_idx(df):
    idx = []
    for col in df.columns:
        i = df[df[col] == '?'].index.values.tolist()
        idx.append(i)
        
    return idx

def null_col(df, null_idx = None):
    if null_idx:
        n = pd.DataFrame({
            'null' : [len(i) for i in null_idx],
            'null_per' : [len(i) / df.shape[0] for i in null_idx]
        }, index = df.columns).sort_values('null', ascending = False)
        
        return n[n.null > 0]
    
    n = pd.DataFrame(df.isnull().sum(), columns = ['nans']) 
    return n[n.nans > 0]


# In[ ]:


df[df.workclass == '?'].head()


# In[ ]:


df[df.workclass == '?'].describe()


# maybe we can fill in the missing value from insights but before that lets check the percent of null values

# In[ ]:


null_indx = null_idx(df)
nans = null_col(df, null_indx)
nans


# occupation and workclass seems to be correlated and percentage of null values are less than 0.1 so its better to drop them 

# In[ ]:


idx_to_drop = []
for i in null_indx:
    idx_to_drop = idx_to_drop + i
idx_to_drop = np.unique(idx_to_drop)
idx_to_drop.shape[0] / df.shape[0]


# In[ ]:


df = df.drop(idx_to_drop)
df.head()


# In[ ]:


cate_feats = df.select_dtypes(include = 'object')
num_feats = df.select_dtypes(exclude = 'object')


# In[ ]:


# unique values for categorical feats 

for col in cate_feats.columns:
    print()
    print(f" {col} ".center(60,'='))
    print(df[col].unique())


# In[ ]:


sns.distplot(df['hours_per_week'][(df.workclass == 'Without-pay') & (df.sex == 'Male')])
sns.distplot(df['hours_per_week'][(df.workclass == 'Without-pay') & (df.sex == 'Female')])
plt.title('hours per week distribution for adults without pay')
plt.legend(['Male', 'Female'])


# it doesnt matter what kind education anyone undergo, if their workclass is without pay then their income is <= 50k which is obvious

# In[ ]:


sns.countplot(df.income)


# In[ ]:


sns.countplot(df.income, hue = df.sex)
plt.title('count plot for income and sex')


# there are lot of men who get > 50k than women, this isn't surprising either  

# In[ ]:


sns.countplot(df.workclass, hue = df.sex)
plt.title('count plot for workclass and sex')


# In[ ]:


print(df['marital_status'].value_counts())
sns.countplot(df['marital_status'])
plt.xticks(rotation = 90);


# In[ ]:


df[df['marital_status'] == 'Married-AF-spouse'].head()


# In[ ]:


print(df.race.value_counts() / df.shape[0])
sns.countplot(df.race , hue = df.income)


# ### data cleaning 

# In[ ]:


df['income'] = df.income.apply(lambda x : 0 if x == '<=50K' else 1)
df['sex'] = df.sex.apply(lambda x : 0 if x == 'Female' else 1)


# one hot encoding 

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

def hot_encode(df, cols):
    dummies = pd.get_dummies(df[cols])
    df = df.drop(cols, axis = 1)
    return pd.concat([df,dummies], axis = 1)


# In[ ]:


pd.DataFrame(df[cate_feats.columns].nunique()).sort_values(0, ascending = False)


# In[ ]:


ord_enc = OrdinalEncoder()

to_enc = ['race', 'relationship','marital_status', 'workclass']
to_ord_enc = ['occupation', 'education', 'native_country']

df = hot_encode(df, to_enc)
df[to_ord_enc] = ord_enc.fit_transform(df[to_ord_enc])
df.head()


# In[ ]:


from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from pdpbox.pdp import *
from shap import TreeExplainer, force_plot
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


def permImp(x, y):
    model = RandomForestClassifier().fit(x, y)
    perm = PermutationImportance(model).fit(x, y)
    return show_weights(perm, feature_names = x.columns.tolist())

def plot_mi(score):
    score = score.sort_values('mi_score', ascending = True)
    plt.barh(score.index, score.mi_score)
    plt.title('important features')
    
def mi_score(x, y, std = True):
    if std:
        x = pd.DataFrame(StandardScaler().fit_transform(x), columns = x.columns)
    score = pd.DataFrame(mutual_info_classif(x, y, discrete_features = False), index = x.columns , columns = ['mi_score']).sort_values('mi_score', ascending = False)
    plot_mi(score)
    return score
    
def isolate(x, y, col):
    model = RandomForestClassifier().fit(x, y)
    pdp_dist = pdp_isolate(model , model_features = x.columns, dataset = x, feature = col)
    return pdp_plot(pdp_dist, feature_name = col)

def interact(x, y, cols):
    model = RandomForestClassifier().fit(x, y)
    pdp_dist = pdp_interact(model , model_features = x.columns, dataset = x, features = cols)
    return pdp_interact_plot(pdp_dist, feature_names = col)

def forceplt(x, y, n_cls = 1):
    idx = y[y == 1].sample(n = 1).index
    x_samp = x.loc[idx]
    print(f'check the sample index : {idx}')
    
    model = RandomForestClassifier().fit(x, y)
    explainer = TreeExplainer(model)
    shap_value = explainer.shap_values(x_samp)[n_cls]
    exp_value = explainer.expected_value[n_cls]
    return force_plot(exp_value, shap_value, feature_names = x.columns.tolist())


# In[ ]:


x = df.drop(['income'], axis = 1)
y = df.income
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123, stratify = y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


df.shape[0] * 0.5


# In[ ]:


sns.heatmap(df.loc[:, :'income'].sample(frac = 0.5).corr(), fmt = '.1f', annot = True)


# let's get the important features 

# In[ ]:


permImp(x_train, y_train)


# In[ ]:


mscore = mi_score(x_train, y_train)


# In[ ]:


isolate(x_train, y_train, 'education')


# In[ ]:


df[to_ord_enc][df.education > 6].head(10)


# In[ ]:


set(ord_enc.inverse_transform(df[to_ord_enc][df.education > 6].head(10))[:, 1])


# according to that pd plot, whoever have education like the above cell all have a chance of getting an income above 50k

# ### model building 

# In[ ]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# In[ ]:


def sample(x, y, frac = 0.5):
    x_big, x, y_big, y = train_test_split(x, y, test_size = frac, stratify = y)
    return x, y

def best_model(x, y, frac = 0, folds = 5):
    
    if frac > 0:
        x, y = sample(x, y)
        
    models = [SVC(), GaussianNB(), RandomForestClassifier(), XGBClassifier(), LGBMClassifier()]
    mnames = ['svm', 'naive bayes', 'random forest', 'xgboost', 'lgbm']
    scalers = [None, StandardScaler(), RobustScaler(), MinMaxScaler()]
    snames = ['none', 'std', 'robust', 'min max']
    scores = [[] for _ in range(4)]
    
    print(f"total number of iterations : {len(models) * len(scalers)}")
    iterr = 0
    for model in models:
        for ind, scaler in enumerate(scalers):
            iterr += 1
            print(f"iteration :===> {iterr} / {len(models) * len(scalers)}")
            if scaler:
                model = Pipeline(steps = [('scaler', scaler), ('model', model)])
            cv = StratifiedKFold(folds, shuffle = True)
            score = cross_val_score(model, x, y, cv = cv, scoring = 'f1_micro').mean()
            scores[ind].append(score)
    print()
    return pd.DataFrame(scores, index = snames, columns = mnames).T

def get_score(xt, yt, xtest, ytest, model, scaler = None, frac = 0):
    if frac > 0:
        xt, yt = sample(xt, yt)
        
    if scaler:
        model = Pipeline(steps = [('scaler', scaler), ('model', model)])
    
    model.fit(xt, yt)
    pred = model.predict(xtest)
    
    print(' Report '.center(70, '='))
    print()
    
    print(f"Training score : {model.score(xt, yt)}")
    print(f"Testing score : {model.score(xtest, ytest)}")
    print(f"Roc Auc score : {roc_auc_score(ytest, pred)}")
    print()
    
    print(classification_report(ytest, pred))
    sns.heatmap(confusion_matrix(ytest, pred), fmt = '.1f', annot = True)
    plt.xlabel('predicted value')
    plt.ylabel('actual value')
    
def gridcv(x, y, model, params, scaler = None, frac = 0, fold = 5):
    if frac > 0:
        x, y = sample(x, y)
        
    if scaler:
        model = Pipeline(steps = [('scaler', scaler), ('model', model)])
    
    cv = StratifiedKFold(fold, shuffle = True)
    clf = GridSearchCV(model, param_grid = params, cv = cv, return_train_score = True, scoring = 'f1_micro')
    clf.fit(x, y)
    res = pd.DataFrame(clf.cv_results_).sort_values('mean_test_score', ascending = False)
    return clf.best_estimator_, clf.best_params_, res[['mean_train_score', 'mean_test_score', 'params']]

def plot_cv(results):
    sns.lineplot(x = results.reset_index().index, y = results.mean_train_score)
    sns.lineplot(x = results.reset_index().index, y = results.mean_test_score)
    plt.title('f1 micro score comparision for training and testing')
    plt.legend(['training score', 'testing score'])    


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_model(x_train, y_train)')


# In[ ]:


params = {
    'n_estimators' : [125, 150, 170],
    'num_leaves' : [25, 27],
    'max_depth' : [-1, 8, 7],
    'reg_alpha' : [0.1, 0.5, 0.8],
    'class_weight' : [None, 'balanced']
}

pip_params = {f"model__{key}" : values for key, values in params.items()}
pip_params


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf, best_params, results = gridcv(x_train, y_train, LGBMClassifier(), pip_params, MinMaxScaler(), 0, 10)')


# In[ ]:


plot_cv(results)


# In[ ]:


results.iloc[0], best_params


# In[ ]:


get_score(x_train, y_train, x_test, y_test, clf)


# In[ ]:


from sklearn.utils import class_weight

class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)


# In[ ]:


get_score(x_train, y_train, x_test, y_test, LGBMClassifier(max_depth = 5, n_estimators = 200, num_leaves = 27, reg_alpha = 0.5, class_weight = {0 : 2, 1: 3.0}, learning_rate = 0.05, reg_lambda= 1), RobustScaler())


# let's see the how the model using these feats to pred the outcome 

# In[ ]:


# lets see how it finding if an adult gets the income > 50k
import shap

shap.initjs()
forceplt(x_train, y_train, 1)


# from the above force plot, age plays a major factor here, let's see the what those features values are 

# In[ ]:


pd.DataFrame(df.loc[18050, ]).T


# In[ ]:


ord_enc.inverse_transform(df[to_ord_enc].loc[18050].values.reshape(1,-1))


# In[ ]:




