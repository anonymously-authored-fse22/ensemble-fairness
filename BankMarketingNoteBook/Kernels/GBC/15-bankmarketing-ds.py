#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# This dataset is a well-known finance related banking dataset. This has information related to the customers who subscribed to 'term deposit'. Aim also is to identify if people would be subscribed or not.

# In[ ]:


df_raw = pd.read_csv("../input/bank-marketing/bank-additional-full.csv",delimiter=";")
df = df_raw.copy()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.head(10)


# In[ ]:


df['y'].value_counts()


# In[ ]:


df['y'].value_counts(normalize=True)


# Highly imbalanced dataset with not subscribing being around 89% and subscribing (desired output) being only 11%. Imbalance treatment is an important step for this.

# Univariate Analysis

# Unique values in categorical features

# In[ ]:


for columns in df.columns:
    if df[columns].dtype == np.object:
        uniqueVals = df[columns].unique()
        print("Column {}'s unique values::: count {}".format(columns, len(uniqueVals)))
        for each in uniqueVals:
            print("    {}".format(each))


# Understanding every categorical value's counts in the dataset

# In[ ]:


for columns in df.columns:
    if df[columns].dtype == np.object and columns not in ['education','job']:
        uniqueVals = df[columns].unique()
        print("Count plot for:: "+columns)
        sns.countplot(x=columns, data = df, order= uniqueVals)
        plt.show()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15,5)
sns.countplot(x = "job", data = df)
ax.set_xlabel('Job', fontsize = 12)
ax.set_ylabel('Count', fontsize = 12)
ax.set_title("Job Count Distribution", fontsize = 13)


# In[ ]:


fig, ax2 = plt.subplots()
sns.countplot(x = "education", data = df, ax = ax2)
ax2.set_title("Education distribution", fontsize = 13)
ax2.set_xlabel("Education level", fontsize = 12)
ax2.set_ylabel("Count", fontsize = 12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 70)
plt.show()
df['education'].value_counts()


# In[ ]:


df['y'].value_counts().plot(kind="bar", title="Target Count")


# Imbalance Treatment
# 
# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
# https://www.kaggle.com/residentmario/undersampling-and-oversampling-imbalanced-data

# In[ ]:


import imblearn
from imblearn.under_sampling import TomekLinks


# In[ ]:


y = (df['y'] == "yes")*1
X = df.drop('y', axis=1)


# In[ ]:


y.value_counts()


# In[ ]:


X.head()


# *Encoding*
# 
# Different encoding techniques:: https://heartbeat.fritz.ai/hands-on-with-feature-engineering-techniques-encoding-categorical-variables-be4bc0715394

# In[ ]:


df['poutcome'].unique()


# In[ ]:


X['contact'] = (X['contact'] == 'telephone') * 1


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


labEncoder = LabelEncoder()


# In[ ]:


X.columns


# In[ ]:


X['job'] = labEncoder.fit_transform(X['job'])
X['marital'] = labEncoder.fit_transform(X['marital'])
X['education'] = labEncoder.fit_transform(X['education'])
X['default'] = labEncoder.fit_transform(X['default'])
X['housing'] = labEncoder.fit_transform(X['housing'])
X['loan'] = labEncoder.fit_transform(X['loan']) 
X['month'] = labEncoder.fit_transform(X['month'])
X['day_of_week'] = labEncoder.fit_transform(X['day_of_week'])
X['duration'] = labEncoder.fit_transform(X['duration'])
X['poutcome'] = labEncoder.fit_transform(X['poutcome'])
#X['loan'] = labEncoder.fit_transform(X['loan'])


# In[ ]:


X.head()


# In[ ]:


tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)

#print('Removed indexes:', id_tl)


# In[ ]:


y_tl.value_counts()


# In[ ]:


y_tl.value_counts()


# In[ ]:


y.value_counts()


# In[ ]:


val = 100 - (y_tl.value_counts()[0]/y.value_counts()[0] * 100)
print("Tomek Links - Under sampling percentage::"+ str(val))


# In[ ]:


df_tl = X_tl.copy()
df_tl['y_tl'] = y_tl


# In[ ]:





# In[ ]:


from imblearn.under_sampling import AllKNN
allKnn = AllKNN()


# In[ ]:





# In[ ]:


x_knn, y_knn  = allKnn.fit_resample(X,y)


# In[ ]:


df_knn = x_knn.copy()
df_knn['y_knn'] = y_knn


# In[ ]:



val = 100 - (y_knn.value_counts()[0]/y.value_counts()[0] * 100)
print("AllKNN - Under sampling percentage::"+ str(val))


# In[ ]:


from imblearn.over_sampling import SMOTENC


# In[ ]:


X.head()


# In[ ]:


smnc = SMOTENC(categorical_features = [1,2,3,4,5,6,7,8,9,13,14])
X_sm, y_sm = smnc.fit_resample(X,y)


# In[ ]:


y_sm.value_counts()
y.value_counts()


# In[ ]:


df_sm = X_sm.copy()
df_sm['y_sm'] = y_sm 


# In[ ]:


val = (y_sm.value_counts()[1]/y.value_counts()[1] * 100)
print("SMOTENC - Over sampling percentage::"+ str(val))


# In[ ]:


df_sm.to_csv('overSampleSMOTENC.csv')
df_knn.to_csv('underSampleAllKNN.csv')
df_tl.to_csv('underSampleTomek.csv')


# *Missing Value Imputation*
# 
# https://www.kaggle.com/solegalli/feature-selection-with-feature-engine

# In[ ]:


get_ipython().system('pip install feature-engine')


# In[ ]:


import feature_engine as fe
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures


# In[ ]:


constant = DropConstantFeatures(tol=0.98)
constant.fit(X)
constant.features_to_drop_


# In[ ]:


dup = DropDuplicateFeatures()
dup.fit(X)
dup.duplicated_feature_sets_


# In[ ]:


## 999 means
X['pdays'].value_counts()


# This dataset does not contain missing values except the pdays feature. Looks like this feature has missing value 999 in majority of the dataset. It is wiser to remove this feature from the dataset. But 999 means never contacted. So it is another potential feature value. So, we should keep this.
# 
# Understanding the Data and it's nature is important in this sense.

# *Encoding*

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df['job'].unique()


# In this dataset, ordinal categorical features is **education**. So, will encode this with ordinal ordered encoder

# In[ ]:


from feature_engine.encoding import OrdinalEncoder
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import WoEEncoder, MeanEncoder


# In[ ]:


X.head()


# In[ ]:


X['education'] = df['education']
# set up the encoder
encoder = OrdinalEncoder(encoding_method='ordered', variables=['education'])

# fit the encoder
encoder.fit(X, y)

# transform the data
X = encoder.transform(X)


encoder.encoder_dict_


# One Hot encoding for other categorical features

# In[ ]:


df['poutcome'].unique()


# In[ ]:


df.info()


# In[ ]:


oneHot = OneHotEncoder( variables=['job', 'marital', 'contact','month','day_of_week','poutcome'], drop_last=False)


# In[ ]:


X['job'] = df['job']
X['marital'] = df['marital']
X['contact'] = df['contact']
X['month'] = df['month']
X['day_of_week'] = df['day_of_week']
X['poutcome'] = df['poutcome']


# In[ ]:


X = oneHot.fit_transform(X,y)
oneHot.encoder_dict_


# As default, loan and housing seems to be important features, prefering to use WoE encoding for those. 

# In[ ]:


X['default'] = df['default']
X['loan'] = df['loan']
X['housing'] = df['housing']


# In[ ]:


X['housing'].value_counts()


# In[ ]:


WoE = WoEEncoder(variables=['loan','housing'])
X = WoE.fit_transform(X,y)
WoE.encoder_dict_


# As default have only 3 values for yes category, this is resulting in probability of occurance is 0. So, unable to use WoE. Will use Mean encoding

# In[ ]:


mea = MeanEncoder(variables=['default'])
X = mea.fit_transform(X,y)
mea.encoder_dict_


# In[ ]:


X.head()


# In[ ]:


X.info()


# In[ ]:


df_enc = X.copy()
df_enc['y'] = y
df_enc.to_csv('encodedDS.csv')


# In[ ]:





# Before we transform the dataset, Let's first see the histogram of all these values

# In[ ]:


for each in df_enc.columns:
    print("Histogram of "+each)
    df[each].hist()
    plt.show()


# In[ ]:


for each in df_enc.columns:
    
    print(df[each].dtype)
    if df[each].dtype != np.object:
        print("Box plot of "+each)
        X[each].plot(kind='box')
        plt.show()


# Campaign, duration has more outliers. So, this needs a log transformation. The age has limited outliers, we can go with capping transformation

# In[ ]:


from feature_engine import transformation as vt


# In[ ]:


X['campaign'].unique()


# In[ ]:


#/#
# if Campaigns are already encoded, then log transformation is not possible, we can go with 
# power transformation
#/
logTrans = vt.LogTransformer(variables=['campaign'])
X= logTrans.fit_transform(X,y)

X['campaign'].plot(kind='box')
#plt.show()
#X['duration'].plot(kind='box')
#plt.show()


# It transformed well. As default has negative values, it is not possible to plot using log transformer. We can go with binning

# In[ ]:


X.head()


# In[ ]:


powrTrans = vt.PowerTransformer(variables=['campaign','default'],  exp=0.5)
X = powrTrans.fit_transform(X,y)


# In[ ]:





# Tranformation is also completed.

# In[ ]:


df_trans_en = X.copy()
df_trans_en['y_trans'] = y
df_trans_en.to_csv("transenc.csv")


# In[ ]:





# **Feature Selection**
# 
# /kaggle/input/bankmarketingstage3/data_ootm_renn_train.csv
# /kaggle/input/bankmarketingstage3/data_ootm_renn_test.csv

# In[ ]:


get_ipython().system('pip install pycaret[full]')


# In[ ]:


get_ipython().system('pip install feature_engine')


# In[ ]:


import pandas as pd
import numpy as np
import pycaret
from pycaret.classification import *
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from feature_engine.selection import (
    RecursiveFeatureElimination,
    DropConstantFeatures,
    DropDuplicateFeatures,
)


# In[ ]:


df_ootm_train = pd.read_csv("/kaggle/input/bankmarketingstage3/data_ootm_renn_train.csv")
df_ootm_test = pd.read_csv("/kaggle/input/bankmarketingstage3/data_ootm_renn_test.csv")
#df_woe_train = pd.read_csv("/content/drive/MyDrive/soma-santhosh/stage3/data_WoE_allknn_train.csv")


# In[ ]:


df_ootm = pd.concat([df_ootm_train,df_ootm_test])


# In[ ]:


df_ootm.columns


# In[ ]:


numer_cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']


# In[ ]:


df_ootm.head()


# In[ ]:


ootm_renn_setup = setup(data=df_ootm, target='Subscription', feature_interaction=True,
                        feature_ratio=True, numeric_features=numer_cols, train_size=0.7,imputation_type='iterative',
                        data_split_stratify = False, transformation=False,session_id=5310, data_split_shuffle=False)


# In[ ]:





# In[ ]:


len(df_ootm.columns)


# In[ ]:


ootm_renn_setup[6]


# In[ ]:


df_ootm_renn = ootm_renn_setup[6]


# In[ ]:


dataY = df_ootm['Subscription']


# In[ ]:


data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(df_ootm_renn,dataY,stratify=dataY,test_size=0.3,random_state=42)


# In[ ]:


ootm_grad_sel = RecursiveFeatureElimination(
    variables=None, # automatically evaluate all numerical variables
    estimator = GradientBoostingClassifier(), # the ML model
    scoring = 'roc_auc', # the metric we want to evalute
    threshold = 0.01, # the maximum performance drop allowed to remove a feature
    cv=5, # cross-validation
)


# In[ ]:


ootm_grad_sel_f = ootm_grad_sel.fit(data_X_train, data_y_train)


# In[ ]:


ootm_grad_sel_f.feature_importances_


# In[ ]:


ootm_grad_sel_f.features_to_drop_


# In[ ]:


ootm_grad_sf_sel = SFS(GradientBoostingClassifier(),
          k_features=11, # the lower the features we want, the longer this will take
          forward=False,
          floating=False,
          verbose=2,
          scoring='roc_auc',
          cv=2)


# In[ ]:


ootm_grad_sf_sel_f = ootm_grad_sf_sel.fit(data_X_train, data_y_train)


# In[ ]:


ootm_grad_sb_sel = SFS(GradientBoostingClassifier(),
          k_features=11, # the lower the features we want, the longer this will take
          forward=True,
          floating=False,
          verbose=2,
          scoring='roc_auc',
          cv=2)


# In[ ]:


ootm_grad_sb_sel_f = ootm_grad_sb_sel.fit(data_X_train, data_y_train)


# In[ ]:




