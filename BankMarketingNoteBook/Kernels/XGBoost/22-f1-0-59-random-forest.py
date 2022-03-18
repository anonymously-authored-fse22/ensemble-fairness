#!/usr/bin/env python
# coding: utf-8

# **Classification problem**\
# **Step 1:** Check missing and dublicate values\
#     No missing values and 12 rows are dublicate values are removed\
# **Step 2:** Convert the categorical features into numerical \
# **Step 3:** Conduct correlation study to study the correlation between the features\
# **Step 4:** Data set is seperated for training and testing\
# **Step 5:** XGBClassifier and randomforest algorithms are used build the classification model\
# **Step 6:** F1 score and accuray is measured in all models\

# # Importing libraries 

# In[ ]:


# Add required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Label encoder order is alphabetical
from sklearn.preprocessing import LabelEncoder
# Ordinal encoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

#from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score,f1_score,precision_score,recall_score

import optuna
seed =42 # for repeatability
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# # Getting Data

# In[ ]:


# Load data set
# Add path for the data set
path = '/kaggle/input/bank-marketing/bank-additional-full.csv'
df = pd.read_csv(path, delimiter=';') # Use delimiter to split the csv file
Data = df;


# # Understanding the data

# In[ ]:


df.head()


# In[ ]:


df.shape


# # Data cleaning

# In[ ]:


df.isnull().sum() # Check the missing elements


# # Data analysis

# In[ ]:


df.info()


# Numerical features (Integers and floats): 10\
# Age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m,nr.employed\
# Categorical features (Object): 10\
# Job, Marital, education, default, housing, loan, contact, month, day_of_week, poutcome

# In[ ]:


# checking for duplicate rows
print(df.shape[0])
print(f'Number of duplicated rows: {df.shape[0] - df.drop_duplicates().shape[0]}')
print('dropping duplicates')
df = df.drop_duplicates()


# In[ ]:


Y = df['y']
df = df.drop(['y'],axis=1)


# In[ ]:


df.nunique() # Count Distinct Values


# In[ ]:


df.describe()


# # Visualizing numerical features

# In[ ]:


fig, axs = plt.subplots(nrows = 10, ncols = 2, figsize = (15, 35))

def dist_and_box(var,Loc):
    # Box plot properties
    PROPS = {
        'boxprops':{'facecolor':'paleturquoise', 'edgecolor':'black'},
        'medianprops':{'color':'black'},
        'whiskerprops':{'color':'black'},
        'capprops':{'color':'black'}
    }
    sns.distplot(df[var], hist = True, color = "darkturquoise",kde_kws={"color": "k"}, hist_kws = {'edgecolor':'blue'},ax = axs[Loc,0])
    sns.boxplot(x= var, data = df, ax = axs[Loc,1],**PROPS)

Numer_feature = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

# Plot disttribution and boxplot of all numerical features
I = 0
for i in Numer_feature:
    dist_and_box(i,I)
    I = I+1


# In[ ]:


df = df.drop(['pdays','previous'],axis=1)


# # Visvalizing categorical features

# In[ ]:



fig, axs = plt.subplots(nrows = 4, ncols = 1, figsize = (15, 10))

sns.countplot(x = "job", data = df, ax=axs[0])
sns.countplot(x = "month", data = df, ax=axs[1])
sns.countplot(x = "education", data = df, ax=axs[2])
sns.countplot(x = "day_of_week", data = df, ax=axs[3])


# In[ ]:


# Marital, education, default, housing, loan, contact, poutcome

fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 15))

# Material  and poutcome
sns.countplot(x = "marital", data = df, ax=axs[0,0])
sns.countplot(x = "poutcome", data = df, ax=axs[0,1])
# Default and housing
sns.countplot(x = "default", data = df, ax=axs[1,0])
sns.countplot(x = "housing", data = df, ax=axs[1,1])
# loan and contact
sns.countplot(x = "loan", data = df, ax=axs[2,0])
sns.countplot(x = "contact", data = df, ax=axs[2,1])


# # Categorical features treatment

# Assigns label to all categorical features

# In[ ]:


# Encode the data as per label (Alphabetical)
L_enc = LabelEncoder()
df['job']      = L_enc.fit_transform(df['job']) 
df['marital']  = L_enc.fit_transform(df['marital']) 
df['education']= L_enc.fit_transform(df['education']) 
df['default']  = L_enc.fit_transform(df['default']) 
df['housing']  = L_enc.fit_transform(df['housing']) 
df['contact']  = L_enc.fit_transform(df['contact'])
df['month']     = L_enc.fit_transform(df['month'])
df['day_of_week'] = L_enc.fit_transform(df['day_of_week'])
df['poutcome'] = L_enc.fit_transform(df['poutcome'])
df['loan'] = L_enc.fit_transform(df['loan'])


# In[ ]:


# One hot encoding
#from sklearn.preprocessing import OneHotEncoder


# creating instance of one-hot-encoder
#enc = OneHotEncoder(handle_unknown='ignore')

#enc_df = pd.DataFrame(enc.fit_transform(df[['month']]).toarray())

# merge with main df bridge_df on key values
#df = df.join(enc_df)
#df


# In[ ]:


df.head()


# # Feature selection - Correlation study

# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot = True,cmap="YlGnBu",fmt='.1g')


# In[ ]:


# Remove pdays
df = df.drop(['emp.var.rate','euribor3m'],axis=1)


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot = True,cmap="YlGnBu",fmt='.1g')


# In[ ]:


YY = L_enc.fit_transform(Y)


# In[ ]:


# Generate categorical plots for features
for col in ["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]:
    sns.catplot(x=col, y=YY, data=df, kind='point', aspect=2, )
    plt.ylim(0, 0.7)


# In[ ]:


df = df.drop(['housing','loan'],axis=1)


# In[ ]:


df.shape


# # Spliting data set for training and testing

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(df,
                                                    YY,
                                                    test_size=.20, random_state = 42,
                                                    stratify= YY)


# In[ ]:


# Scaling the model
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.fit_transform(X_test)


# # Model Building

# In[ ]:


# XGBoost - classification
model_xgb_clf = xgb.XGBClassifier()
model_xgb_clf.fit(X_train, y_train)
xgb_preds_clf = model_xgb_clf.predict(X_test)
print('acc: ',accuracy_score(y_test,xgb_preds_clf))
print('F1: ', f1_score(y_test,xgb_preds_clf))
print('Precision: ', precision_score(y_test,xgb_preds_clf))
print('Recall: ', recall_score(y_test,xgb_preds_clf))


# In[ ]:


# Random forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=25, random_state=0)
clf.fit(X_train, y_train)
RF_preds_clf = clf.predict(X_test)

print('acc: ',accuracy_score(y_test,RF_preds_clf))
print('F1: ', f1_score(y_test,RF_preds_clf))
print('Precision: ', precision_score(y_test,RF_preds_clf))
print('Recall: ', recall_score(y_test,RF_preds_clf))


# In[ ]:


# 

