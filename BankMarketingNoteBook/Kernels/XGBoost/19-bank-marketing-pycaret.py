#!/usr/bin/env python
# coding: utf-8

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


# # Introduction

# **Problem Statement:** ABC Bank wants to sell it's term deposit product to customers and before launching the product they want to develop a model which help them in understanding whether a particular customer will buy their product or not (based on customer's past interaction with bank or other Financial Institution).

# **Why ML Model:** Bank wants to use ML model to shortlist customer whose chances of buying the product is more so that their marketing channel (tele marketing, SMS/email marketing etc)  can focus only to those customers whose chances of buying the product is more. This will save resource and their time ( which is directly involved in the cost ( resource billing)).

# **Data Set Information :** The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

# **Objective:** Obtain a model that determines whether or not X customer will buy your product, based on past interactions with the bank and other financial institutions. To apply pycaret machine modeling library.

# ### Let's install pycaret

# 

# In[ ]:


get_ipython().system('pip install pycaret[full]')


# In[ ]:


#Importings
import pandas as pd
import numpy as np
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Data loading
data = pd.read_csv("../input/bank-marketing/bank-additional-full.csv", sep=";")


# ## Exploratory Data Analysis

# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


#Target distribution
fig, ax = plt.subplots(figsize = (8, 5))
sns.countplot(x= 'y', data= data, palette='rocket')
plt.xlabel("Target") 
plt.ylabel('Number of Clients')
plt.title("Target distribution")  
plt.show()
#Finding the percentages of our target
per_target= (data['y'] == 'yes').value_counts().to_frame('q')
n_target_q = per_target.q.iloc[0]
y_target_q = per_target.q.iloc[1]
total=per_target.q.sum() 
print('The amount of "no" target is =  {:,}'.format(n_target_q))
print('The amount of "yes" target is = {:,}'.format(y_target_q))
print(f'The percentage of "no" target is = {round((n_target_q/total)*100,2)}','%')
print(f'The percentage of "yes" target is = {round((y_target_q/total)*100,2)}','%')


# In[ ]:


categorical_cols = data.select_dtypes(include="object").columns.to_list()
categorical_cols.remove("y")
categorical_cols


# In[ ]:


numerical_cols = data.select_dtypes(include=[np.number]).columns.to_list()
numerical_cols


# In[ ]:


for cat in categorical_cols:
    print(f"Column: {cat} unique values")
    print(data[cat].unique())


# In[ ]:


plt.figure(figsize = (20,20))
plt.subplot(5, 2, 1)
sns.boxplot(data = data, x= 'age', palette = 'colorblind')
plt.title('Age boxplot')
plt.xlabel('Age')

plt.subplot(5, 2, 2)
sns.boxplot(data = data, x= 'duration', palette = 'colorblind')
plt.title('Duration boxplot')
plt.xlabel('Duration')

plt.subplot(5, 2, 3)
sns.boxplot(data = data, x= 'campaign', palette = 'colorblind')
plt.title('Campaign boxplot')
plt.xlabel('Campaign')

plt.subplot(5, 2, 4)
sns.boxplot(data = data, x= 'pdays', palette = 'colorblind')
plt.title('pdays boxplot')
plt.xlabel('pdays')

plt.subplot(5, 2, 5)
sns.boxplot(data = data, x= 'previous', palette = 'colorblind')
plt.title('Previous boxplot')
plt.xlabel('Previous')

plt.subplot(5, 2, 6)
sns.boxplot(data = data, x= 'emp.var.rate', palette = 'colorblind')
plt.title('emp.var.rate boxplot')
plt.xlabel('emp.var.rate')

plt.subplot(5, 2, 7)
sns.boxplot(data = data, x= 'cons.price.idx', palette = 'colorblind')
plt.title('cons.price.idx boxplot')
plt.xlabel('cons.price.idx')

plt.subplot(5, 2, 8)
sns.boxplot(data = data, x= 'cons.conf.idx', palette = 'colorblind')
plt.title('cons.conf.idx boxplot')
plt.xlabel('cons.conf.idx')

plt.subplot(5, 2, 9)
sns.boxplot(data = data, x= 'euribor3m', palette = 'colorblind')
plt.title('euribor3m boxplot')
plt.xlabel('euribor3m')

plt.subplot(5, 2, 10)
sns.boxplot(data = data, x= 'nr.employed', palette = 'colorblind')
plt.title('nr.employed boxplot')
plt.xlabel('nr.employed')

plt.tight_layout()
plt.show()


# In[ ]:


# Outlier filter

def remove_outliers(dfx):
    q1 = dfx.quantile(0.25)
    q3 = dfx.quantile(0.75)
    iqr = q3 - q1
    cut_off = iqr*1.5
    
    df_filtred = dfx[~((dfx < (dfx.quantile(0.25) - cut_off)) | (dfx > (dfx.quantile(0.75) + cut_off))).any(axis=1)]
    
    
    return df_filtred


# In[ ]:


data2 = remove_outliers(data)


# In[ ]:


data2.reset_index(drop=True, inplace=True)
data2.info()


# ### Categorical features analysis

# In[ ]:


data2.job.value_counts()


# In[ ]:


sns.countplot(data=data2, y='job',hue ='y', edgecolor ='black')
print('##############################################')
print('Unkown job clients = ', data2[(data2['job'] =='unknown')].shape[0])
print('##############################################')
print("Percentage of unknown client's job =" ,round(data2[(data2['job'] =='unknown')].shape[0]/data2.shape[0]*100,2),'%')


# In[ ]:


data2.marital.value_counts()


# In[ ]:


sns.countplot(data=data2, x='marital',hue ='y', edgecolor ='black')
print('##############################################')
print('Unkown job clients = ', data2[(data2['marital'] =='unknown')].shape[0])
print('##############################################')
print("Percentage of unknown client's marital =" ,round(data2[(data2['marital'] =='unknown')].shape[0]/data2.shape[0]*100,2),'%')


# In[ ]:


data2.education.value_counts()


# In[ ]:


sns.countplot(data=data2, y='education',hue ='y', edgecolor ='black')
print('##############################################')
print('Unkown job clients = ', data2[(data2['education'] =='unknown')].shape[0])
print('##############################################')
print("Percentage of unknown client's education level =" ,round(data2[(data2['education'] =='unknown')].shape[0]/data2.shape[0]*100,2),'%')


# In[ ]:


data2.default.value_counts()


# In[ ]:


sns.countplot(data=data2, y='default',hue ='y', edgecolor ='black')
print('##############################################')
print('Unkown job clients = ',data2[(data2['default'] =='unknown')].shape[0])
print('##############################################')
print("Percentage of unknown client's default =" ,round(data2[(data2['default'] =='unknown')].shape[0]/data2.shape[0]*100,2),'%')


# In[ ]:


data2.housing.value_counts()


# In[ ]:


sns.countplot(data=data2, y='housing',hue ='y', edgecolor ='black')
print('##############################################')
print('Unkown job clients = ', data2[(data2['housing'] =='unknown')].shape[0])
print('##############################################')
print("Percentage of unknown client's default =" ,round(data2[(data2['housing'] =='unknown')].shape[0]/data2.shape[0]*100,2),'%')


# In[ ]:


data2.loan.value_counts()


# In[ ]:


sns.countplot(data=data2, y='housing',hue ='y', edgecolor ='black')
print('##############################################')
print('Unkown job clients = ', data2[(data2['housing'] =='unknown')].shape[0])
print('##############################################')
print("Percentage of unknown client's default =" ,round(data2[(data2['housing'] =='unknown')].shape[0]/data2.shape[0]*100,2),'%')


# In[ ]:


data2.replace('unknown', np.nan, regex=True,inplace=True)
data2.head()


# In[ ]:


data2.isna().sum().plot.bar()
plt.ylabel("NAs")


# In[ ]:


eliminacion = data2.copy()
eliminacion = eliminacion.dropna()
print(data2.shape[0]-eliminacion.shape[0], 'rows were eliminated')
print('Equivalent to',round(((data2.shape[0]-eliminacion.shape[0])/data2.shape[0])*100,2),' % of the original dataset')
print( 'Current dataset size = ',eliminacion.shape[0])


# In[ ]:


data3 = eliminacion.copy()
data3 = data3.drop("default", axis=1)


# In[ ]:


#Numerical features

numerical = data3.select_dtypes(include=[np.number])
numerical_cols = numerical.columns.to_list()
numerical.describe()


# In[ ]:


numerical.hist(bins = 15, figsize = (10,10), xlabelsize = 0.1, ylabelsize = 0.1)
plt.show()


# In[ ]:


data4 = data3.drop(["pdays", "previous"], axis=1)


# In[ ]:


final_ds = data4.copy()

print(final_ds.shape)
final_ds.head()


# In[ ]:


categorical_cols = final_ds.select_dtypes(include="object").columns.to_list()
categorical_cols.remove("y")
numerical_cols = final_ds.select_dtypes(include=[np.number]).columns.to_list()


# In[ ]:


categorical_cols


# # Modeling

# In[ ]:


clfs = setup(data = final_ds, target = 'y',
             session_id=234, normalize=True, transformation = True,
             numeric_features=numerical_cols, categorical_features=categorical_cols, 
             remove_multicollinearity = True, multicollinearity_threshold = 0.95, 
             fix_imbalance=True, data_split_stratify=True,
             silent=True
             )


# In[ ]:


top3 = compare_models(n_select=3, sort ="AUC")


# In[ ]:


tuned_models = [tune_model(i, n_iter=100, search_library="optuna", optimize="AUC", verbose=False) for i in top3]


# In[ ]:


catboost = create_model("catboost", verbose=False)
xgb = create_model("xgboost", verbose=False)
lgbm = create_model("lightgbm", verbose=False)


# ### 1. Catboost Classifier

# In[ ]:


tuned_ctb = tune_model(catboost, optimize='auc', n_iter=20)


# In[ ]:


plot_model(tuned_ctb, plot='class_report' )


# In[ ]:


plot_model(tuned_ctb, plot='confusion_matrix')


# ### 2. Extreme Gradient Boosting

# In[ ]:


tuned_xgb = tune_model(xgb, n_iter=20, optimize='auc')


# In[ ]:


plot_model(tuned_xgb, plot='class_report' )


# In[ ]:


plot_model(tuned_xgb , plot='confusion_matrix')


# ### 3. Light Gradient Boosting Machine

# In[ ]:


tuned_lgbm = tune_model(lgbm, n_iter=20, optimize='auc')


# In[ ]:


plot_model(tuned_lgbm, plot='class_report' )


# In[ ]:


plot_model(tuned_lgbm , plot='confusion_matrix')


# ### Blending

# In[ ]:


blender = blend_models(top3, method = 'soft', optimize="auc")


# In[ ]:


plot_model(blender)


# In[ ]:


plot_model(blender, plot="confusion_matrix")


# In[ ]:


plot_model(blender, plot="class_report")


# ### Best model

# In[ ]:


best_model = automl(optimize='auc')


# In[ ]:


best_model


# In[ ]:


plot_model(best_model, plot="calibration")


# In[ ]:


best_model_cal = calibrate_model(best_model)


# In[ ]:


plot_model(best_model_cal, plot="calibration")

