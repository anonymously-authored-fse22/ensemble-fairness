#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import sklearn
from sklearn.model_selection import train_test_split


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

file = ('../input/adult-census-income/adult.csv')
df = pd.read_csv(file, encoding='latin-1')


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


# df.workclass.value_counts()['?']# ,df.occupation.value_counts()['?'],  df['native.country'].value_counts()['?']


# In[ ]:


df = df[df.occupation !='?']


# In[ ]:


plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[ ]:


df.info()


# In[ ]:


# Converting Yes to 1 and No to 0
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})


# In[ ]:


df


# In[ ]:


df['native.country'].value_counts()


# In[ ]:



# Encode the age group of passengers based on above tableau
df['native.country'][df['native.country']!='United-States'] = "Non-United-States"


# In[ ]:





# In[ ]:


import seaborn as sns
fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(25,20))
plt.xticks(rotation=45)
sns.countplot(df['workclass'],hue=df['income'],ax=f)
sns.countplot(df['relationship'],hue=df['income'],ax=b)
sns.countplot(df['marital.status'],hue=df['income'],ax=c)
sns.countplot(df['race'],hue=df['income'],ax=d)
sns.countplot(df['sex'],hue=df['income'],ax=e)
sns.countplot(df['native.country'],hue=df['income'],ax=a)


# In[ ]:


fig, (a,b)= plt.subplots(1,2,figsize=(20,6))
sns.boxplot(y='hours.per.week',x='income',data=df,ax=a)
sns.boxplot(y='age',x='income',data=df,ax=b)


# In[ ]:


df.info()


# In[ ]:


df_backup =df


# In[ ]:


df.workclass.value_counts()


# In[ ]:


# To convert the object type to Int via LABELLED Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



for i in df.columns:
    df[i]= le.fit_transform(df[i])


# In[ ]:


df


# In[ ]:



random.seed(100)
train,test = train_test_split(df,test_size=0.2)


# In[ ]:


train.size,test.shape


# In[ ]:


l=pd.DataFrame(test['income'])
l['baseline'] =0
k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],l['baseline']))
print(k)
(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=500)
X = df.drop('income',axis=1)
y = df['income']
clf.fit(X,y)


# In[ ]:


clf.score(X,y)


# In[ ]:


pred = clf.predict(test.drop('income',axis=1))


# In[ ]:


import sklearn
k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred))
print(k)


# In[ ]:


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[ ]:


y_score = clf.fit(X,y).decision_function(test.drop('income',axis=1))

fpr,tpr,the=sklearn.metrics.roc_curve(test['income'],y_score)
sklearn.metrics.roc_auc_score(test['income'],pred)
plt.plot(fpr,tpr,)


# In[ ]:


sklearn.metrics.roc_auc_score(test['income'],y_score)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
X_std = MinMaxScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=14)
Y_sklearn = sklearn_pca.fit_transform(X_std)

cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

sklearn_pca.explained_variance_ratio_[:10].sum()

cum_sum = cum_sum*100

fig, ax = plt.subplots(figsize=(8,8))
plt.bar(range(14), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier(max_features=14,min_samples_leaf=100,random_state=10)
clf.fit(X,y)


# In[ ]:


pred2 = clf.predict(test.drop('income',axis=1))


# In[ ]:


k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))
print(k)


# In[ ]:


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[ ]:


from xgboost import XGBClassifier

clf= XGBClassifier()

clf.fit(X,y)

pred2 = clf.predict(test.drop('income',axis=1))

k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[ ]:


from catboost import CatBoostClassifier

clf= CatBoostClassifier(learning_rate=0.3)

clf.fit(X,y)

pred2 = clf.predict(test.drop('income',axis=1))

k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))



# In[ ]:


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[ ]:


clf.score(X,y)


# In[ ]:




