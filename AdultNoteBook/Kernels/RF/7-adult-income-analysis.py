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
import warnings 
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **IMPORT DATASET**

# In[ ]:


df=pd.read_csv("/kaggle/input/adult-census-income/adult.csv")
df.head()


# As we see from above that we have "?" in the dataset. So we need to replace this. So we replaced it with nan.

# In[ ]:


df[df=='?']=np.nan


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# **Now filling the missing values with the mode of the respective columns**

# In[ ]:


null_columns =['workclass','occupation','native.country']
for i in null_columns:
    df.fillna(df[i].mode()[0], inplace=True)


# In[ ]:


df.head()


# **DATA VISUALISATION**

# In[ ]:


corr=df.corr()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# In[ ]:


sns.countplot(y=df['workclass'], hue=df['income'])


# From here we see that most of the people having salary less than 50k is from the private sector.

# In[ ]:


sns.countplot(df['sex'], hue=df['income'])


# We see that most of the people working are having salary less than 50k.

# In[ ]:


sns.countplot(x=df['race'], hue=df['income'])


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(y=df['education'], hue=df['sex'], order = df['education'].value_counts().index)


# In[ ]:


sns.countplot(y=df['relationship'], hue=df['income'], order = df['relationship'].value_counts().index)


# In[ ]:


sns.countplot(y=df['marital.status'], hue=df['income'], order = df['marital.status'].value_counts().index)


# We see that most of  the married people are earning more than 50k as compared to other classes

# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(y=df['education'], hue=df['income'], order = df['education'].value_counts().index)


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.head()


# **CHECKING UNIQUE VALUES**
# 
# If we have many unique values then we will try some other encoding technique otherwise we will use label encoder

# In[ ]:


df["workclass"].unique()


# In[ ]:


df["education"].unique()


# In[ ]:


df["marital.status"].unique()


# In[ ]:


df["occupation"].unique()


# In[ ]:


df["relationship"].unique()


# In[ ]:


df["race"].unique()


# In[ ]:


df["sex"].unique()


# In[ ]:


df["native.country"].unique()


# As we have many unique values of native.country column so we will store its count in a dictonary and then map it on the dataset

# In[ ]:


native=df["native.country"].value_counts().to_dict()


# In[ ]:


df["native.country"]=df["native.country"].map(native)


# In[ ]:


df.head()


# **ENCODING CATEGORICAL VALUES**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


df_cols=("workclass","education","marital.status","occupation","relationship","race","sex")
for i in df_cols:
    df[i]=le.fit_transform(df[i])


# In[ ]:


df.head()


# **SPLITTING DATASET INTO DEPENDENT AND INDEPENDENT**

# In[ ]:


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# **SPLITTING DATASET INTO TRAINING AND TEST SET**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# **TRAINING THE DATASET ON LOGISTIC REGRESSION MODEL**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# **PREDICTING THE RESULT**

# In[ ]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# **MAKING CONFUSION MATRIX AND CHECKING THE ACCURACY**

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# As we got a moderate accuracy so will try any other model to see if accuracy can increase

# **TRAINING THE DATASET ON RANDOM FOREST CLASSIFIER**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_2.fit(X_train, y_train)


# **PREDICTING THE VALUES**

# In[ ]:


y_pred_2 = classifier_2.predict(X_test)
print(np.concatenate((y_pred_2.reshape(len(y_pred_2),1), y_test.reshape(len(y_test),1)),1))


# **MAKING THE CONFUSION MATRIX AND CHECKING THE ACCURACY**

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_2)
print(cm)
accuracy_score(y_test, y_pred_2)


# **WE GOT AN ACCURACY HIGHER THAN LOGISTIC REGRESSION.**
