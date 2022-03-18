#!/usr/bin/env python
# coding: utf-8

# In this notebook, I have tried to use the three boosting algorithms - Adaboost, gradient boosting and xgboost - to solve the classification problem in the Adult Income dataset. This is my first notebook while learning boosting algorithms and I would appreciate any kind of suggestions or improvements.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


file = ('../input/adult-census-income/adult.csv')
df = pd.read_csv(file)


# In[ ]:


print(df.head())


# In[ ]:


print(df.info())


# In[ ]:


#Count the occuring of the '?' in all the columns
for i in df.columns:
    t = df[i].value_counts()
    index = list(t.index)
    print ("Count of ? in", i)
    for i in index:
        temp = 0
        if i == '?':
            print (t['?'])
            temp = 1
            break
    if temp == 0:
        print ("0")


# In[ ]:


df=df.loc[(df['workclass'] != '?') & (df['native.country'] != '?')]
print(df.head())


# In[ ]:


df["income"] = [1 if i=='>50K' else 0 for i in df["income"]]
print(df.head())


# In[ ]:


df_more=df.loc[df['income'] == 1]
print(df_more.head())


# In[ ]:


workclass_types = df_more['workclass'].value_counts()
labels = list(workclass_types.index) 
aggregate = list(workclass_types)
print(workclass_types)
print(aggregate)
print(labels)


# In[ ]:


plt.figure(figsize=(7,7))
plt.pie(aggregate, labels=labels, autopct='%1.1f%%', shadow = True)
plt.axis('equal')
plt.show()


# In[ ]:


#Count plot on single categorical variable 
sns.countplot(x ='income', data = df)
plt.show() 
df['income'].value_counts()


# In[ ]:


#Plot figsize
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
print(plt.show())


# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df['age'], color="red", bins=100)
plt.ylabel("Distribution", fontsize = 10)
plt.xlabel("Age", fontsize = 10)
plt.show()


# In[ ]:


#To find distribution of categorical columns w.r.t income
fig, axes = plt.subplots(figsize=(20, 10))

plt.subplot(231)
sns.countplot(x ='workclass', 
              hue='income', 
              data = df,
              palette="BuPu") 
plt.xticks(rotation=90)

plt.subplot(232)
sns.countplot(x ='marital.status', 
              hue='income', 
              data = df,
              palette="deep") 
plt.xticks(rotation=90)

plt.subplot(233)
sns.countplot(x ='education', 
              hue='income', 
              data = df,
              palette = "autumn") 
plt.xticks(rotation=90)

plt.subplot(234)
sns.countplot(x ='relationship', 
              hue='income', 
              data = df,
              palette = "inferno") 
plt.xticks(rotation=90)

plt.subplot(235)
sns.countplot(x ='sex', 
              hue='income', 
              data = df,
              palette = "coolwarm") 
plt.xticks(rotation=90)

plt.subplot(236)
sns.countplot(x ='race', 
              hue='income', 
              data = df,
              palette = "cool") 
plt.xticks(rotation=90)
plt.subplots_adjust(hspace=1) 
plt.show()


# In[ ]:


df1 = df.copy()


# In[ ]:


categorical_features = list(df1.select_dtypes(include=['object']).columns)
print(categorical_features)
df1


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feat in categorical_features:
    df1[feat] = le.fit_transform(df1[feat].astype(str))
df1


# In[ ]:


X = df1.drop(columns = ['income'])
y = df1['income'].values

# Splitting the data set into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 0)

print ("Train set size: ", X_train.shape)
print ("Test set size: ", X_test.shape)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

# Train Adaboost Classifer
abc = AdaBoostClassifier(n_estimators = 300, learning_rate=1)
abc_model = abc.fit(X_train, y_train)

#Prediction
y_pred_abc = abc_model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred_abc))
print("F1 score :",f1_score(y_test, y_pred_abc, average='binary'))
print("Precision : ", precision_score(y_test, y_pred_abc))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_abc)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = "coolwarm");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix - score:' + str(round(accuracy_score(y_test, y_pred_abc), 2)), size = 15);
plt.show()
print(classification_report(y_test, y_pred_abc))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier 

#Training the model with gradient boosting
gbc = GradientBoostingClassifier(
    learning_rate = 0.1, 
    n_estimators = 500,
    max_depth = 5, 
    subsample = 0.9,
    min_samples_split = 100,
    max_features='sqrt', 
    random_state=10)
gbc.fit(X_train,y_train)

# Predictions
y_pred_gbc = gbc.predict(X_test)

print("Accuracy : ",accuracy_score(y_test, y_pred_gbc))
print("F1 score : ", f1_score(y_test, y_pred_gbc, average = 'binary'))
print("Precision : ", precision_score(y_test, y_pred_gbc))


# In[ ]:


rms = np.sqrt(mean_squared_error(y_test, y_pred_gbc))
print("RMSE for gradient boost: ", rms)


# In[ ]:


cm = confusion_matrix(y_test, y_pred_gbc)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot = True, fmt=".3f", linewidths = 0.5, square = True, cmap = "coolwarm");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix - score:' + str(round(accuracy_score(y_test, y_pred_gbc),2)), size = 15);
plt.show()
print(classification_report(y_test, y_pred_gbc))


# In[ ]:


import xgboost as xgb
from xgboost import XGBClassifier

#Training the model with gradient boosting
xgboost = XGBClassifier(learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      n_estimators=1000, 
                      max_depth=20, 
                      gamma=1)
                      
xgboost_model = xgboost.fit(X_train, y_train)

# Predictions
y_pred_xgboost = xgboost_model.predict(X_test)

print("Accuracy : ",accuracy_score(y_test, y_pred_xgboost))
print("F1 score : ", f1_score(y_test, y_pred_xgboost, average = 'binary'))
print("Precision : ", precision_score(y_test, y_pred_xgboost))


# In[ ]:


rms = np.sqrt(mean_squared_error(y_test, y_pred_xgboost))
print("RMSE for xgboost: ", rms)


# In[ ]:


cm = confusion_matrix(y_test, y_pred_xgboost)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = "coolwarm");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix - score:'+str(round(accuracy_score(y_test, y_pred_xgboost),2)), size = 15);
plt.show()
print(classification_report(y_test,y_pred_xgboost))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, xgboost.predict_proba(X_test)[:,1])  
plt.figure(figsize = (10,5))
plt.plot([0,1],[0,1], 'k--')                                                 
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE for Xgboost')
plt.show()


# 
# I would like to conclude this notebook by mentioning that here, I have tuned the hyperparameters myself instead of using Grid Search or random search as it didn't seem to increase my accuracies. I would appreciate any suggestions or improvements that I can make to make this better.
