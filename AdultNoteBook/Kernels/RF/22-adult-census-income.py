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
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


census_data=pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
census_data.head()


# In[ ]:


census_data[census_data=='?']=np.nan


# In[ ]:


census_data.head()


# In[ ]:


census_data.isnull().sum()


# In[ ]:


null_columns =['workclass','occupation','native.country']
for i in null_columns:
    census_data.fillna(census_data[i].mode()[0], inplace=True)


# In[ ]:


census_data.head()


# In[ ]:


sns.countplot(y=census_data['workclass'], hue=census_data['income'])


# In[ ]:


sns.countplot(census_data['sex'], hue=census_data['income'])


# In[ ]:


sns.countplot(x=census_data['race'], hue=census_data['income'])


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(y=census_data['education'], hue=census_data['sex'], order = census_data['education'].value_counts().index)


# In[ ]:


sns.countplot(y=census_data['relationship'], hue=census_data['income'], order = census_data['relationship'].value_counts().index)


# In[ ]:


sns.countplot(y=census_data['marital.status'], hue=census_data['income'], order = census_data['marital.status'].value_counts().index)


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(y=census_data['education'], hue=census_data['income'], order = census_data['education'].value_counts().index)


# In[ ]:


sns.pairplot(census_data)


# In[ ]:


sns.heatmap(census_data.corr(), annot=True, cmap="Greens")


# In[ ]:


census_data.head()


# In[ ]:


census_data['marital.status'].unique()


# In[ ]:


census_data['income'] = census_data['income'].replace({'<=50K':0, '>50K':1})
census_data['sex'] = census_data['sex'].replace({'Female':0, 'Male':1})
census_data['race'] = census_data['race'].replace({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Other':3,'Amer-Indian-Eskimo':4})
census_data['workclass'] = census_data['workclass'].replace({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3,
       'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6, 'Never-worked':7})
census_data['native.country'] = census_data['native.country'].replace({'United-States':0, 'Private':1, 'Mexico':2, 'Greece':3, 'Vietnam':4, 'China':5,
       'Taiwan':6, 'India':7, 'Philippines':8, 'Trinadad&Tobago':9, 'Canada':10,
       'South':11, 'Holand-Netherlands':12, 'Puerto-Rico':13, 'Poland':14, 'Iran':15,
       'England':16, 'Germany':17, 'Italy':18, 'Japan':19, 'Hong':20, 'Honduras':21, 'Cuba':22,
       'Ireland':23, 'Cambodia':24, 'Peru':25, 'Nicaragua':26, 'Dominican-Republic':27,
       'Haiti':28, 'El-Salvador':29, 'Hungary':30, 'Columbia':31, 'Guatemala':32,
       'Jamaica':33, 'Ecuador':34, 'France':35, 'Yugoslavia':36, 'Scotland':37,
       'Portugal':38, 'Laos':39, 'Thailand':40, 'Outlying-US(Guam-USVI-etc)':41})
census_data['occupation'] = census_data['occupation'].replace({'Private':0, 'Exec-managerial':1, 'Machine-op-inspct':2,
       'Prof-specialty':3, 'Other-service':4, 'Adm-clerical':5, 'Craft-repair':6,
       'Transport-moving':7, 'Handlers-cleaners':8, 'Sales':9,
       'Farming-fishing':10, 'Tech-support':11, 'Protective-serv':12,
       'Armed-Forces':13, 'Priv-house-serv':14})
census_data['relationship'] = census_data['relationship'].replace({'Not-in-family':0, 'Unmarried':1, 'Own-child':2, 'Other-relative':3,
       'Husband':4, 'Wife':5})
census_data['education'] = census_data['education'].replace({'HS-grad':0, 'Some-college':1, '7th-8th':2, '10th':3, 'Doctorate':4,
       'Prof-school':5, 'Bachelors':6, 'Masters':7, '11th':8, 'Assoc-acdm':9,
       'Assoc-voc':10, '1st-4th':11, '5th-6th':12, '12th':13, '9th':14, 'Preschool':15})
census_data['marital.status'] = census_data['marital.status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Single')
census_data['marital.status'] = census_data['marital.status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
census_data['marital.status'] = census_data['marital.status'].map({'Married':1, 'Single':0})


# In[ ]:


census_data.head()


# In[ ]:


X = census_data.drop(['income'], axis=1)
y = census_data['income']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model_log = LogisticRegression()
model_log.fit(X_train,y_train)


# In[ ]:


pred_log=model_log.predict(X_test)
log_score =model_log.score(X_train,y_train)
log_pred_score =round(log_score*100,2)
log_pred_score


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_rfc = RandomForestClassifier(max_depth=2, random_state=0)
model_rfc.fit(X_train,y_train)


# In[ ]:


pred_rfc=model_rfc.predict(X_test)
rfc_score =model_rfc.score(X_train,y_train)
rfc_pred_score =round(rfc_score*100,2)
rfc_pred_score


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model_decision=DecisionTreeClassifier()
model_decision.fit(X_train,y_train)


# In[ ]:


pred_decision=model_decision.predict(X_test)
decision_score =model_decision.score(X_train,y_train)
decision_pred_score =round(decision_score*100,2)
decision_pred_score


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


print("Accuracy: %s%%" % (100*accuracy_score(y_test, pred_decision)))
print(confusion_matrix(y_test, pred_decision))
print(classification_report(y_test, pred_decision))


# In[ ]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score


# In[ ]:


def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)


# In[ ]:


print_evaluate(y_train,model_decision.predict(X_train))


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


model_predict_test=model_decision.predict_proba(X_test)[:,1]
fpr, tpr, thresholds  = roc_curve(y_test,model_predict_test)
plt.figure(figsize=(15,7))
plt.subplot(1,2,2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('ROC Curve',fontsize=15)

