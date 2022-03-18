#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as  plt
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I have imported an updated version of the dataset with an additional column for risk profile- Good/Bad to make predictions with the dataset whether the transaction is likely to be a fraud or genuine.

# In[ ]:


data= pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")
data


# In[ ]:


data= data.drop(['Unnamed: 0'], axis=1)
data


# We'll first convert the target variable (**Risk**) to numeric form so that we can make some visualizations. We'll be using **LabelBinarize**r function present in the sklearn library for that purpose,

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
lb= LabelBinarizer()
data["Risk"]= lb.fit_transform(data["Risk"])


# In[ ]:


sns.countplot('Risk', data=data)
plt.title('Risk Distribution', fontsize=14)
plt.show()


# In[ ]:





# In[ ]:


ax = sns.scatterplot(x="Duration", y="Age", hue="Risk", data=data)


# From the scatterplot we can see a lot of straight lines. Duration is a continous variable from 0-70 . The lines show that it is practically possible to convert them into categories of Time duration groups. 

# In[ ]:


ax = sns.scatterplot(x="Age", y="Duration", hue="Risk", data=data)


# Similiar observation can be made regarding Age columns. They are organized into groups for various age segments. 
# Next what we will be doing is creating categorical columns for both duration and Ages and plot a histogram to see the frequency distribution plot.

# In[ ]:


ax = sns.scatterplot(x="Credit amount", y="Age", hue="Risk", data=data)


# In[ ]:





# An inference that can be made regarding the Credit amount is that people with lower credit amount have a risk possibiity of 1, i.e. those transactions are likely to be genuine.

# In[ ]:


from scipy.stats import norm

f, (ax1,ax2) =plt.subplots(1,2, figsize=(20, 6))

credit_amount_dist = data['Credit amount'].loc[data['Risk'] == 1].values
sns.distplot(credit_amount_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('Credit amount distribution for good transactions', fontsize=14)

credit_amount_dist = data['Credit amount'].loc[data['Risk'] == 0].values
sns.distplot(credit_amount_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('Credit amount distribution for bad transactions', fontsize=14)


# The distribution is uneven in the sense that there are nearly 700 good transactions and 300 bad transactions
# This is a very likely scenario because from a dataset of large no. of transactions, there are obviously more no of genuine transactions.
# Basically, our dataset is imbalanced as a primitive model which preicts 1 always will also obtain an accuracy of 70% .
# 2 things can be done: 
# 1. Resample (Over-sample/ Under-sample) the dataset and obtain an even distribution of the 2 classes.
# 2. Use the precision/Recall scores to evaluate the model.
# We'll first make visualizations on our dataset, and then use these techniques.

# In[ ]:


from sklearn.preprocessing import StandardScaler
SC= StandardScaler()
credit=data['Credit amount'].values
credit= credit.reshape(-1,1)
data["Credit amount"]= SC.fit_transform(credit)


# In[ ]:





# In[ ]:


Saving_accounts= data["Saving accounts"]
Saving_accounts.isnull().values.sum()


# In[ ]:


Checking_accounts= data["Checking account"]
Checking_accounts.isnull().values.sum()


# Upon analysis, we notice that there are only 2 columns which have missing values, Savings account and Checking account.  It is likely that there were no mistakes in updating the datasets, but the users didnt actually have a savings or a checking account. We'll impute the 'NaN' by 'NoSavingAcc'/ 'NoChecAcc and only then. we will LabelEncode/ OneHotEncode the data. 

# 

# In[ ]:


data["Saving accounts"].fillna('NoSavingAcc', inplace= True)
data["Checking account"].fillna('NoCheckAcc', inplace= True)


# In[ ]:





# In[ ]:


interval = (0, 12, 24, 36, 48, 60, 72, 84)
cats = ['year1', 'year2', 'year3', 'year4', 'year5', 'year6', 'year7']
data["Duration"] = pd.cut(data.Duration, interval, labels=cats)


# We'll convert age column into a categorical column by creating intervals . This will help us know the customer base in a broader sense. We'll create an interval for Students, Youth, Adults and Senior Citizens.

# In[ ]:


interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Youth', 'Adult', 'Senior']
data["Age"] = pd.cut(data.Age, interval, labels=cats)


# In[ ]:


data


# Now, we'll encode the categorial data into 0s and 1s creating seperate columns, and we will aslso take care of the dummy variable trap using drop_first feature. Our data is now encoded and we'll drop the original features

# In[ ]:


data = data.merge(pd.get_dummies(data.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data.Job, drop_first=True, prefix='Job'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data.Duration, drop_first=True, prefix='Duration'), left_index=True, right_index=True)
data = data.merge(pd.get_dummies(data.Age, drop_first=True, prefix='Age'), left_index=True, right_index=True)


# In[ ]:


del data["Checking account"]
del data["Saving accounts"]
del data["Job"]
del data["Duration"]
del data["Sex"]
del data["Purpose"]
del data["Housing"]
del data["Age"]


# In[ ]:


X= data.drop('Risk', axis= 1)
y=data["Risk"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)


# In[ ]:


X_train


# In[ ]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=17, n_iter=7, random_state= 0)
X_train_svd= svd.fit_transform(X_train)


# In[ ]:


explained_variance=svd.explained_variance_ratio_
explained_variance


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(10, 10))
    plt.bar(range(17), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


X_train_svd= pd.DataFrame(X_train_svd)
X_train_svd


# In[ ]:


X_test_svd= svd.transform(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #To evaluate our model
from sklearn.model_selection import GridSearchCV
# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[ ]:


classifier = LogisticRegression()
parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator= classifier,param_grid= parameters, cv=5,  n_jobs= -1)


# In[ ]:


grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)


# In[ ]:


cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))


# In[ ]:


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()
grid_search = GridSearchCV(estimator= svc, param_grid= parameters, cv=5, n_jobs= -1)


# In[ ]:


grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)


# In[ ]:


cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))


# In[ ]:


parameters = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
 'criterion' :['gini', 'entropy']
}
classifier= RandomForestClassifier()
grid_search= GridSearchCV(estimator=classifier, param_grid=parameters, cv= 5, n_jobs= -1)


# In[ ]:


grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)


# In[ ]:


cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))


# In[ ]:


classifier= XGBClassifier()
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)


# In[ ]:


cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))

