#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import io
import requests
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.1f}'.format


# # IMPORT DATA

# In[ ]:


df_adult = pd.read_csv('/kaggle/input/adult-census-income/adult.csv',na_values='?')
df_adult.head()


# # DATA EXPLORATION

# ## Numerical Data

# In[ ]:


data_numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
display(df_adult.select_dtypes(include=data_numerics).columns)
print(df_adult.select_dtypes(include=data_numerics).shape)
data_numerics = df_adult.select_dtypes(include=data_numerics)
data_numerics.describe()


# In[ ]:


features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss','hours.per.week']
plt.figure(figsize=(15, 7))
for i in range(0, len(features)):
    plt.subplot(1, 6, i+1)
    sns.boxplot(df_adult[features[i]],color='#982642')
    plt.tight_layout()
    
plt.show


# In[ ]:


data_num = df_adult[features]
k = len(data_num.columns)
n = 3
m = (k - 1) // n + 1
fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
for i, (name, col) in enumerate(data_num.iteritems()):
    r, c = i // n, i % n
    ax = axes[r, c]
    col.hist(ax=ax, color='#73bbff')
    ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name, color='red')
    ax2.set_ylim(0)

fig.tight_layout()
plt.show


# ## Non-Numerical Data

# In[ ]:


display(df_adult.select_dtypes(include=['object']).columns)
print(df_adult.select_dtypes(include=object).shape)
data_cat = df_adult.select_dtypes(include=['object'])
data_cat.describe()


# ## Check Missing Value

# In[ ]:


data_missing_value = df_adult.isnull().sum().reset_index()
data_missing_value.columns = ['feature','missing_value']
data_missing_value['percentage'] = round((data_missing_value['missing_value']/len(df_adult))*100,2)
data_missing_value = data_missing_value.sort_values('percentage', ascending=False).reset_index(drop=True)
data_missing_value = data_missing_value[data_missing_value['percentage']>0]
print(data_missing_value)

x = data_missing_value['feature']
y = data_missing_value['percentage']
plt.figure(figsize=(10,8))
barh = plt.bar(x=x, height=y, data=data_missing_value, 
       color = '#842e2e', 
       edgecolor= '#2e2e2e',
       linewidth = 2) 

plt.title('Missing Value', fontsize = 16)
plt.ylabel('Persentage', fontsize=14)
plt.xlabel('Feature', fontsize=14)

x_numbers = range(len(x))
for i in x_numbers:
    plt.text(x = x_numbers[i]-0.12,y = y[i]+0.08,s = str(round(y[i],2))+'%',size = 15)
plt.tight_layout
plt.show


# In[ ]:


df_adult = df_adult.dropna()
df_adult = df_adult.drop_duplicates()
df_adult=df_adult.drop('fnlwgt', 1)
df_adult=df_adult.drop('education.num', 1)
df_adult.head()


# ## Visualization of Ocupation vs Income

# In[ ]:


fig = plt.figure(figsize = (10,6))
ax=sns.countplot(df_adult['occupation'], hue=df_adult['income'])
ax.set_title('Occupation vs Income')
plt.xlabel("Occupation",fontsize = 10);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Work Class vs Income

# In[ ]:


fig = plt.figure(figsize = (10,6))
ax=sns.countplot(df_adult['workclass'], hue=df_adult['income'])
ax.set_title('workclass vs Income')
plt.xlabel("workclass",fontsize = 10);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Education vs Income

# In[ ]:


fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['education'], hue=df_adult['income'])
ax.set_title('Education vs Income')
plt.xlabel("Education",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Ocupation vs Income

# I simplified the value of **marital.status** to **Married** and **Not Married**

# In[ ]:


df_adult['marital.status'] = df_adult['marital.status'].map({'Widowed': 'Not married', 'Married-spouse-absent': 'Married', 'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Divorced': 'Not married','Separated': 'Not married', 'Never-married': 'Married'}).astype(str)
fig = plt.figure(figsize = (10,6))
ax=sns.countplot(df_adult['marital.status'], hue=df_adult['income'])
ax.set_title('Marital Status vs Income')
plt.xlabel("Marital Status",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Race vs Income

# In[ ]:


fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['race'], hue=df_adult['income'])
ax.set_title('Race vs Income')
plt.xlabel("Race",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Sex vs Income

# In[ ]:


fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['sex'], hue=df_adult['income'])
ax.set_title('Sex vs Income')
plt.xlabel("Sex",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Relationship vs Income

# In[ ]:


fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['relationship'], hue=df_adult['income'])
ax.set_title('relationship vs Income')
plt.xlabel("relationship",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Age Distribustion

# In[ ]:


plt.figure(figsize=(15,8))
ax=sns.distplot(df_adult['age'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
ax.set_title('Age Distribution')
ax.set_xlabel('Age',fontsize = 14);


# ## Visualization of Age vs Income

# In[ ]:


df_adult['age'] = pd.cut(df_adult['age'], bins = [0, 49, 65, 100], labels = ['Productive-age', 'Above-productive', 'Aged'])

fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['age'], hue=df_adult['income'])
ax.set_title('Age vs Income')
plt.xlabel("Age",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Hours/Week Distribustion

# In[ ]:


plt.figure(figsize=(15,8))
ax=sns.distplot(df_adult['hours.per.week'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
ax.set_title('Hours/Week Distribution')
ax.set_xlabel('Hours Per Week', fontsize = 14);


# In[ ]:


df_adult['hours.per.week'] = pd.cut(df_adult['hours.per.week'], 
                                   bins = [0, 30, 40, 100], 
                                   labels = ['Below Normal Hours', 'Normal Hours', 'Above Normal Hours'])
fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['hours.per.week'], hue=df_adult['income'])
ax.set_title('Hours/Week vs Income')
plt.xlabel("Hours/Week",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Native Country vs Income

# I simplified the value of **native.country** to **United-States** and **Non-USA**

# In[ ]:


df_adult.loc[df_adult['native.country']!='United-States','native.country'] = 'Non-USA'

fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['native.country'], hue=df_adult['income'])
ax.set_title('Native Country vs Income')
plt.xlabel("Native Country",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Capital Gain vs Income

# I define a value of 0 is not getting capital gain **(No)** and besides that is getting capital gain **(Yes)**

# In[ ]:


df_adult.loc[df_adult['capital.gain']!=0,'capital.gain'] = 'Yes'
df_adult.loc[df_adult['capital.gain']!='Yes','capital.gain'] = 'No'

fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['capital.gain'], hue=df_adult['income'])
ax.set_title('Capital Gain vs Income')
plt.xlabel("Capital Gain",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# ## Visualization of Capital Loss vs Income

# I define a value of 0 is not getting capital loss **(No)** and besides that is getting capital loss **(Yes)**

# In[ ]:


df_adult.loc[df_adult['capital.loss']!=0,'capital.loss'] = 'Yes'
df_adult.loc[df_adult['capital.loss']!='Yes','capital.loss'] = 'No'

fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df_adult['capital.loss'], hue=df_adult['income'])
ax.set_title('Capital Loss vs Income')
plt.xlabel("Capital Loss",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('');


# In[ ]:


df_adult


# # Modeling

# Converts target values to numeric data and creates one hot encode of data

# In[ ]:


df_adult['income'] = df_adult['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
df_adult = pd.get_dummies(df_adult)


# ## Splitting data into Train and Test

# In[ ]:


X = df_adult.drop(['income'], axis = 1) 
y = df_adult.iloc[:,0]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.3)


# ## Logistic Regression

# ### Fit & Predict

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
model_1 = logreg.fit(X_train,y_train)

pred_1 = model_1.predict(X_test)


# ### Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('\nConfustion Matrix')
print(confusion_matrix(y_test, pred_1))

from sklearn.metrics import accuracy_score
print('\nTest Accuracy')
print(accuracy_score(y_test, pred_1))

from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, pred_1))


# ## Decision Tree Classifier

# ### Fit & Predict

# In[ ]:


from sklearn import tree
dectree = tree.DecisionTreeClassifier()
model_2 = dectree.fit(X_train,y_train)

pred_2 = model_2.predict(X_test)


# ### Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('\nConfustion Matrix')
print(confusion_matrix(y_test, pred_2))

from sklearn.metrics import accuracy_score
print('\nTest Accuracy')
print(accuracy_score(y_test, pred_2))

from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, pred_2))


# ## Random Forest Classifier

# ### Fit & Predict

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model_3 = rf.fit(X_train, y_train)

pred_3 = model_3.predict(X_test)


# ### Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('\nConfustion Matrix')
print(confusion_matrix(y_test, pred_3))

from sklearn.metrics import accuracy_score
print('\nTest Accuracy')
print(accuracy_score(y_test, pred_3))

from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, pred_3))


# ## Naive Bayes

# ### Fit & Predict

# In[ ]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
model_4 = NB.fit(X_train,y_train)

pred_4 = model_4.predict(X_test)


# ### Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('\nConfustion Matrix')
print(confusion_matrix(y_test, pred_3))

from sklearn.metrics import accuracy_score
print('\nTest Accuracy')
print(accuracy_score(y_test, pred_3))

from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, pred_3))


# ## KNN

# ### Fit & Predict

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
model_5 = classifier.fit(X_train, y_train)

pred_5 = model_5.predict(X_test)


# ### Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('\nConfustion Matrix')
print(confusion_matrix(y_test, pred_5))

from sklearn.metrics import accuracy_score
print('\nTest Accuracy')
print(accuracy_score(y_test, pred_5))

from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, pred_5))


# ## XGBoost

# ### Fit & Predict

# In[ ]:


from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
model_6 = XGBClassifier.fit(X_train, y_train)
pred_6 = model_6.predict(X_test)


# ### Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('\nConfustion Matrix')
print(confusion_matrix(y_test, pred_6))

from sklearn.metrics import accuracy_score
print('\nTest Accuracy')
print(accuracy_score(y_test, pred_6))

from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, pred_6))

