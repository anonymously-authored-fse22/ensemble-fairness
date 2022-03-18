#!/usr/bin/env python
# coding: utf-8

# In[12]:


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


# In[13]:


# Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz 
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


# In[14]:


# Pipeline Libraries

from sklearn import neighbors
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


# In[15]:


# Model Libraries

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[16]:


data = pd.read_csv("/kaggle/input/adult-census-income/adult.csv", na_values='?')


# In[17]:


data.shape


# In[18]:


data.head()


# # **Pre-Processing Data**

# In[19]:


data.columns=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']


# In[20]:


data.isna().sum()


# In[21]:


# Replacing the NA values with mode 
for col in ['workclass', 'occupation', 'native_country']:
    data[col].fillna(data[col].mode()[0], inplace=True)


# 'education' and 'education_num' represent same kind of data. So, One of them can be removed.

# In[22]:


data = data.drop(['education'],axis=1)


# In[23]:


# Replacing '50K' with '0' and '1'

data = data.replace({'<=50K':0,'>50K':1})
data = data.replace({'<=50K':0,'>50K':1})


# # **Data Exploration**

# In[24]:


# Lists that will be manipulated in the data processing
list_columns = []
list_categorical_col = []
list_numerical_col = []


# In[25]:


def get_col(df: 'dataframe', type_descr: 'numpy') -> list:
    """
    Function get list columns 
    
    Args:
    type_descr
        np.number, np.object -> return list with all columns
        np.number            -> return list numerical columns 
        np.object            -> return list object columns
    """
    try:
        col = (df.describe(include=type_descr).columns)  # pandas.core.indexes.base.Index  
    except ValueError:
        print(f'Dataframe not contains {type_descr} columns !', end='\n')    
    else:
        return col.tolist()


# In[26]:


list_numerical_col = get_col(df=data,
                             type_descr=np.number)
list_categorical_col = get_col(df=data,
                               type_descr=np.object)
list_columns = get_col(df=data,
                       type_descr=[np.object, np.number])


# In[27]:


list_numerical_col


# In[28]:


list_categorical_col


# In[29]:


x = data[list_numerical_col].hist(figsize=[25,22], 
                                density=True, 
                                bins=25, 
                                grid=False, 
                                color='orange', 
                                zorder=2, 
                                rwidth=0.9)


# **Observation :** 
# Majority of value in 'capital_gain' and 'capital_loss' is 0 (zero).So we can drop both columns as they are sparse.

# In[30]:


sns.pairplot(data, kind='scatter', diag_kind='kde',corner=True, hue='income')


# **Observation :** There is no particular feature which is highly correlated.

# In[31]:


data.corr()


# In[32]:


sns.heatmap(data.corr(), annot=True, cmap="PiYG")


# **Observation :**
# - Except 'fnlwgt', all numerical features are positively correlated with 'income'. 

# In[33]:


eda_percentage = data['income'].value_counts(normalize = True).rename_axis('income').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'income', y = 'Percentage', data = eda_percentage.head(10), palette='RdGy_r')
eda_percentage


# **Observation :**
# - The number of records with income "<=50K" is around 76% and ">50K" is around 24%.
# - The class distribution is moderate imbalanced.

# In[34]:


def age_group(x):
    x = int(x)
    x = abs(x)
    if(x <= 18 ):
        return "Less than 18"    
    if( 18 < x < 31 ):
        return "19-30"
    if( 30 < x < 41 ):
        return "31-40"
    if( 40 < x < 51 ):
        return "41-50"
    if( 50 < x < 61 ):
        return "51-60"
    if( 60 < x < 71 ):
        return "61-70"
    else:
        return "Greater than 70"

data['age_group'] = data['age'].apply(age_group)


# In[35]:


plt.figure(figsize=(12,6))
order_list = ['Less than 18','19-30', '31-40', '41-50', '51-60', '61-70', 'Greater than 70']
sns.countplot(data['age_group'], hue = data['income'], palette='autumn_r', order = order_list)
plt.title('Income of Individuals of Different Age Groups', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)


# **Observation :**
# - There is gradual decrease in number of people earning less than $50K'.
# 
# - People earning more than $50K rise as they become older, up to a point.

# In[37]:


plt.figure(figsize=(12,6))
#order_list = ['19-30', '31-40', '41-50', '51-60', '61-70', 'Greater than 70']
sns.countplot(data['workclass'], hue = data['income'],  palette='autumn_r')
plt.title('Income of Individuals of Different Working CLasses', fontsize=14)
plt.xticks(fontsize=10,rotation = 45)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)


# **Observation**
# - There is significant income difference between people working in 'Private' working class.
# - The People working at 'federal government', 'local government' have almost similar number of people on both classes.
# - The 'Self Employed included' is the only working class having more people in more tha >50K category instead of less than <=$50K.

# In[39]:


ig = plt.figure(figsize = (12, 6))
sns.catplot(x = 'relationship', y='hours_per_week', data=data, kind="violin")
plt.xticks(rotation = 45)


# In[41]:


sns.displot(data, x='age',kind= 'kde', hue = 'income', fill= 'income')


# # **Models**

# In[42]:


data.head()


# In[43]:


data = data.drop(columns=['age_group'])


# In[45]:


sns.histplot(data, x='age')
plt.axvline(np.mean(data['age']), color='crimson', ls=':')
plt.show()


# Applying **Standardization** as it handles outliers better compared to Normalization.

# In[47]:


# Standardize the inputs
scaler = StandardScaler()
list_scaling = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
data[list_scaling] = scaler.fit_transform(data[list_scaling])


# In[48]:


for feature in list_categorical_col:
    le = preprocessing.LabelEncoder()
    data[feature] = le.fit_transform(data[feature])


# One-Hot encoding is also an option for label encoding.

# In[51]:


# Train-Test Split - Hold-out Method
X = data.drop(columns=['income'])
y = data.income


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# # KNN

# In[54]:


# Generate a kNN model
knn_model = neighbors.KNeighborsClassifier()
print(knn_model.get_params())

# Distance and Weight Array
n = np.arange(1,21)
dist = np.array(['euclidean', 'manhattan'])
weight = np.array(['uniform', 'distance'])

#Finding Best Performing Parameters
results = pd.DataFrame(columns = ['#neighbors','distance', 'weight', 'acc_score'])
index = 0
for i in n:
  for j in dist:
    for k in weight:
      knn_model = neighbors.KNeighborsClassifier(n_neighbors=i, metric=j, weights=k)
      knn_model.fit(X_train, y_train)
      pred = knn_model.predict(X_test)
      acc_score = accuracy_score(y_test, pred)
      #rmse = mean_squared_error(y_test, pred, squared=False)
      results.loc[index] = [i,j,k,acc_score]
      index +=1
print('All results')
print(tabulate(results, headers = 'keys', tablefmt = 'psql'))
print('Best performing model')      
print(tabulate(results[results['acc_score']==results['acc_score'].max()], headers = 'keys', tablefmt = 'psql'))


# In[55]:


knn_model = neighbors.KNeighborsClassifier(n_neighbors=20, metric='manhattan', weights='uniform')
knn_model.fit(X_train, y_train)
pred_knn = knn_model.predict(X_test)

acc_score_knn = accuracy_score(y_test, pred_knn)
recall_knn = recall_score(y_test, pred_knn)
precision_knn = precision_score(y_test, pred_knn)
f1_knn = f1_score(y_test, pred_knn)
roc_knn = roc_auc_score(y_test, pred_knn)

print(classification_report(y_test, pred_knn))


# # Logistic

# In[56]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
# Tune LogisticRegression

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)

rs_log_reg.best_params_

rs_log_reg.score(X_test, y_test)


# # Naive Bayes ( Gaussian NB )

# In[61]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
#print(classification_report(y_test, y_pred_nb))


# In[62]:


accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Gaussian Accuracy : ", accuracy_nb)


# In[63]:


recall_nb = recall_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
roc_nb = roc_auc_score(y_test, y_pred_nb)
print(classification_report(y_test, y_pred_nb))


# # Decision Tree Classifier

# In[64]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred_decision_tree = model.predict (X_test)
print('Accuracy: ', accuracy_score(y_test, pred_decision_tree))


# In[66]:


# Hyper Parameter Tuning - Decision Tree
ind = 1
crit = ['gini', 'entropy']
df_results = pd.DataFrame(columns = ['depth', 'puritymethod', 'Accuracy'])

for i in np.arange(1, 31):
  for j in crit:
    model = DecisionTreeClassifier(max_depth=i, criterion=j)
    model.fit(X_train, y_train)
    pred = model.predict (X_test)
    df_results.loc[ind] = [i, j, accuracy_score(y_test, pred)]
    ind+=1
sns.lineplot(x = 'depth', y = 'Accuracy', hue='puritymethod', data = df_results)


# In[67]:


print('Best performing model - Decision Tree')      
print(tabulate(df_results[df_results['Accuracy']==df_results['Accuracy'].max()], headers = 'keys', tablefmt = 'psql'))


# 

# # Neural  Networks

# In[71]:


model = MLPClassifier(max_iter=800, learning_rate_init= 0.005)
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc_neural= accuracy_score(y_test,pred)
print("Accuracy Score - Neural Networks ",acc_neural)


# In[ ]:


result = pd.DataFrame(columns=['Learning Rate', 'Transfer Function',  'Accuracy'])
lr = [0.00001, 0.0001, 0.01, 0.03, 1, 3, 10]
transfer_function = ['identity', 'logistic', 'tanh', 'relu']
k=0
for i in lr:
  for j in transfer_function:
    model = MLPClassifier(learning_rate_init= i, activation=j)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result.loc[k] = [i, j,  accuracy_score(y_test, pred)]
    k+=1
#print(tabulate(result, headers=result.columns, tablefmt='grid'))
print('All results')
print(tabulate(results, headers = result.columns, tablefmt = 'psql'))
print('Best performing model')      
print(tabulate(results[results['Accuracy']==results['Accuracy'].max()], headers = 'keys', tablefmt = 'psql'))


# In[ ]:





# 
