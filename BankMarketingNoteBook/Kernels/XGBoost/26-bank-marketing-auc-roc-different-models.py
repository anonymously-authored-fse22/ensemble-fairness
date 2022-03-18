#!/usr/bin/env python
# coding: utf-8

# # Bank_Marketing

# ## Classification problem - will client subscribe a deposit

# # Libraries

# In[ ]:


# Import data and modules
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from catboost import Pool, CatBoostClassifier


from scipy.stats import norm
from scipy import stats
from scipy.stats import skew 
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder


# # Data

# In[ ]:


data = pd.read_csv("../input/bank-marketing/bank-additional-full.csv", delimiter=';')


# In[ ]:


#Imbalanced data
print('Imbalanced data','\n',data['y'].value_counts())
print('Null',data.isnull().sum().sum())


# # Y

# In[ ]:



label=LabelEncoder()
data['y']=label.fit_transform(data['y'])


# # Age

# In[ ]:


sns.FacetGrid(data, hue="y", height=5)    .map(sns.distplot, "age")    .add_legend();
bins=[0,29,32,37,43,52,58,62,100]
for i in bins:
    plt.axvline(i,c='green',linewidth=1,linestyle="--")  #vertical line


# In[ ]:



labels = [1,2,3,4,5,6,7,8]
data['age_range'] = (pd.cut(data.age, bins, labels = labels)).astype(int)


# # Duration

# In[ ]:


list([1,2])


# In[ ]:


sns.FacetGrid(data, hue="y", height=5)    .map(sns.distplot, "duration")    .add_legend();
bins=[-1,30,100,180,319,650,1000,1800,5500]
for i in bins:
    plt.axvline(i,c='green',linewidth=1,linestyle="--")  #vertical line
labels = [1,2,3,4,5,6,7,8]
data['dur_range'] = (pd.cut(data.duration, bins, labels = labels)).astype(int)


# In[ ]:


data.dur_range.isnull().sum()


# # Pdays - days after 1st Call (999 if 0)

# In[ ]:


data['1st_call'] = data['pdays'].map(lambda x: 1 if x == 999 else 0)


# In[ ]:


data['1st_call'].value_counts()


# In[ ]:


data['pdays'] = data['pdays'].map(lambda x: 0 if x == 999 else x)


# # Num and Cat data

# In[ ]:


y=data['y'].copy()
data=data.drop(['y'],axis=1)


# In[ ]:


data.info()


# In[ ]:


data['campaign'] = data['campaign'].astype('object')
feat=data.columns


# In[ ]:


#data_cat = np.where(data[feat].dtypes == np.object)[0]


# In[ ]:


#dum object column
data= pd.get_dummies(data)


# In[ ]:


data.columns


# # Prediction

# In[ ]:


def model_mass_calc(X,y):

    #Some parameters

    svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

    #Split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    #Standartize

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    a=[]
   
    #Search knn_param
    a_index=list(range(1,11))
    knn=[1,2,3,4,5,6,7,8,9,10]
    a=[]
    for i in knn:
        model=KNeighborsClassifier(n_neighbors=i) 
        model.fit(X_train_std, y_train)
        prediction=model.predict(X_test_std)
        a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))


    #Max_Score_KNN
    knn=pd.DataFrame(knn)
    a=pd.DataFrame(a)
    knn_data=pd.concat([knn,a],axis=1)
    knn_data.columns=['Neig','Score']
    knn_take=int(knn_data[knn_data['Score']==knn_data['Score'].max()][:1]['Neig'])

    #model
    #SolveLater How to write names automat
    x=['CatB','XGB','RandomF','NB','svm.SVC','Log','DTr',str('KN='+str(knn_take))]
    #Form for cycle

    models=[CatBoostClassifier(),XGBClassifier(),RandomForestClassifier(),GaussianNB(),svm,LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=knn_take)]
    a_index=list(range(1,len(models)+1))
    a=[]
    for model in models:

        model.fit(X_train_std, y_train)
        prediction=model.predict(X_test_std)
        a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
    plt.plot(x, a)
    #plt.xticks(x)
    #MAX_Score+Model
    x=pd.DataFrame(x)
    a=pd.DataFrame(a)
    all_scores=pd.concat([x,a],axis=1)
    all_scores.columns=['model','Score']
    print('Max_score:',all_scores[all_scores['Score']==all_scores['Score'].max()])


# In[ ]:


model_mass_calc(data,y)


# ## The best XGB  0.92 by accuracy

# ### AUC

# In[ ]:


def model_mass_calc(X,y,Score):

    #Some parameters

    svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

    #Split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    #Standartize

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    a=[]
   
    #Search knn_param
    a_index=list(range(1,11))
    knn=[1,2,3,4,5,6,7,8,9,10]
    a=[]
    for i in knn:
        model=KNeighborsClassifier(n_neighbors=i) 
        model.fit(X_train_std, y_train)
        prediction=model.predict(X_test_std)
        a.append(pd.Series(Score(prediction,y_test)))


    #Max_Score_KNN
    knn=pd.DataFrame(knn)
    a=pd.DataFrame(a)
    knn_data=pd.concat([knn,a],axis=1)
    knn_data.columns=['Neig','Score']
    knn_take=int(knn_data[knn_data['Score']==knn_data['Score'].max()][:1]['Neig'])

    #model
    #SolveLater How to write names automat
    x=['CatB','XGB','RandomF','NB','svm.SVC','Log','DTr',str('KN='+str(knn_take))]
    #Form for cycle

    models=[CatBoostClassifier(),XGBClassifier(),RandomForestClassifier(),GaussianNB(),svm,LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=knn_take)]
    a_index=list(range(1,len(models)+1))
    a=[]
    for model in models:

        model.fit(X_train_std, y_train)
        prediction=model.predict(X_test_std)
        a.append(pd.Series(Score(prediction,y_test)))
    plt.plot(x, a)
    #plt.xticks(x)
    #MAX_Score+Model
    x=pd.DataFrame(x)
    a=pd.DataFrame(a)
    all_scores=pd.concat([x,a],axis=1)
    all_scores.columns=['model','Score']
    print('Max_score:',all_scores[all_scores['Score']==all_scores['Score'].max()])


# In[ ]:


model_mass_calc(data,y,metrics.roc_auc_score)


# #### The best XGB  0.81 by auc_roc

# # Check XGB stability using Cross_Validation

# In[ ]:




