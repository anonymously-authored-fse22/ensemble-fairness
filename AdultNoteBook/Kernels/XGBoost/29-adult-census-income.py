#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix

from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
import warnings
warnings.filterwarnings ("ignore")


# In[ ]:


df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.head()


# In[ ]:


print("Data Info          :")
df.info()

print("\nFeatures         :",df.shape[1],'\nRows             :',df.shape[0])
print("\nAny null values? :" ,df.isnull().any().sum())


# In[ ]:


print("Features :", df.columns.values[:-1])
print("Target   :", df.columns[-1])


# # Analyze Data

# In[ ]:


cat_cols = df.select_dtypes(include='O').columns
num_cols = df.select_dtypes(include='int').columns


# In[ ]:


print("Numeric Columns:\n",num_cols)


# In[ ]:


print("Unique Values In Categorical Columns\n")
for c in cat_cols:
    print(c)
    print( df[c].value_counts())
    print("\n")


# In[ ]:


df.replace({'?':'unknown'},inplace=True)
df.drop(['fnlwgt'],axis=1,inplace=True)


# # Data Visualization

# In[ ]:


plt.figure(figsize=(25,15))

plt.subplot(311)
sns.countplot(x='age',hue='income',data=df)
plt.xticks(rotation=35) 
plt.ylabel("")

plt.subplot(312)
sns.countplot(x='education.num',hue='income',data=df)
plt.xticks(rotation=35) 
plt.ylabel("")

plt.subplot(313)
sns.countplot(x='hours.per.week',hue='income',data=df)
plt.xticks(rotation=35) 
plt.ylabel("")

plt.subplots_adjust(hspace=1) 
plt.show()


# In[ ]:


sns.set(font_scale=2)
plt.figure(figsize=(32,16)) 

plt.subplot(321)
sns.countplot(x="workclass",hue="income",data=df)
plt.xticks(rotation=45) 

plt.subplot(322)
sns.countplot(x="marital.status",hue="income",data=df)
plt.xticks(rotation=45)

plt.subplot(323)
sns.countplot(x="sex",hue="income",data=df)
plt.xticks(rotation=45)

plt.subplot(324)
sns.countplot(x="race",hue="income",data=df)
plt.xticks(rotation=45)

plt.subplot(325)
sns.countplot(x="occupation",hue="income",data=df)
plt.xticks(rotation=45)


plt.subplot(326)
sns.countplot(x="relationship",hue="income",data=df)
plt.xticks(rotation=45)


plt.subplots_adjust(hspace=1) 
plt.show()


# In[ ]:


plt.figure(figsize=(32,16)) 
sns.countplot(x="workclass",hue="income",data=df)


# In[ ]:


plt.figure(figsize=(24, 12))
df1 = df.drop('income',axis=1)
corr = df1.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, annot=True,yticklabels=corr.columns, 
                 linewidths=.2, cmap="YlGnBu")


# Sex and marital status and relationship are related while native.country and race are related. Will drop the unwanted further.

# # Feature Engineering

# In[ ]:


df['marital.status'] = df['marital.status'].replace({"Never-married":"Single","Divorced":"Single","Separated":"Single",
                                                     "Widowed":"Single","Married-spouse-absent":"Single",
                                                     "Married-civ-spouse":"Married","Married-AF-spouse":"Married"})

df["workclass"] = df["workclass"].replace({"Private":"Paid_Employed","Self-emp-not-inc":"Paid_Employed",
                                           "Local-gov":"Paid_Employed","unknown":"Paid_Employed",
                                           "State-gov":"Paid_Employed","Self-emp-inc":"Paid_Employed",
                                           "Federal-gov":"Paid_Employed","Without-pay":"Unpaid_Employed",
                                           "Never-worked":"UnEmployed",})

df['education'] = df['education'].replace({'Preschool':'Not-grad','1st-4th':'Not-grad','5th-6th':'Not-grad',
                                           '7th-8th':'Not-grad','9th':'Not-grad','10th':'Not-grad',
                                           '11th':'Not-grad','12th':'Not-grad',})

df['native.country'] = np.where(df['native.country'].str.contains("United-States"), "United-States", "Other")


df['hours.per.week'] = np.where(df['hours.per.week'] < 40, "ls_40", "gt_40")
df = pd.concat([df,pd.get_dummies(df['hours.per.week'],prefix='hrs_per_wk')],axis=1).drop('hours.per.week',axis=1)

df['age'] = np.where(df['age'] < 45, "ls_45", "gt_45")

df = pd.concat([df,pd.get_dummies(df['age'],prefix='age')],axis=1).drop('age',axis=1)

df['income'] = df['income'].map({'<=50K':0,'>50K':1})
df['sex'] = df['sex'].map({'Male':0,'Female':1})

for col in cat_cols.drop(['sex','income','relationship','occupation','race']):
    df = pd.concat([df,pd.get_dummies(df[col],prefix=col)],axis=1).drop(col,axis=1)
    
df.drop(['relationship','education.num','sex','race'],axis=1,inplace=True)


# In[ ]:


le = LabelEncoder()
df['occupation'] = le.fit_transform(df['occupation'])


sc = StandardScaler()
df['capital.gain'] = sc.fit_transform(df[['capital.gain']].values).astype(int)
df['capital.loss'] = sc.fit_transform(df[['capital.loss']].values).astype(int)


df.info()


# In[ ]:


X = df.drop(['income'],axis=1)
y = df['income']

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3)


# # DecisionTreeClassifier

# In[ ]:


param_grid = {"criterion":['gini','entropy'], 
              "max_depth":[5,10,15,20]
             }    
grid = GridSearchCV(DecisionTreeClassifier(), param_grid,verbose=True)
grid.fit(X_train,y_train)
best_param = grid.best_params_
best_param


# In[ ]:


dt_model = DecisionTreeClassifier(criterion=best_param['criterion'],max_depth=best_param['max_depth'])
dt_model.fit(X_train,y_train)

dt_pred  =dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(dt_accuracy)
print(confusion_matrix(y_test, dt_pred))


# # RandomForestClassifier

# In[ ]:


param_grid = {"n_estimators":[5,20,50], 'max_depth':range(5,16,5), 'min_samples_split':range(200,1001,500),
              'min_samples_leaf':range(30,71,20), 
             }    
grid = GridSearchCV(RandomForestClassifier(), param_grid,verbose=True)
grid.fit(X_train,y_train)
best_param = grid.best_params_
best_param


# In[ ]:


rfc_model = RandomForestClassifier(max_depth = best_param['max_depth'],
                                   min_samples_leaf = best_param['min_samples_leaf'],
                                   min_samples_split = best_param['min_samples_split'],
                                   n_estimators = best_param['n_estimators'])
rfc_model.fit(X_train,y_train)
rfc_pred =  rfc_model.predict(X_test)

rfc_accuracy = accuracy_score(y_test, rfc_pred)
print(rfc_accuracy)

print(confusion_matrix(y_test, rfc_pred))


# # Logistic Regression

# In[ ]:


param_grid={'C': np.logspace(-3, 0, 20)}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(X_train,y_train)
best_param = grid.best_params_
best_param


# In[ ]:


log_model = LogisticRegression(C = best_param['C'])
log_model.fit(X_train,y_train)
log_pred =  log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)

print(log_accuracy)
print(confusion_matrix(y_test, log_pred))


# # GradientBoostingClassifier

# In[ ]:


param_test = {"n_estimators":[5,20,50,100],
               'max_depth':range(5,10,15),
                "learning_rate":[0.1,1,10]}

grid = GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=param_test,verbose=True);
grid.fit(X_train,y_train)
best_param = grid.best_params_
best_param


# In[ ]:


gbc_model = GradientBoostingClassifier(n_estimators=best_param['n_estimators'],max_depth=best_param['max_depth'],learning_rate=best_param['learning_rate'])
gbc_model.fit(X_train,y_train)
gbc_pred = gbc_model.predict(X_test)
gbc_accuracy = accuracy_score(y_test,gbc_pred)

print(gbc_accuracy )
print(confusion_matrix(y_test,gbc_pred))


# # XGBClassifier

# In[ ]:


param_test = {"n_estimators":[5,20,50,100],
               'max_depth':range(5,10,15),
                "learning_rate":[0.1,1,10]}

grid = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic',eval_metric = "error"),param_grid=param_test,verbose=True);
grid.fit(X_train,y_train)
best_param = grid.best_params_
best_param


# In[ ]:


xgb_model = XGBClassifier(learning_rate=best_param['learning_rate'], max_depth = best_param['max_depth'], n_estimators = best_param['n_estimators'],objective='binary:logistic',eval_metric = "error")
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test,xgb_pred)

print(xgb_accuracy)
print(confusion_matrix(y_test,xgb_pred))


# # Comparision of accuracies of different models

# In[ ]:


models_scores = pd.DataFrame({'accuracy':[log_accuracy,dt_accuracy,rfc_accuracy,gbc_accuracy,xgb_accuracy]
                             
                             },index = ['Logic Regression','Decision Tree','Random Forest','Gradient Boost','XGB Boost'])

models_scores

