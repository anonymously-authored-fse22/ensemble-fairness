#!/usr/bin/env python
# coding: utf-8

# # **<h1 style="color:red">Hello folks, in case you like this notebook dont forget to <span style="color:purple">UPVOTE</span> it and thanks for viewing :)</h1>**

# # **<h1 style="color:green;">Variable Notes :</h1>**
# <p style="color:purple;">pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# </p>
# <p style="color:purple;">
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# </p>  
# <p style="color:purple;">
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# </p>
# <p style="color:purple;">
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# </p>

# In[1]:


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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **<h1 style="color:skyblue;">Importing Libraries :</h1>**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,log_loss,accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
sns.set_palette('pastel')


# # **<h1 style="color:skyblue;">Loading the dataset :</h1>**

# In[3]:


train = pd.read_csv(r'/kaggle/input/titanic/train.csv')
test = pd.read_csv(r'/kaggle/input/titanic/test.csv')
train.tail()


# In[4]:


test.head()


# In[5]:


train.head()


# In[6]:


print(train.shape)
print(test.shape)


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


train.describe()


# In[10]:


test.describe()


# In[11]:


train.dtypes


# # **<h1 style="color:skyblue;">Null Values :</h1>**

# In[12]:


train.isnull().sum()


# In[13]:


test.isnull().sum()


# **<p style="color:purple;">The age feature consists of many missing values.</p>**

# **<p style="color:purple;">To handle the missing values in the age column of the dataset I have calculated the average age of the males and the females in the dataset and replaced the missing values accoring to their sex.</p>**

# In[14]:


avg_age_train = (train.groupby("Sex")['Age']).mean()
print(avg_age_train)
avg_age_test = (test.groupby("Sex")['Age']).mean()
print(avg_age_test)


# In[15]:


for i in range(len(train['Age'])):
    if train['Age'].isnull()[i] == True and train['Sex'][i] == 'male':
        train['Age'][i] = np.round(avg_age_train['male'],decimals=1)
    elif train['Age'].isnull()[i] == True and train['Sex'][i] == 'female':
        train['Age'][i] = np.round(avg_age_train['female'],decimals=1)
        
for i in range(len(test['Age'])):
    if test['Age'].isnull()[i] == True and test['Sex'][i] == 'male':
        test['Age'][i] = np.round(avg_age_test['male'],decimals=1)
    elif test['Age'].isnull()[i] == True and test['Sex'][i] == 'female':
        test['Age'][i] = np.round(avg_age_test['female'],decimals=1)


# **<p style="color:purple;">Therefore the missing age values are handled.</p>**

# In[16]:


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[17]:


print(train.shape)
print(test.shape)


# In[18]:


train.isnull().sum()


# In[19]:


test.isnull().sum()


# **<p style="color:purple;">I will handle the remaining missing values in the Cabin column after some feature engineering.</p>**

# **<p style="color:green;">Combining the dataset :</p>**

# In[20]:


Y = train['Survived']
train = train.drop('Survived',axis=1)
data = pd.concat([train,test],axis=0)
data.head()


# **<p style="color:purple;">I am going to replace the missing value in the Fare column by the average fare of according to the Sex</p>**

# In[21]:


avg_fare = data.groupby("Sex")['Fare'].mean()
avg_fare


# In[22]:


print("Index of the null value is: ",test[test['Fare'].isnull()].index.tolist())
print(test['Sex'][152])


# In[23]:


data['Fare'][152] = avg_fare['male']


# **<p style="color:purple;">Hence the missing value in the Fare column in the data is handled</p>**

# In[24]:


data.isnull().sum()


# **<p style="color:purple;">The missing values in the cabin column will be handled after some feature engineering.</p>**

# # **<h1 style="color:skyblue;">Duplicate Values :**

# In[25]:


print("Number of duplicate rows in the train dataset :",train.duplicated().sum())
print("Number of duplicate rows in the test dataset :",test.duplicated().sum())


# **<p style="color:purple">There are no duplicate rows present in the dataset .**

# # **<h1 style="color:skyblue;">Feature Engineering :**

# 1. **<p style="color:green;">Creating a feature with the titles of the name.</p>**

# In[26]:


Name_title_data = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(Name_title_data)
data['Name_title'] = Name_title_data
data = data.reset_index(drop=True)
data.head()


# 2. **<p style="color:green;">Creating the category of the age section.**

# In[27]:


age_group_data = [None] * len(data['Age'])
for i in range(len(data['Age'])):
    if data['Age'][i] <= 3:
        age_group_data[i] = 'Baby'
    elif data['Age'][i] >3 and data['Age'][i] <= 13:
        age_group_data[i] = 'Child'
    elif data['Age'][i] >13 and data['Age'][i] <= 19:
        age_group_data[i] = 'Teenager'
    elif data['Age'][i] >19 and data['Age'][i] <= 30:
        age_group_data[i] = 'Young Adult'
    elif data['Age'][i] >30 and data['Age'][i] <= 45:
        age_group_data[i] = 'Middle Aged Adult'
    elif data['Age'][i] >45 and data['Age'][i] <65:
        age_group_data[i] = 'Adult'
    else:
        age_group_data[i] = 'Old'

data['age_group'] = age_group_data


# In[28]:


np.unique(data['Name_title'])


# 3. **<p style="color:green;">Creating features that the person is married or not,and the family size with SibSp and parch column**

# In[29]:


data['Is_Married'] = 0
data['Is_Married'].loc[data['Name_title'] == 'Mrs'] = 1
data['FamSize'] = data['SibSp'] + data['Parch'] + 1
data['Single'] = data['FamSize'].map(lambda s: 1 if s == 1 else 0)


# In[30]:


data.head()


# 4. **<p style="color:green;">Creating a feature which tells us that the person is travelling with someone or not according to the similar number on the tickets.**

# In[31]:


np.unique(data['Ticket'])
tic = data.groupby('Ticket',sort=True,group_keys=True)
groups = list(tic.groups)
togther = [None] * len(data['Ticket'])
k=0
for i in range(len(groups)):
    for j in range(len(data['Ticket'])):
        if data['Ticket'][j] == groups[i]:
            togther[j] = i
data['Togther'] = togther


# In[32]:


data.head()


# 5. **<p style="color:green;">A feature which categorizes the fare rates of the person.**

# In[33]:


np.unique(data['Fare'])


# In[34]:


rates = [None]*len(data['Fare'])
for i in range(len(data['Fare'])):
    if data['Fare'][i]<=10:
        rates[i] = 1
    elif data['Fare'][i] >10 and data['Fare'][i]<=30:
        rates[i] = 2
    elif data['Fare'][i] >30 and data['Fare'][i]<=70:
        rates[i] = 3
    elif data['Fare'][i] >70 and data['Fare'][i]<=100:
        rates[i] = 4 
    else:
        rates[i] = 5
data['Rates'] = rates


# 6. **<p style="color:green;">A feature which tells us the the cabin value is present or not since the cabin feature has so many null values.**

# In[35]:


data['Cabin_present'] = 1
data['Cabin_present'].loc[data['Cabin'].isnull()] = 0


# In[36]:


data.shape


# **<p style="color:purple;">Now removing the useless columns.**

# In[37]:


data = data.drop('Cabin',axis=1)
data = data.drop('Ticket',axis=1)
data = data.drop('Name',axis=1)
data = data.drop('PassengerId',axis=1)


# # **<h1 style="color:skyblue;">One-Hot Encoding of features :**

# In[38]:


data_ohe = pd.get_dummies(data,drop_first=True)
data_ohe.head()


# # **<h1 style="color:skyblue;">Data Visualisation :**

# **<p style="color:green;">Hetamaps :**

# In[39]:


plt.figure(figsize=(20,20))
sns.heatmap(train.corr(),annot=True,fmt="0.3f",cmap='GnBu',linewidth=1.2,linecolor='black',square=True)
plt.show()


# In[40]:


plt.figure(figsize=(20,20))
sns.heatmap(test.corr(),annot=True,fmt="0.3f",cmap='YlOrBr',linewidth=1.2,linecolor='black',square=True)
plt.show()


# **<p style="color:purple;">Bar graphs and CountPlots :**

# In[41]:


train['Survived'] = Y


# In[42]:


plt.figure(figsize=(20,12))

plt.subplot(2,2,1)
sns.countplot('Sex',data=train,palette=['darkblue','red'])
plt.title("Train data Sex Count")
plt.grid()

plt.subplot(2,2,2)
sns.countplot('Sex',data=test,palette=['teal','purple'])
plt.title("Test data Sex Count")
plt.grid()

plt.show()


# In[43]:


plt.figure(figsize=(20,12))
plt.subplot(2,2,1)
sns.countplot('Survived',data=train,palette=['black','yellow'],hue='Sex')
plt.grid()
plt.title("Survival Graph")

plt.subplot(2,2,2)
sns.countplot('SibSp',data=train,palette=['green','pink'],hue='Sex')
plt.grid()
plt.title("No of siblings / spouses aboard the Titanic")

plt.subplot(2,2,3)
sns.countplot('Parch',data=train,palette=['orange','greenyellow'],hue='Sex')
plt.grid()
plt.title("No of parents / children aboard the Titanic")
plt.legend(loc='upper right')

plt.subplot(2,2,4)
sns.countplot('Pclass',data=train,palette=['brown','magenta'],hue='Sex')
plt.grid()
plt.title("Passenger Class with sex")
plt.legend(loc='upper right')

plt.show()


# In[44]:


plt.figure(figsize=(20,20))
sns.countplot(y='Age',data=train)
plt.grid()
plt.title("Train data Age ranges")
plt.show()


# In[45]:


plt.figure(figsize=(20,20))
sns.countplot(y='Age',data=test)
plt.grid()
plt.title("Test data Age ranges")
plt.show()


# In[46]:


plt.figure(figsize=(20,12))

plt.subplot(2,2,1)
sns.countplot('Embarked',data=train,hue='Survived',palette=['red','purple'])
plt.grid()
plt.title("Embarked plotted")

plt.subplot(2,2,2)
sns.countplot('Pclass',data=train,hue='Survived',palette=['teal','darkblue'])
plt.grid()
plt.title("Types of Passenger Classes")

plt.subplot(2,2,3)
sns.countplot('Parch',data=train,palette=['orange','greenyellow'],hue='Survived')
plt.grid()
plt.title("No of parents / children aboard the Titanic")
plt.legend(loc='upper right')

plt.subplot(2,2,4)
sns.countplot('SibSp',data=train,palette=['brown','magenta'],hue='Survived')
plt.grid()
plt.title("No of siblings / spouses aboard the Titanic")
plt.legend(loc='upper right')

plt.show()


# **<p style="color:green;">Pie Charts :**

# In[47]:


fig = px.pie(train,names='Sex',color='Survived')
fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))
fig.show()


# In[48]:


fig = px.pie(train,names='Embarked',color='Survived',color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))
fig.show()


# In[49]:


fig = px.pie(train,names='Pclass',color='Survived',color_discrete_sequence=px.colors.sequential.GnBu)
fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))
fig.show()


# In[50]:


fig = px.pie(train,names='SibSp',color='Survived',template='seaborn')
fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))
fig.show()


# **<p style="color:green;">Violin Plots :**

# In[51]:


fig = px.violin(train,x='Sex',y='Age',points='all',box=True,color='Survived')
fig.show()

fig = px.violin(train,x='Sex',y='Pclass',points='all',box=True,color='Survived')
fig.show()

fig = px.violin(train,x='Sex',y='SibSp',points='all',box=True)
fig.show()


# In[52]:


fig = px.violin(train,x='Survived',y='Age',points='all',box=True,color='Survived')
fig.show()

fig = px.violin(train,x='Survived',y='Pclass',points='all',box=True,color='Survived')
fig.show()

fig = px.violin(train,x='Survived',y='SibSp',points='all',box=True)
fig.show()


# **<p style="color:green;">Scatter Plots :**

# In[53]:


fig = px.scatter(train,x='Age',y='Fare',color='Survived',size='Age')
fig.show()

fig = px.scatter(train,x='Age',y='Fare',color='Sex',size='Age')
fig.show()


# In[54]:


train = train.drop('Survived',axis=1)


# # **<h1 style="color:skyblue;">Data Modeling :**

# In[55]:


train_ohe = data_ohe[:train.shape[0]]
test_ohe = data_ohe[train.shape[0]:]


# In[56]:


len(data)


# # **<h1 style="color:skyblue;">Train-Test Split :**

# In[57]:


X_train,X_test,Y_train,Y_test = train_test_split(train_ohe,Y,test_size=0.2)


# In[58]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # **<h1 style="color:skyblue;">ML Models :**

# In[59]:


def plot_conf_matrix(Y_test,Y_pred):
    conf = confusion_matrix(Y_test,Y_pred)
    recall =(((conf.T)/(conf.sum(axis=1))).T)
    precision =(conf/conf.sum(axis=0))

    print("Confusion Matrix : ")
    class_labels = [0,1]
    plt.figure(figsize=(10,8))
    sns.heatmap(conf,annot=True,fmt=".3f",cmap="GnBu",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("Precision Matrix ; ")
    plt.figure(figsize=(10,8))
    sns.heatmap(precision,annot=True,fmt=".3f",cmap="YlOrBr",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("Recall Matrix ; ")
    plt.figure(figsize=(10,8))
    sns.heatmap(recall,annot=True,fmt=".3f",cmap="Blues",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()


# # **<h1 style="color:skyblue;">Random Forest :**

# In[60]:


# params = dict(
#     n_estimators = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],
#     criterion = ['gini','entropy'],
#     max_depth = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],
#     min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
# )
# rf = RandomForestClassifier()
# clf = RandomizedSearchCV(rf,params,random_state=0,verbose=0,n_jobs=-1,n_iter=20,cv=10)
# rsc = clf.fit(X_train,Y_train)
# rsc.best_params_


# In[61]:


rf = RandomForestClassifier(n_estimators=15,min_samples_leaf=6,max_depth=400,criterion='gini')
rf.fit(X_train,Y_train)
pred = rf.predict(X_test)
acc = accuracy_score(Y_test,pred)*100
print(acc)
plot_conf_matrix(Y_test,pred)


# # **<h1 style="color:skyblue;">GBDT :**

# In[62]:


# params = dict(
#     learning_rate = [0.001,0.01,0.1,1,10,100,1000],
#     n_estimators = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],
#     criterion = ['friedman_mse','mse','mae'],
#     max_depth = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],
#     min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
# )
# gbdt = GradientBoostingClassifier()
# clf = RandomizedSearchCV(gbdt,params,random_state=0,verbose=0,n_jobs=-1,n_iter=20,cv=10)
# gb = clf.fit(X_train,Y_train)
# gb.best_params_


# In[63]:


gbdt = GradientBoostingClassifier(n_estimators=700,min_samples_leaf=8,max_depth=1000,criterion='mse',learning_rate=0.01)
gbdt.fit(X_train,Y_train)
pred = gbdt.predict(X_test)
acc = accuracy_score(Y_test,pred)*100
print(acc)
plot_conf_matrix(Y_test,pred)


# #  **<h1 style="color:skyblue;">Voting Classifier :**

# In[64]:


vc = VotingClassifier(estimators=[('rf', rf), ('gbdt', gbdt)],voting='soft')
vc = vc.fit(X_train,Y_train)

pred = vc.predict(X_test)
acc = accuracy_score(Y_test,pred)*100
print(acc)
plot_conf_matrix(Y_test,pred)


# # **<h1 style="color:skyblue;">Predictions :**

# In[65]:


X_train.shape


# In[66]:


test['PassengerId']


# In[67]:


predictions = gbdt.predict(test_ohe)
predictions.shape


# In[68]:


submit = pd.DataFrame(test['PassengerId'],columns=['PassengerId'])
submit['Survived'] = predictions
submit.head()


# In[69]:


submit.to_csv("Submissions.csv",index=False)
print("Finished saving the file")


# **<p style="color:red;">And we are done with the predictions.</p>**

# **<p style="color:orange">Thanks for viewing.**
