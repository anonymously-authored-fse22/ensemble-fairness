#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/bank-marketing/bank-additional-full.csv")
df.head(5)


# In[ ]:


df = pd.read_csv("../input/bank-marketing/bank-additional-full.csv",sep=';')
df.head(5)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df[df.duplicated()]


# In[ ]:


df=df.drop(df[df.duplicated()].index).reset_index(drop=True)


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.head()


# In[ ]:


for i in df.columns:
    print('{}:{}'.format(i,len(df[i].unique())))


# In[ ]:


df.describe()


# In[ ]:


pd.options.display.max_columns = None
df.head()


# In[ ]:


sns.distplot(df['age'])


# In[ ]:


sns.boxplot(df['age'])


# In[ ]:


df[df['age']>70].sort_values(by='age')


# In[ ]:


plt.subplots(figsize = (25, 5))
sns.countplot(df['age'])


# In[ ]:


import plotly.express as px
fig = px.histogram(df, x="age", marginal="box")
fig.show()


# In[ ]:


q1=df['age'].quantile(q = 0.25)
q3=df['age'].quantile(q = 0.75)
iqr=q3-q1
Upper_tail = q3 + 1.5 * iqr
med = np.median(df['age'])
for i in df['age']:
    if i>Upper_tail:
        df['age'] = df['age'].replace(i, med)


# In[ ]:


fig = px.histogram(df, x="age", marginal="box")
fig.show()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', data = df)


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# In[ ]:


fig = px.pie(df['job'].value_counts().reset_index(), values='job', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='job', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['marital'].value_counts().reset_index(), values='marital', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='marital', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['education'].value_counts().reset_index(), values='education', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='education', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['default'].value_counts().reset_index(), values='default', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='default', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['housing'].value_counts().reset_index(), values='housing', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='housing', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['loan'].value_counts().reset_index(), values='loan', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='loan', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['contact'].value_counts().reset_index(), values='contact', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='contact', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['month'].value_counts().reset_index(), values='month', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='month', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['day_of_week'].value_counts().reset_index(), values='day_of_week', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='day_of_week', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()
fig = px.pie(df['campaign'].value_counts().reset_index(), values='campaign', names='index',width=500,height=400)
fig.update_traces(textposition='inside',hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(annotations=[dict(text='campaign', x=0.5, y=0.5, font_size=8, showarrow=False)])
fig.show()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(4,4))
sns.countplot(x = 'marital', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'job', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'education', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'default', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'housing', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'loan', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'contact', data = df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(x = 'month', data = df)
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


df.info()


# In[ ]:


cat=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']


# In[ ]:


for i in df[cat]:
    lbl=LabelEncoder()
    df[i]=lbl.fit_transform(df[i])
    


# In[ ]:


pd.qcut(df['age'],4).value_counts()


# In[ ]:


df.loc[df['age']<=32,'age']=0
df.loc[(df['age']>32)&(df['age']<=38),'age']=1
df.loc[(df['age']>38)&(df['age']<=47),'age']=2
df.loc[df['age']>47,'age']=3


# In[ ]:


sns.kdeplot(data=df, x="duration")


# In[ ]:


fig = px.histogram(df, x="duration", marginal="box")
fig.show()


# In[ ]:


len(df[df['duration']>645])


# In[ ]:


df[(df['duration'] == 0)]


# In[ ]:


df=df[(df['duration'] != 0)].reset_index(drop=True)


# In[ ]:


fig = px.histogram(df,x='duration', marginal="box")
fig.show()


# In[ ]:


pd.qcut(df['duration'],5).value_counts()


# In[ ]:


df.loc[df['duration']<=89,'duration']=0
df.loc[(df['duration']>89)&(df['duration']<=146),'duration']=1
df.loc[(df['duration']>146)&(df['duration']<=207.252),'duration']=2
df.loc[(df['duration']>207.252)&(df['duration']<=264),'duration']=3
df.loc[(df['duration']>264)&(df['duration']<=492),'duration']=4


# 

# In[ ]:


Y=df['y'].replace({'no':0,'yes':1}).values


# In[ ]:


df.shape


# In[ ]:


Y.shape


# In[ ]:


X=df.iloc[:,0:-1].values


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=88)


# In[ ]:


train_X=train_X.reshape(-1,1)
test_X=test_X.reshape(-1,1)
train_Y= train_Y.reshape(-1,1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X= sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


#rbf-SVM
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))
sns.heatmap(confusion_matrix(test_Y,prediction1),annot=True,fmt='2.0f')
print('rbf-SVM\n',classification_report(test_Y,prediction1))


# In[ ]:


#linear-SVM
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
sns.heatmap(confusion_matrix(test_Y,prediction2),annot=True,fmt='2.0f')
print('linear-SVM\n',classification_report(test_Y,prediction2))


# In[ ]:


#Logistic Regression
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
sns.heatmap(confusion_matrix(test_Y,prediction3),annot=True,fmt='2.0f')
print('Logistic Regression\n',classification_report(test_Y,prediction3))


# In[ ]:


#Decision Tree Classifier
model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))
sns.heatmap(confusion_matrix(test_Y,prediction4),annot=True,fmt='2.0f')
print('Decision Tree Classifier\n',classification_report(test_Y,prediction4))


# In[ ]:


#KNeighbors Classifier
model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction5=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
sns.heatmap(confusion_matrix(test_Y,prediction5),annot=True,fmt='2.0f')
print('KNeighbors Classifier\n',classification_report(test_Y,prediction5))


# In[ ]:


a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())


# In[ ]:


#Random Forest Classifier
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))
sns.heatmap(confusion_matrix(test_Y,prediction7),annot=True,fmt='2.0f')
print('Random Forest Classifier\n',classification_report(test_Y,prediction7))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




