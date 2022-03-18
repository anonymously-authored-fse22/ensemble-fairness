#!/usr/bin/env python
# coding: utf-8

# Classification Problem 
# "Bank Marketing Data Set"
# 
#     source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv("../input/bank-marketing/bank-additional-full.csv",sep=';')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.apply(lambda x : len(x.unique()))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#correlation_matrix
corr_m = df.corr() 
f, ax = plt.subplots(figsize =(7,6)) 
sns.heatmap(corr_m,annot=True, cmap ="YlGnBu", linewidths = 0.1)


# In[ ]:


sns.scatterplot(x = 'previous',y = 'pdays',data = df,alpha = 0.5);


# In[ ]:


#sns.scatterplot(x = 'day',y = 'campaign',data = df,alpha = 0.5);


# In[ ]:


sns.countplot(x = 'age', data = df)


# In[ ]:


sns.catplot('contact',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('marital',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('education',kind = 'count',data = df,aspect =3)


# In[ ]:


df['education'].replace({'unknown':'secondary'},inplace = True)


# In[ ]:


sns.catplot('education',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('default',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('housing',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('loan',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('y',kind = 'count',data = df,aspect =3)


# In[ ]:


df['contact'].replace({'unknown':'cellular'},inplace = True)


# In[ ]:


sns.catplot('contact',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('month',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('poutcome',kind = 'count',data = df,aspect =3)


# In[ ]:


cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(exclude=['object']).columns


# In[ ]:


cat_cols,num_cols


# In[ ]:


#treat for age column 
age_q1=df['age'].quantile(q = 0.25)
age_q2=df['age'].quantile(q = 0.50)
age_q3=df['age'].quantile(q = 0.75)
age_q4=df['age'].quantile(q = 1.00)


# In[ ]:


print('Quartiles:',age_q1,age_q2,age_q3,age_q4)
outliers=age_q3+1.5*age_q3-1.5*age_q1
print('outliers:',outliers)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le=LabelEncoder()
def clusters(x):
    if x<=33:
        return 0
    elif x>32 & x<=49:
        return 1
    elif x>49 & x<=73:
        return 2
    elif x>73 & x<=87:
        return 4
df['age'] = df['age'].astype('int').apply(clusters)
#df['age_new'] = le.fit_transform(df['age_new'])


# In[ ]:


df.head()


# In[ ]:


#converting cat to conti...
df['marital'] = le.fit_transform(df['marital'])
df['education'] = le.fit_transform(df['education'])
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['poutcome'] = le.fit_transform(df['poutcome'])
df['y'] = le.fit_transform(df['y'])


# In[ ]:


df.head(10)


# In[ ]:


df.apply(lambda x : len(x.unique()))


# In[ ]:


#explore numerical columns
sns.catplot('job',kind = 'count',data = df,aspect =3)


# In[ ]:


df['job'].replace({'unknown':'self-employed'},inplace=True)


# In[ ]:


sns.catplot('job',kind = 'count',data = df,aspect =3)


# In[ ]:


#sns.catplot('day',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('campaign',kind = 'count',data = df,aspect =3)


# In[ ]:


sns.catplot('previous',kind = 'count',data = df,aspect =3)


# In[ ]:


df['month'] = le.fit_transform(df['month'])
df['job'] = le.fit_transform(df['job'])


# In[ ]:


df.head()


# In[ ]:


#treat for age column 
dur_q1=df['duration'].quantile(q = 0.25)
dur_q2=df['duration'].quantile(q = 0.50)
dur_q3=df['duration'].quantile(q = 0.75)
dur_q4=df['duration'].quantile(q = 1.00)


# In[ ]:


print('Quartiles:',dur_q1,dur_q2,dur_q3,dur_q4)
outliers2=dur_q3+1.5*dur_q3-1.5*dur_q1
print('outliers:',outliers2)


# In[ ]:


def clusters2(y):
    if y<=104:
        return 0
    elif y>104 & y<=185:
        return 1
    elif y>185 & y<=329:
        return 2
    elif y>329 & y<=666.5:
        return 4
    elif y>666.5:
        return 5
df['duration'] = df['duration'].astype('int').apply(clusters2)
#df['age_new'] = le.fit_transform(df['age_new'])


# In[ ]:


df.head()


# In[ ]:


df.apply(lambda x : len(x.unique()))


# In[ ]:


df['marital'].value_counts()


# In[ ]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()


# In[ ]:


df['pdays'] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df['pdays'])))


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# In[ ]:


X=df.drop(['balance','y'],axis=1)
y=df['y']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)


# In[ ]:


#model_1 (LogisticRegression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
lr=LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)
y_pred1=lr.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix1=confusion_matrix(y_test, y_pred1)
precision_score1=precision_score(y_test, y_pred1)
recall_score1=recall_score(y_test, y_pred1)
accuracy_score1=accuracy_score(y_test, y_pred1)
f1_score1=f1_score(y_test, y_pred1)


# In[ ]:


print('confusion_matrix:\n',confusion_matrix1)
print('precision_score:',precision_score1)
print('recall_score:',recall_score1)
print('accuracy_score:',accuracy_score1)
print('f1_score:',f1_score1)


# In[ ]:


print(classification_report(y_test, y_pred1))


# In[ ]:





# In[ ]:


#model_2 (RandomForest)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
y_pred2=rfc.predict(X_test)


# In[ ]:


confusion_matrix2=confusion_matrix(y_test, y_pred2)
precision_score2=precision_score(y_test, y_pred2)
recall_score2=recall_score(y_test, y_pred2)
accuracy_score2=accuracy_score(y_test, y_pred2)
f1_score2=f1_score(y_test, y_pred2)


# In[ ]:


print('confusion_matrix:\n',confusion_matrix2)
print('precision_score:',precision_score2)
print('recall_score:',recall_score2)
print('accuracy_score:',accuracy_score2)
print('f1_score:',f1_score2)


# In[ ]:


print(classification_report(y_test, y_pred2))


# In[ ]:





# In[ ]:


#model_3 (DecisionTrees)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtc.fit(X_train, y_train)
y_pred3 = dtc.predict(X_test)


# In[ ]:


confusion_matrix3=confusion_matrix(y_test, y_pred3)
precision_score3=precision_score(y_test, y_pred3)
recall_score3=recall_score(y_test, y_pred3)
accuracy_score3=accuracy_score(y_test, y_pred3)
f1_score3=f1_score(y_test, y_pred3)


# In[ ]:


print('confusion_matrix:\n',confusion_matrix3)
print('precision_score:',precision_score3)
print('recall_score:',recall_score3)
print('accuracy_score:',accuracy_score3)
print('f1_score:',f1_score3)


# In[ ]:





# In[ ]:


#model_4 (xgboost)
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
y_pred4 = xgbc.predict(X_test)


# In[ ]:


confusion_matrix4=confusion_matrix(y_test, y_pred4)
precision_score4=precision_score(y_test, y_pred4)
recall_score4=recall_score(y_test, y_pred4)
accuracy_score4=accuracy_score(y_test, y_pred4)
f1_score4=f1_score(y_test, y_pred4)


# In[ ]:


print('confusion_matrix:\n',confusion_matrix4)
print('precision_score:',precision_score4)
print('recall_score:',recall_score4)
print('accuracy_score:',accuracy_score4)
print('f1_score:',f1_score4)


# # Final Results

# In[ ]:


F_scores = {'Model':  ['Log_R', 'RF','DT','XGB'],
         'conf_matrix': [confusion_matrix1, confusion_matrix2 , confusion_matrix3, confusion_matrix4],
         'precision': [precision_score1,precision_score2,precision_score3,precision_score4],
         'recall': [recall_score1,recall_score2,recall_score3,recall_score4],
         'accuracy': [accuracy_score1,accuracy_score2,accuracy_score3,accuracy_score4],
         'f1': [f1_score1,f1_score2,f1_score3,f1_score4] 
           }


# In[ ]:


df_scores = pd.DataFrame (F_scores, columns = ['Model','conf_matrix','precision','recall','accuracy','f1'])
df_scores


# In[ ]:


print(df_scores.to_markdown(tablefmt="grid"))


# # Thank you

# In[ ]:





# In[ ]:


#Additional graphs to compare actual vs predicted
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
ax1 = sns.distplot(df1['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df1['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)


# In[ ]:


df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
ax1 = sns.distplot(df2['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df2['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)


# In[ ]:


df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred3})
ax1 = sns.distplot(df3['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df3['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)


# In[ ]:


df4 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred4})
ax1 = sns.distplot(df4['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df4['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)


# In[ ]:




