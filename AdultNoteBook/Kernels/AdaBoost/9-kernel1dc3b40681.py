#!/usr/bin/env python
# coding: utf-8

# # Adult census income

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing


# In[2]:


pd.set_option('display.max_columns',500)


# In[3]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[4]:



data1 = pd.read_csv("../input/adult-census-income/adult.csv")
x=data1.drop('income',axis=1)
y=data1['income']


# In[5]:


x,y = train_test_split(data1,random_state=7)


# In[ ]:





# In[6]:


x.head()


# In[7]:


x.groupby('income').size()


# In[8]:


x.info()


# In[9]:


x.describe()


# In[10]:


x.shape


# In[11]:


(x=='?').sum()


# In[12]:


((x=='?').sum()*100/32561).round(2)


# In[13]:


((y=='?').sum()*100/32561).round(2)


# In[14]:


#data[data[::] != '?']
x = x[(x['workclass']!='?')& (x['occupation']!='?') & (x['native.country']!='?')]


# In[15]:


#data[data[::] != '?']
y = y[(y['workclass']!='?')& (y['occupation']!='?') & (y['native.country']!='?')]


# In[16]:


(x=='?').sum()


# In[17]:


(y=='?').sum()


# In[18]:


x.info()


# In[19]:


sns.pairplot(x)


# In[20]:


correlation = x.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
#sns.heatmap(x.select_dtypes([object]), annot=True, annot_kws={"size": 7})


# In[21]:



name = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

for c in name:
    sns.boxplot(x=x[c],data=x)

    plt.show()


# In[22]:


x.select_dtypes(['object']).head()


# In[23]:


x['income'].unique()


# In[24]:


x['workclass'].unique()


# In[25]:


x['education'].unique()


# In[26]:


x['occupation'].unique()


# In[27]:


x['sex'].unique()


# In[28]:


x['workclass'].unique()


# In[29]:


x['native.country'].unique()


# In[30]:


y['native.country'].unique()


# In[31]:


y.replace(['South','Hong'],['South korea','Hong kong'],inplace=True)


# In[32]:


x.replace(['South','Hong'],['South korea','Hong kong'],inplace=True)


# In[33]:


x['native.country'].unique()


# In[34]:


x['net_capital']=x['capital.gain']-x['capital.loss']
x.drop(['capital.gain','capital.loss'],1,inplace=True)


# In[35]:


y['net_capital']=y['capital.gain']-y['capital.loss']
y.drop(['capital.gain','capital.loss'],1,inplace=True)


# In[36]:


y.head()


# In[37]:


x.head()


# In[38]:



name = ['age','fnlwgt','education.num','net_capital','hours.per.week']
for c in name:
    sns.distplot(x[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[39]:



name = ['age','fnlwgt','education.num','net_capital','hours.per.week']
for c in name:
    sns.distplot(y[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[40]:


d = x.loc[:,['age','fnlwgt','education.num','net_capital','hours.per.week']]


# In[41]:


d1 = y.loc[:,['age','fnlwgt','education.num','net_capital','hours.per.week']]


# In[42]:


d.head()


# In[43]:


d1.head()


# In[44]:


from sklearn.preprocessing import Normalizer


# In[45]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d=pd.DataFrame(pt.fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[46]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d1=pd.DataFrame(pt.fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[47]:



d=pd.DataFrame(Normalizer().fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[48]:



d1=pd.DataFrame(Normalizer().fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[49]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d=pd.DataFrame(pt.fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[50]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d1=pd.DataFrame(pt.fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# quantile
# normalizer
# quantile

# In[51]:


name = ['age','fnlwgt','education.num','net_capital','hours.per.week']

for c in name:
    sns.distplot(d[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[52]:


name = ['age','fnlwgt','education.num','net_capital','hours.per.week']

for c in name:
    sns.distplot(d1[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[53]:


sns.heatmap(x.corr(),annot = True)


# In[54]:


sns.heatmap(y.corr(),annot = True)


# In[55]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for c in x.select_dtypes(['object']).columns:
    
        x[c]=le.fit_transform(x[c])
        


# In[56]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for c in y.select_dtypes(['object']).columns:
    
        y[c]=le.fit_transform(y[c])
        


# In[57]:


d.head()


# In[58]:


d1.head()


# In[59]:


x.drop(['age','fnlwgt','education.num','net_capital','hours.per.week'],1,inplace=True)


# In[60]:


y.drop(['age','fnlwgt','education.num','net_capital','hours.per.week'],1,inplace=True)


# In[61]:


x=pd.merge(x,d,left_index=True,right_index=True)


# In[62]:


y=pd.merge(y,d,left_index=True,right_index=True)


# In[63]:


x.head()


# In[64]:


x.shape


# In[65]:


y.head()


# In[66]:


#pca
#treebaseapproach
#rfe


# In[67]:


plt.figure(figsize=(20,10))
sns.heatmap(x.corr(),annot = True)


# In[68]:


plt.figure(figsize=(20,10))
sns.heatmap(y.corr(),annot = True)


# In[69]:


x_train = x.drop('income',1)
y_train = x['income']


# In[70]:


x_test = x.drop('income',1)
y_test = x['income']


# In[71]:


from sklearn.feature_selection import RFECV


# In[72]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


# In[73]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[74]:


import warnings
warnings.filterwarnings('ignore')


# In[75]:


rfe = RFECV(estimator = DecisionTreeClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))


plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[76]:


rfe = RFECV(estimator = RandomForestClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[77]:


rfe = RFECV(estimator = AdaBoostClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[78]:


rfe = RFECV(estimator = GradientBoostingClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[79]:


from sklearn.ensemble import RandomForestClassifier
 
# Feature importance values from Random Forests
rf = RandomForestClassifier(n_jobs=-1, random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = RandomForestClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[80]:




rf = AdaBoostClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = AdaBoostClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[81]:


rf = GradientBoostingClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = GradientBoostingClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[82]:


rf = xgb.XGBClassifier(random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = xgb.XGBClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# # one hot

# In[83]:


x['income']=le.fit_transform(x['income'])


# In[84]:


x.head()


# In[85]:


for c in x.select_dtypes(['object']).columns:
    cont = pd.get_dummies(x[c],prefix='Contract')
    x = pd.concat([x,cont],axis=1)
    x.drop(c,1,inplace=True)
    


# In[86]:


for c in y.select_dtypes(['object']).columns:
    cont = pd.get_dummies(y[c],prefix='Contract')
    y = pd.concat([y,cont],axis=1)
    y.drop(c,1,inplace=True)
    


# In[87]:


x.head()


# In[88]:


x.shape


# In[89]:


x_train = x.drop('income',1)
y_train = x['income']


# In[90]:


x_test = x.drop('income',1)
y_test = x['income']


# In[91]:


rfe = RFECV(estimator = DecisionTreeClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))


plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[92]:


rfe = RFECV(estimator = RandomForestClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[93]:


rfe = RFECV(estimator = AdaBoostClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[94]:


rfe = RFECV(estimator = GradientBoostingClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[95]:


from sklearn.ensemble import RandomForestClassifier
 
# Feature importance values from Random Forests
rf = RandomForestClassifier(n_jobs=-1, random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = RandomForestClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[96]:




rf = AdaBoostClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = AdaBoostClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[97]:


rf = GradientBoostingClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = GradientBoostingClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[98]:


rf = xgb.XGBClassifier(random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = xgb.XGBClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))

