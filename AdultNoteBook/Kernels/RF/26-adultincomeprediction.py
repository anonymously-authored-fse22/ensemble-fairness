#!/usr/bin/env python
# coding: utf-8

# **Problem Statement:-** 
# 
# The data here is for the "Census Income". This data is labeled with whether the person's yearly income is above or below $50K (and you are trying to model and predict this).
# 
# The data contains the following columns, along with a brief description of the data type (either "continuous" for numerical values, or a list of categorical values):
# 
# 1)age: continuous.
# 
# 2)workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# 
# 3)fnlwgt: continuous.
# 
# 4)education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 
# 5)education-num: continuous.
# 
# 6)marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# 
# 7)occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# 
# 8)relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# 
# 9)race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# 
# 10)gender: Female, Male.
# 
# 11)capital-gain: continuous.
# 
# 12)capital-loss: continuous.
# 
# 13)hours-per-week: continuous.
# 
# 14)native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 
# 15)Output adult.data contains one additional column for the label, which is >50K if the person's yearly income is greater than $50K, and otherwise <=50K.
# 
# DATA VISUALIZATION IS DONE BY:-
# 
# 1)Count plot
# 
# 2)Histogram
# 
# 3)Boxplot
# 
# 4)Heatmap
# 
# 5)Pairplot
# 
# Machine learning algorithm used:-
# 
# 1)Logistic Regression
# 
# 2)Decision Tree classifier
# 
# 3)Bagging Classifier
# 
# 4)Random Forest classifier
# 
# 5)SGD Classifier
# 
# 6)Gradient Boosting Classifier

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import read_csv
from pandas.plotting import scatter_matrix

from numpy import mean
from numpy import std

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier


from sklearn import model_selection
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# # Importing dataset 

# In[ ]:


df = pd.read_csv('../input/adult-census-income/adult.csv')
df.head()


# # Drop missing values

# In[ ]:


# drop rows with missing
df = df.dropna()


# In[ ]:


# summarize the shape of the dataset
print(df.shape)


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isnull().values.any()


# # Classifying on the basis of income 

# 0=more than 50k
# 
# 1=less than 50k

# In[ ]:


df['income']=LabelEncoder().fit_transform(df['income'])


# # count plot

# 1.Workclass

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('workclass',data=df,hue="income" )
plt.tight_layout()
plt.show()


# 2.Eduacation

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('education',data=df)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 3.Marital Status 

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('marital.status',data=df )
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 4.Occupation 

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('occupation',data=df )
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 5.sex

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('sex',data=df,hue="income")
plt.tight_layout()
plt.show()


# 6.Race

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('race',data=df )
plt.tight_layout()
plt.show()


# 7.Native Country

# In[ ]:


fig=plt.figure(figsize=(10,6))
sns.countplot('native.country',data=df.head(200) )
plt.tight_layout()
plt.show()


# # Box plot 

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(25,5))
box1=sns.boxplot(data=df["fnlwgt"],ax=ax[0],color='m')
ax[0].set_xlabel('fnlwgt')
box1=sns.boxplot(data=df["hours.per.week"],ax=ax[1],color='m')
ax[1].set_xlabel('hours.per.week')
box1=sns.boxplot(data=df["age"],ax=ax[2],color='m')
ax[2].set_xlabel('age')


# In[ ]:


sns.boxplot(x="age",y="sex",hue="income",data=df)


# # Heatmap 

# In[ ]:


#df1= df.corr()
corr = (df.corr())
plt.subplots(figsize=(9, 9))
sns.heatmap(corr, vmax=.8,annot=True,cmap="viridis", square=True);


# # Histograms of each features 

# In[ ]:


df1=df.drop(['income'],axis=1)
df1.hist (bins=10,figsize=(20,20))
plt.show ()


# # Pairplot 

# In[ ]:


sns.pairplot(data=df,kind='reg',size=5)


# In[ ]:


sns.pairplot(df,hue = 'income',vars = ['fnlwgt','hours.per.week','education.num'] )


# **Violinplot**

# In[ ]:


ax = sns.violinplot(x="education.num", y="income", data=df, palette="muted")


# In[ ]:


df=df.dropna()


# # classifying input and output 

# In[ ]:


df['sex'] = LabelEncoder().fit_transform(df['sex'])


# In[ ]:


x = df.drop(['income','workclass','education','marital.status','occupation','relationship','race','native.country'],axis=1)
y= df['income']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


cat_ix = x.select_dtypes(include=['object', 'bool']).columns 
num_ix = x.select_dtypes(include=['int64', 'float64']).columns 


# # Cross Validation 

# In[ ]:


seed=101
models = []
models.append(('RF',RandomForestClassifier()))
models.append(('SGDC',SGDClassifier()))
models.append (('CART',DecisionTreeClassifier()))
models.append (('BAG',BaggingClassifier()))
models.append(('LR',LogisticRegression()))
models.append(('GBM',GradientBoostingClassifier()))
results = []
names = []
for name, model in models:
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train,scoring='accuracy',cv=cv,n_jobs=-1)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)


# # 1.Logistic Regression 

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
logistic = LogisticRegression()
logistic.fit(x_train,y_train)
y_pred=logistic.predict(x_test)
print(classification_report(y_test,y_pred))
accuracy1=logistic.score(x_test,y_test)
print (accuracy1*100,'%')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot= True)


# # 2.Decision Tree Classifier 

# In[ ]:


des_class=DecisionTreeClassifier()
des_class.fit(x_train,y_train)
des_predict=des_class.predict(x_test)
print(classification_report(y_test,des_predict))
accuracy3=des_class.score(x_test,y_test)
print(accuracy3*100,'%')
cm = confusion_matrix(y_test, des_predict)
sns.heatmap(cm, annot= True)


# # 3.Bagging Classifier 

# In[ ]:


Bag=BaggingClassifier()
Bag.fit(x_train,y_train)
Bag_predict=Bag.predict(x_test)
print(classification_report(y_test,Bag_predict))
accuracy3=Bag.score(x_test,y_test)
print(accuracy3*100,'%')
cm = confusion_matrix(y_test, Bag_predict)
sns.heatmap(cm, annot= True)


# # 4.Random Forest classifier 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 
ran_class=RandomForestClassifier()
ran_class.fit(x_train,y_train)
ran_predict=ran_class.predict(x_test)
print(classification_report(y_test,ran_predict))
accuracy3=ran_class.score(x_test,y_test)
print(accuracy3*100,'%')
cm = confusion_matrix(y_test, ran_predict)
sns.heatmap(cm, annot= True)


# # 5.SGD Classifier

# In[ ]:


Sgdc=SGDClassifier()
Sgdc.fit(x_train,y_train)
Sgdc_predict=Sgdc.predict(x_test)
print(classification_report(y_test,Sgdc_predict))
accuracy3=Sgdc.score(x_test,y_test)
print(accuracy3*100,'%')
cm = confusion_matrix(y_test, Sgdc_predict)
sns.heatmap(cm, annot= True)


# # 6.Gradient Boosting Classifier 

# In[ ]:


gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
gbc_predict=gbc.predict(x_test)
print(classification_report(y_test,gbc_predict))
accuracy3=gbc.score(x_test,y_test)
print(accuracy3*100,'%')
cm = confusion_matrix(y_test, gbc_predict)
sns.heatmap(cm, annot= True)

