#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
print('Imported')


# # Step 1: Select Data

# In[2]:


path = '../input/adult.csv'
data = pd.read_csv(path, na_values=['?']);
data.shape


# # Step 2: Preprocess Data

# In[3]:


data.info()


# In[4]:


data.head(5)


# When we see here we can see that we have missing values in fetaures:
# 1. workclass 
# 2. occupation
# 3. native_country

# We can fill these missing values:
# All these 3 featues are categorical varibales so thier missing value can be filled using mode 

# In[6]:


#fill missing values
data['workclass'] = data['workclass'].fillna(data['workclass'].mode()[0])
data['occupation'] = data['occupation'].fillna(data['occupation'].mode()[0])
data['native.country'] = data['native.country'].fillna(data['native.country'].mode()[0])


# In[7]:


data.head()


# In[8]:


data.info()


# We have filled the missing values

# # EDA on our dataset

# In[9]:


data['income'].value_counts()


# In[10]:


sns.set_style('whitegrid');
sns.pairplot(data, hue = 'income', size = 10)
plt.show()


# ## We have to look for a graph which is well seprated 

# In[14]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'age', 'workclass')   .add_legend();
plt.title('Age vs Workclass');
plt.show();


# **Important Insights**
# 
# 1.1 After visualizing this graph plot we can observe that the people aged between 24 and 75 earns >50K 
# 
# 1.2 As a obvious reason who Never-worked and Without-Pay will never earn >50K
# 
# 1.3 People below age 24 and 75 are those who earns <=50K
# 

# In[15]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'age', 'hours.per.week')   .add_legend();
plt.title('Age vs Hours_per_week');
plt.show();


# **Important Insights**
# 
# 1.4 Here we can see that majority of people who earns >50K are the persons works >30 hours per week

# In[16]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'age', 'fnlwgt')   .add_legend();
plt.title('Age vs fnlwgt');
plt.show();


# In[17]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'age', 'education.num')   .add_legend();
plt.title('Age vs education.num');
plt.show();


# In[18]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'age', 'capital.gain')   .add_legend();
plt.title('Age vs capital.gain');
plt.show();


# In[19]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'age', 'capital.loss')   .add_legend();
plt.title('Age vs Capital Loss');
plt.show();


# In[20]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)   .map(plt.scatter, 'capital.loss', 'capital.gain')   .add_legend();
plt.title('Capital Loss vs Capital Gain');
plt.show();


# # Step 3: Transform Data

# As we know we have categorical attributes in our dataset, so we need transfrom them into numerical values it can be feed to our model

# In[21]:


# We already know from data.info() that our dataset has TWO variables types: int64, Object
# So variable with datatype Object are categorical variables/fetaures
categorical_var = data.select_dtypes(include=['object']).columns
print(categorical_var)
print(len(categorical_var))


# We have to Trasform these 9 variable to numerical values to convert them form text based values

# In[22]:


# Before tranforming the fetures we need to seprate them as: variables and target varibles
# here 'income' is our target variable we need to seperate from our main dataset beofore transformation
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]


# In[23]:


X.head()


# In[24]:


y.value_counts()


# In[25]:


# we can use pd.dummies for handling categorical data
X = pd.get_dummies(X)


# In[26]:


X.head()


# In[27]:


y.value_counts()


# In[28]:


data['income'].value_counts()


# We have 2 classes in our target variables:
# y = (<=50K, >50K)

# In[29]:


from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score


# In[30]:


# split the data set into train and test
X_1, X_test, y_1, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.2)


for i in range(1,30,2):
    # instantiate learning model (k = 30)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model on crossvalidation train
    knn.fit(X_tr, y_tr)

    # predict the response on the crossvalidation train
    pred = knn.predict(X_cv)

    # evaluate CV accuracy
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))
    


# In[31]:


k = 17
knn = KNeighborsClassifier(k)
knn.fit(X_tr,y_tr)
pred = knn.predict(X_test)
knnacc = accuracy_score(y_test, pred, normalize=True) * float(100)
print('\n****Test accuracy for k =',k,' is %d%%' % (knnacc))


# In[32]:


#Using Gradienboosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier().fit(X_tr,y_tr)

pred = gbc.predict(X_test)

gbcacc = gbc.score(X_test, y_test)

print('GBC: ', gbcacc * 100, '%')


# In[34]:


# using Decision tree 
from sklearn import tree
dtc = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
dtc.fit(X_tr,y_tr)

pred = dtc.predict(X_test)

dtcacc = dtc.score(X_test, y_test)

print('Decision Tree classifier:', dtcacc * 100,'%')


# In[35]:


# Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=12, random_state=0)
rfc.fit(X_tr, y_tr)

pred = rfc.predict(X_test)

rfcacc = rfc.score(X_test, y_test)

print('Random Forest:', rfcacc * 100,'%')


# In[36]:


accuracyScore = [knnacc, gbcacc * 100, dtcacc * 100, rfcacc * 100]
algoName = ['KNN', 'GBC', 'DT', 'RF']


# In[37]:


plt.scatter(algoName, accuracyScore)
plt.grid()
plt.title('Algorithm Accuracy Comparision')
plt.xlabel('Algorithm')
plt.ylabel('Score in %')
plt.show()


# Here we concluded that we achived highest accuracy with GBC.
# Still lots of improvement can be by hyper tuning the parameters of classifier and data preprocessing technic
