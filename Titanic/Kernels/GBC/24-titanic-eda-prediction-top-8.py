#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 50px; color : #532e1c ; font-family : 'Comic Sans MS'; text-align : center; background-color : #bedcfa; border-radius: 5px 5px;"><strong>Titanic EDA and Prediction</strong></p>

# <img style="float: center;  border:5px solid #ffb037; width:100%" src = https://sn56.scholastic.com/content/dam/classroom-magazines/sn56/issues/2018-19/020419/the-titanic-sails-again/SN56020919_Titanic-Hero.jpg> 

# <a id = '0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Table of Contents</strong></p> 
# 
# * [Data Description](#1.0)
# * [EDA](#2.0)
#     * [Survived Column](#2.1)
#     * [Pclass Column](#2.2)
#     * [Name Column](#2.3)
#     * [Sex Column](#2.4)
#     * [Age Column](#2.5)
#     * [Fare Column](#2.6)
#     * [SibSp Column](#2.7)
#     * [Parch Column](#2.8)
#     * [Ticket Column](#2.9)
#     * [Embarked Column](#2.10)
#     
# * [Findings From EDA](#3.0)
# * [Data Preprocessing](#4.0)
# * [Models](#5.0)
#     * [Logistic Regression](#5.1)
#     * [Knn](#5.2)
#     * [Decision Tree Classifier](#5.3)
#     * [Random Forest Classifier](#5.4)
#     * [Ada Boost Classifier](#5.5)
#     * [Gradient Boosting Classifier](#5.6)
#     * [Stochastic Gradient Boosting (SGB)](#5.7)
#     * [XgBoost](#5.8)
#     * [Cat Boost Classifier](#5.9)
#     * [Extra Trees Classifier](#5.10)
#     * [LGBM Classifier](#5.11)
#     * [Voting Classifier](#5.12)
# 
# * [Models Comparison](#6.0)
# 

# <a id = '1.0'></a>
# <p style = "font-size : 30px; color : #4e8d7c ; font-family : 'Comic Sans MS';  "><strong>Data Description :-</strong></p>
# 
# <ul>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Survival : 0 = No, 1 = Yes</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>pclass(Ticket Class) : 1 = 1st, 2 = 2nd, 3 = 3rd</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Sex(Gender) : Male, Female</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Age : Age in years</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>SibSp : Number of siblings/spouses abroad the titanic</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Parch : Number of parents/children abrod the titanic</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Ticket : Ticket Number</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Fare : Passenger fare</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Cabin : Cabin Number</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Embarked : Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southampton</strong></li>
# </ul>

# In[ ]:


# necessary imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.var()


# In[ ]:


train_df.info()


# In[ ]:


# Checking for null values

train_df.isna().sum()


# <a id = '2.0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Exploratory Data Analysis (EDA)</strong></p> 

# In[ ]:


# visualizing null values

import missingno as msno

msno.bar(train_df)
plt.show()


# In[ ]:


# heatmap

plt.figure(figsize = (18, 8))

corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(corr, mask = mask, annot = True, fmt = '.2f', linewidths = 1, annot_kws = {'size' : 15})
plt.show()


# <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>Heatmap is not useful in case of categorical variables, so we will analyse each column to check how each column is contributing in prediction.</strong></p> 
# 

# <a id = '2.1'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Survived Column</strong></p> 

# In[ ]:


plt.figure(figsize = (12, 7))

sns.countplot(y = 'Survived', data = train_df)
plt.show()


# In[ ]:


values = train_df['Survived'].value_counts()
labels = ['Not Survived', 'Survived']

fig, ax = plt.subplots(figsize = (5, 5), dpi = 100)
explode = (0, 0.06)

patches, texts, autotexts = ax.pie(values, labels = labels, autopct = '%1.2f%%', shadow = True,
                                   startangle = 90, explode = explode)

plt.setp(texts, color = 'grey')
plt.setp(autotexts, size = 12, color = 'white')
autotexts[1].set_color('black')
plt.show()


# <a id = '2.2'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Pclass Column</strong></p> 

# In[ ]:


train_df.Pclass.value_counts()


# In[ ]:


train_df.groupby(['Pclass', 'Survived'])['Survived'].count()


# In[ ]:


plt.figure(figsize = (16, 8))

sns.countplot('Pclass', hue = 'Survived', data = train_df)
plt.show()


# In[ ]:


values = train_df['Pclass'].value_counts()
labels = ['Third Class', 'Second Class', 'First Class']
explode = (0, 0, 0.08)

fig, ax = plt.subplots(figsize = (5, 6), dpi = 100)
patches, texts, autotexts = ax.pie(values, labels = labels, autopct = '%1.2f%%', shadow = True,
                                   startangle = 90, explode = explode)

plt.setp(texts, color = 'grey')
plt.setp(autotexts, size = 13, color = 'white')
autotexts[2].set_color('black')
plt.show()


# In[ ]:


sns.catplot('Pclass', 'Survived', kind = 'point', data = train_df, height = 6, aspect = 2)
plt.show()


# <a id = '2.3'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Name Column</strong></p> 

# In[ ]:


train_df.Name.value_counts()


# In[ ]:


len(train_df.Name.unique()), train_df.shape


# <a id = '2.4'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Sex Column</strong></p> 

# In[ ]:


train_df.Sex.value_counts()


# In[ ]:


train_df.groupby(['Sex', 'Survived'])['Survived'].count()


# In[ ]:


plt.figure(figsize = (16, 7))

sns.countplot('Sex', hue = 'Survived', data = train_df)
plt.show()


# In[ ]:


sns.catplot(x = 'Sex', y = 'Survived', data = train_df, kind = 'bar', col = 'Pclass')
plt.show()


# In[ ]:


sns.catplot(x = 'Sex', y = 'Survived', data = train_df, kind = 'point', height = 6, aspect =2)
plt.show()


# In[ ]:


plt.figure(figsize = (15, 6))

sns.catplot(x = 'Pclass', y = 'Survived', kind = 'point', data = train_df, hue = 'Sex', height = 6, aspect = 2)
plt.show()


# <a id = '2.5'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Age Column</strong></p> 

# In[ ]:


plt.figure(figsize = (15, 6))
plt.style.use('ggplot')

sns.distplot(train_df['Age'])
plt.show()


# In[ ]:


sns.catplot(x = 'Sex', y = 'Age', kind = 'box', data = train_df, height = 5, aspect = 2)
plt.show()


# In[ ]:


sns.catplot(x = 'Sex', y = 'Age', kind = 'box', data = train_df, col = 'Pclass')
plt.show()


# <a id = '2.6'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Fare Column</strong></p> 

# In[ ]:


plt.figure(figsize = (14, 6))

plt.hist(train_df.Fare, bins = 60, color = 'orange')
plt.xlabel('Fare')
plt.show()


# <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>We can see that lot of zero values are there in Fare column so we will replace zero values with mean value of Fare column later.</strong></p> 

# In[ ]:


sns.catplot(x = 'Sex', y = 'Fare', data = train_df, kind = 'box', col = 'Pclass')
plt.show()


# <a id = '2.7'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>SibSp Column</strong></p> 

# In[ ]:


train_df['SibSp'].value_counts()


# In[ ]:


plt.figure(figsize = (16, 5))

sns.countplot(x = 'SibSp', data = train_df, hue = 'Survived')
plt.show()


# In[ ]:


sns.catplot(x = 'SibSp', y = 'Survived', kind = 'bar', data = train_df, height = 5, aspect =2)
plt.show()


# In[ ]:


sns.catplot(x = 'SibSp', y = 'Survived', kind = 'bar', hue = 'Sex', data = train_df, height = 6, aspect = 2)
plt.show()


# In[ ]:


sns.catplot(x = 'SibSp',  y = 'Survived', kind = 'bar', col = 'Sex', data = train_df)
plt.show()


# In[ ]:


sns.catplot(x = 'SibSp', y = 'Survived', col = 'Pclass', kind = 'bar', data = train_df)
plt.show()


# In[ ]:


sns.catplot(x = 'SibSp', y = 'Survived', kind = 'point', hue = 'Sex', data = train_df, height = 6, aspect = 2)
plt.show()


# <a id = '2.8'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Parch Column</strong></p> 

# In[ ]:


train_df.Parch.value_counts()


# In[ ]:


sns.catplot(x = 'Parch', y = 'Survived', data = train_df, hue = 'Sex', kind = 'bar', height = 6, aspect = 2)
plt.show()


# <a id = '2.9'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Ticket Column</strong></p> 

# In[ ]:


train_df.Ticket.value_counts()


# In[ ]:


len(train_df.Ticket.unique())


# <a id = '2.10'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Embarked Column</strong></p> 

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


plt.figure(figsize = (14, 6))

sns.countplot('Embarked', hue = 'Survived', data = train_df)
plt.show()


# In[ ]:


sns.catplot(x = 'Embarked', y = 'Survived', kind = 'bar', data = train_df, col = 'Sex')
plt.show()


# <a id = '3.0'></a>
# <p style = "font-size : 30px; color : #4e8d7c ; font-family : 'Comic Sans MS';"><strong>Findings From EDA :-</strong></p>
# 
# <ul>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Females Survived more than Males.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Passengers Travelling in Higher Class Survived More than Passengers travelling in Lower Class.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Name column is having all unique values so this column is not suitable for prediction, we have to drop it.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>In First Class Females were more than Males, that's why Fare of Females Passengers were high.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Survival Rate is higher for those who were travelling with siblings or spouses.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Passengers travelling with parents or children have higher survival rate.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Ticket column is not useful and does not have an impact on survival.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Cabin column have a lot of null values , it will be better to drop this column.</strong></li>
#     <li style = "color : #03506f; font-size : 18px; font-family : 'Comic Sans MS';"><strong>Passengers travelling from Cherbourg port survived more than passengers travelling from other two ports.</strong></li>
# </ul>

# <a id = '4.0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Data Pre-Processing</strong></p> 

# In[ ]:


# dropping useless columns

train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.isna().sum()


# In[ ]:


# replacing Zero values of "Fare" column with mean of column

train_df['Fare'] = train_df['Fare'].replace(0, train_df['Fare'].mean())


# In[ ]:


# filling null values of "Age" column with mean value of the column

train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)


# In[ ]:


# filling null values of "Embarked" column with mode value of the column

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)


# In[ ]:


# checking for null values after filling null values

train_df.isna().sum()


# In[ ]:


train_df.head()


# In[ ]:


train_df['Sex'] = train_df['Sex'].apply(lambda val: 1 if val == 'male' else 0)


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.var()


# <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>Variance in "Fare" column is very high so we have to normalize these columns.</strong></p> 

# In[ ]:


train_df['Age'] = np.log(train_df['Age'])
train_df['Fare'] = np.log(train_df['Fare'])


# In[ ]:


train_df.head()


# <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>Now training data looks much better let's take a look at test data.</strong></p> 

# In[ ]:


test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_df.head()


# <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>Performing same steps on test data.</strong></p> 

# In[ ]:


# dropping useless columns

test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[ ]:


# replacing Zero values of "Fare" column with mean of column

test_df['Fare'] = test_df['Fare'].replace(0, test_df['Fare'].mean())


# In[ ]:


# filling null values of "Age" column with mean value of the column

test_df['Age'].fillna(test_df['Age'].mean(), inplace = True)


# In[ ]:


# filling null values of "Embarked" column with mode value of the column

test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)


# In[ ]:


test_df.isna().sum()


# In[ ]:


# filling null values of "Fare" column with mean value of the column

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace = True)


# In[ ]:


test_df['Sex'] = test_df['Sex'].apply(lambda val: 1 if val == 'male' else 0)


# In[ ]:


test_df['Embarked'] = test_df['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})


# In[ ]:


test_df.head()


# In[ ]:


test_df['Age'] = np.log(test_df['Age'])
test_df['Fare'] = np.log(test_df['Fare'])


# In[ ]:


test_df.var()


# In[ ]:


test_df.isna().any()


# In[ ]:


test_df.head()


# <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>Now both training and test data is cleaned and preprocessed, let's start with model building.</strong></p> 

# In[ ]:


# creating X and y

X = train_df.drop('Survived', axis = 1)
y = train_df['Survived']


# In[ ]:


# splitting data intp training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# <a id = '5.0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong> Models</strong></p> 

# <a id = '5.1'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Logistic Regression</strong></p> 

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of logistic regression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lr_acc = accuracy_score(y_test, lr.predict(X_test))

print(f"Training Accuracy of Logistic Regression is {accuracy_score(y_train, lr.predict(X_train))}")
print(f"Test Accuracy of Logistic Regression is {lr_acc}")

print(f"Confusion Matrix :- \n {confusion_matrix(y_test, lr.predict(X_test))}")
print(f"Classofocation Report : -\n {classification_report(y_test, lr.predict(X_test))}")


# In[ ]:


# hyper parameter tuning of logistic regression

from sklearn.model_selection import GridSearchCV

grid_param = {
    'penalty': ['l1', 'l2'],
    'C' : [0.001, 0.01, 0.1, 0.005, 0.5, 1, 10]
}

grid_search_lr = GridSearchCV(lr, grid_param, cv = 5, n_jobs = -1, verbose = 1)
grid_search_lr.fit(X_train, y_train)


# In[ ]:


# best parameters and best score

print(grid_search_lr.best_params_)
print(grid_search_lr.best_score_)


# In[ ]:


# best estimator

lr = grid_search_lr.best_estimator_

# accuracy score, confusion matrix and classification report of logistic regression

lr_acc = accuracy_score(y_test, lr.predict(X_test))

print(f"Training Accuracy of Logistic Regression is {accuracy_score(y_train, lr.predict(X_train))}")
print(f"Test Accuracy of Logistic Regression is {lr_acc}")

print(f"Confusion Matrix :- \n {confusion_matrix(y_test, lr.predict(X_test))}")
print(f"Classofocation Report : -\n {classification_report(y_test, lr.predict(X_test))}")


# <a id = '5.2'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>KNN</strong></p> 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of knn

knn_acc = accuracy_score(y_test, knn.predict(X_test))

print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
print(f"Test Accuracy of KNN is {knn_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")


# <a id = '5.3'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Decision Tree Classifier</strong></p> 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


# In[ ]:


# hyper parameter tuning of decision tree 

grid_param = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'splitter' : ['best', 'random'],
    'min_samples_leaf' : [1, 2, 3, 5, 7],
    'min_samples_split' : [1, 2, 3, 5, 7],
    'max_features' : ['auto', 'sqrt', 'log2']
}

grid_search_dtc = GridSearchCV(dtc, grid_param, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dtc.fit(X_train, y_train)


# In[ ]:


# best parameters and best score

print(grid_search_dtc.best_params_)
print(grid_search_dtc.best_score_)


# In[ ]:


# best estimator

dtc = grid_search_dtc.best_estimator_

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


# <a id = '5.4'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Random Forest Classifier</strong></p>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of random forest

rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")


# <a id = '5.5'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Ada Boost Classifier</strong></p>

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of ada boost

ada_acc = accuracy_score(y_test, ada.predict(X_test))

print(f"Training Accuracy of Ada Boost Classifier is {accuracy_score(y_train, ada.predict(X_train))}")
print(f"Test Accuracy of Ada Boost Classifier is {ada_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, ada.predict(X_test))}")


# In[ ]:


# hyper parameter tuning ada boost

grid_param = {
    'n_estimators' : [100, 120, 150, 180, 200],
    'learning_rate' : [0.01, 0.1, 1, 10],
    'algorithm' : ['SAMME', 'SAMME.R']
}

grid_search_ada = GridSearchCV(ada, grid_param, cv = 5, n_jobs = -1, verbose = 1)
grid_search_ada.fit(X_train, y_train)


# In[ ]:


# best parameter and best score

print(grid_search_ada.best_params_)
print(grid_search_ada.best_score_)


# In[ ]:


ada = grid_search_ada.best_estimator_

# accuracy score, confusion matrix and classification report of ada boost

ada_acc = accuracy_score(y_test, ada.predict(X_test))

print(f"Training Accuracy of Ada Boost Classifier is {accuracy_score(y_train, ada.predict(X_train))}")
print(f"Test Accuracy of Ada Boost Classifier is {ada_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, ada.predict(X_test))}")


# <a id = '5.6'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Gradient Boosting Classifier</strong></p>

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of gradient boosting classifier

gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")


# <a id = '5.7'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Stochastic Gradient Boosting (SGB)</strong></p>

# In[ ]:


sgb = GradientBoostingClassifier(subsample = 0.90, max_features = 0.70)
sgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier

sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")


# <a id = '5.8'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>XgBoost</strong></p>

# In[ ]:


from xgboost import XGBClassifier

xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
xgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of xgboost

xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

print(f"Training Accuracy of XgBoost is {accuracy_score(y_train, xgb.predict(X_train))}")
print(f"Test Accuracy of XgBoost is {xgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, xgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, xgb.predict(X_test))}")


# <a id = '5.9'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Cat Boost Classifier</strong></p>

# In[ ]:


from catboost import CatBoostClassifier

cat = CatBoostClassifier(iterations=10)
cat.fit(X_train, y_train)


# In[ ]:


# accuracy score, confusion matrix and classification report of cat boost

cat_acc = accuracy_score(y_test, cat.predict(X_test))

print(f"Training Accuracy of Cat Boost Classifier is {accuracy_score(y_train, cat.predict(X_train))}")
print(f"Test Accuracy of Cat Boost Classifier is {cat_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, cat.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, cat.predict(X_test))}")


# <a id = '5.10'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Extra Trees Classifier</strong></p>

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")


# <a id = '5.11'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>LGBM Classifier</strong></p>

# In[ ]:


from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate = 1)
lgbm.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of lgbm classifier

lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
print(f"Test Accuracy of LGBM Classifier is {lgbm_acc} \n")

print(f"{confusion_matrix(y_test, lgbm.predict(X_test))}\n")
print(classification_report(y_test, lgbm.predict(X_test)))


# <a id = '5.12'></a>
# <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Voting Classifier</strong></p>

# In[ ]:


from sklearn.ensemble import VotingClassifier

classifiers = [('Gradient Boosting Classifier', gb), ('Stochastic Gradient Boosting', sgb),  ('Cat Boost Classifier', cat), 
               ('XGboost', xgb),  ('Decision Tree', dtc), ('Extra Tree', etc), ('Light Gradient', lgbm),
               ('Random Forest', rd_clf), ('Ada Boost', ada), ('Logistic', lr)]
vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)


# In[ ]:


# accuracy score, confusion matrix and classification report of voting classifier

vc_acc = accuracy_score(y_test, vc.predict(X_test))

print(f"Training Accuracy of Voting Classifier is {accuracy_score(y_train, vc.predict(X_train))}")
print(f"Test Accuracy of Voting Classifier is {vc_acc} \n")

print(f"{confusion_matrix(y_test, vc.predict(X_test))}\n")
print(classification_report(y_test, vc.predict(X_test)))


# <a id = '6.0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong> Models Comparison</strong></p> 

# In[ ]:


models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
             'Gradient Boosting Classifier', 'Stochastic Gradient Boosting', 'XgBoost', 'Cat Boost', 'Extra Trees Classifier', 'Voting Classifier'],
    'Score' : [lr_acc, knn_acc, dtc_acc, rd_clf_acc, ada_acc, gb_acc, sgb_acc, xgb_acc, cat_acc, etc_acc, vc_acc]
})


models.sort_values(by = 'Score', ascending = False)


# In[ ]:


plt.figure(figsize = (15, 10))

sns.barplot(x = 'Score', y = 'Model', data = models)
plt.show()


# In[ ]:


final_prediction = sgb.predict(test_df)
prediction = pd.DataFrame(final_prediction)
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = prediction
submission.to_csv('Submission.csv', index = False)


# <p style = "font-size : 25px; color : #f55c47 ; font-family : 'Comic Sans MS'; "><strong>If you like my work, please do Upvote.</strong></p> 
