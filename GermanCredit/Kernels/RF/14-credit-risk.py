#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Analysis and prediction

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import confusion_matrix, precision_recall_curve, fbeta_score
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

from xgboost import XGBClassifier

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

pd.set_option('display.max_columns', 100)


# In[ ]:


df = df_credit = pd.read_csv("../input/german-credit-risk-labelled/credit_risk_labelled.csv",index_col=0)


# ## Filling missing data

# In[ ]:


df.isna().sum()


# In[ ]:


print(df['Saving accounts'].unique())
df['Saving accounts'][df['Saving accounts'].isna()] = 'None'
print(df['Saving accounts'].unique())


# In[ ]:


print(df['Checking account'].unique())
df['Checking account'][df['Checking account'].isna()] = 'None'
print(df['Checking account'].unique())


# In[ ]:


df.isna().sum()


# In[ ]:


df.nunique()


# ## Data Analysis

# In[ ]:


sns.countplot(x=df['Risk'])
print('proportion of good credit: ', sum(sum([df['Risk']=='good']))/len(df))
print('proportion of bad credit: ', sum(sum([df['Risk']=='bad']))/len(df))


# 70% of the credits in the dataset are labelled 'bad'

# In[ ]:


plt.figure(figsize = (12, 7))
sns.distplot(df['Age'][df['Risk']=='bad'], label = 'risk=bad')
sns.distplot(df['Age'][df['Risk']=='good'], label = 'risk=good')
sns.distplot(df['Age'], label='all')
plt.legend()


# In[ ]:


g = sns.FacetGrid(df, col="Risk", height=5, aspect=1)
g.map(sns.distplot, "Age")


#  Younger people appear to be more at risk for bad credit than older people

# In[ ]:


plt.figure(figsize = (10, 5))
sns.countplot(data=df, x='Job', hue='Risk', palette=[(.4, .8, .2), (.9, .1, .3)])
categories = df['Job'].unique()
categories.sort()

for category in categories:
    fail = len((df)[(df['Job']==category) & (df['Risk']=='bad')]) / len((df)[(df['Job']==category)])
    print('proportion of fail for category {}: {}%'.format(category, round(fail*100, 1)))


# In[ ]:


plt.figure(figsize = (10, 5))
sns.boxplot(data=df, x='Job', y='Credit amount', hue='Risk', palette=[(.4, .8, .2), (.9, .1, .3)])


# Most of the credits are made by people in job category 2, but the proportion of bad credits is similar for each category  
# Nonetheless, the highest credits are made by job category 3, and the largest variation is made by category 0   
# People in category 3 fail more often

# In[ ]:


plt.figure(figsize = (10, 5))
sns.countplot(x='Housing', hue='Risk', data=df, palette=[(.4, .8, .2), (.9, .1, .3)])

categories = df['Housing'].unique()
for category in categories:
    fail = len((df)[(df['Housing']==category) & (df['Risk']=='bad')]) / len((df)[(df['Housing']==category)])
    print('proportion of fail for category {}: {}%'.format(category, round(fail*100, 1)))


# In[ ]:


plt.figure(figsize = (10, 5))
sns.boxplot(x='Housing', hue='Risk', y='Credit amount', data=df, palette=[(.4, .8, .2), (.9, .1, .3)])


# People owning their own house tend to make more credits, and have a lower proportion of fail  
# Those who are free housing make larger credits

# In[ ]:


plt.figure(figsize = (10, 5))
sns.countplot(x='Saving accounts', hue='Risk', data=df, palette=[(.4, .8, .2), (.9, .1, .3)])

categories = df['Saving accounts'].unique()
for category in categories:
    fail = len((df)[(df['Saving accounts']==category) & (df['Risk']=='bad')]) / len((df)[(df['Saving accounts']==category)])
    print('proportion of fail for category {}: {}%'.format(category, round(fail*100, 1)))


# In[ ]:


plt.figure(figsize = (10, 5))
sns.boxplot(x='Saving accounts', hue='Risk', y='Credit amount', data=df, palette=[(.4, .8, .2), (.9, .1, .3)])


# People with little saving account make more credits than other categories, with a higher proportion a failing.  
# Poeple with moderate saving fail quite often as well.  
# Nonetheless, each category makes credit of similar amount.  
# Rich people have a larger variation, meaning they are more susceptible to borrow large amounts

# In[ ]:


plt.figure(figsize = (13, 5))
sns.countplot(data=df, x='Purpose', hue='Risk', palette=[(.4, .8, .2), (.9, .1, .3)])

categories = df['Purpose'].unique()
for category in categories:
    fail = len((df)[(df['Purpose']==category) & (df['Risk']=='bad')]) / len((df)[(df['Purpose']==category)])
    print('proportion of fail for category {}: {}%'.format(category, round(fail*100, 1)))


# In[ ]:


plt.figure(figsize = (13, 5))
sns.boxplot(data=df, x='Purpose', hue='Risk', y='Credit amount', palette=[(.4, .8, .2), (.9, .1, .3)])


# Very interesting plot, as credits made for vacation have the largest proportion of failing. 
# Business is second in raking of failing purpose  
# People tend to make more credits to buy radio/TV and cars.  
# Every category is more susceptible to lead to a failing credit than to success...

# In[ ]:


plt.figure(figsize = (12, 7))
sns.distplot(df['Credit amount'][df['Risk']=='bad'], label = 'risk=bad')
sns.distplot(df['Credit amount'][df['Risk']=='good'], label = 'risk=good')
plt.legend()


# High credit amounts fail more often than low credit. 

# In[ ]:


plt.figure(figsize = (12, 7))
sns.distplot(df['Duration'][df['Risk']=='bad'], label = 'risk=bad')
sns.distplot(df['Duration'][df['Risk']=='good'], label = 'risk=good')
plt.legend()


# Credits engaged on a long duration happen to fail more than on short duration, which follows the previous plot: larger credits lead to longer duration, hence failing more ofter

# In[ ]:


plt.figure(figsize = (13, 5))
sns.countplot(data=df, x='Sex', hue='Risk', palette=[(.4, .8, .2), (.9, .1, .3)])

categories = df['Sex'].unique()
for category in categories:
    fail = len((df)[(df['Sex']==category) & (df['Risk']=='bad')]) / len((df)[(df['Sex']==category)])
    print('proportion of fail for category {}: {}%'.format(category, round(fail*100, 1)))


# In[ ]:


plt.figure(figsize = (13, 5))
sns.barplot(data=df, x='Sex', hue='Risk', y='Credit amount', palette=[(.4, .8, .2), (.9, .1, .3)])


# Males engage more credits than females, but they fail less often in proportion than females  
# Males also tend to borrow larger amounts, and also fail on the largest ones.  
# The variation of amounts is similar for males and females

# In[ ]:


df_inter=pd.DataFrame()
categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']

for col in categorical_features:
    dummies = pd.get_dummies(df[col],prefix=col)
    df_inter = pd.concat([df_inter, dummies], axis=1)
    
df = pd.concat([df, df_inter], axis=1)
df = df.drop(categorical_features, axis=1)


    
continuous_features = ['Age', 'Credit amount', 'Duration']
for col in continuous_features:
    column = np.array(df[col])
    column = column.reshape(len(column), 1)
    sc = StandardScaler()
    sc.fit(column)
    df[col] = sc.transform(column)
    
    

y = df['Risk_good']
X=df.drop(['Risk_good', 'Risk_bad'], axis=1)


# In[ ]:


y = df['Risk_bad']


# In[ ]:


df.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)


# ### Find best parameters for Random Forest

# In[ ]:


#Seting the Hyper Parameters
param_grid = {"max_depth": [3,5, 7, 10,None],
              "n_estimators":[3,5,10,25,50],
              "max_features": [4,7,15,20]}

#Creating the classifier
model = RandomForestClassifier(random_state=2)

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)


# In[ ]:


print(grid_search.best_score_)
print(grid_search.best_params_)


# Prediction with RF

# In[ ]:


rf = RandomForestClassifier(max_depth=None, max_features=20, n_estimators=5, random_state=0)
rf.fit(X_train, y_train)


# In[ ]:


prediction = rf.predict(X_test)


# In[ ]:


#Testing the model 
#Predicting using our  model
y_pred = rf.predict(X_test)

# Validation of the results
print(accuracy_score(y_test,prediction))
print("\n")
print(confusion_matrix(y_test, prediction))
print("\n")
print(fbeta_score(y_test, prediction, beta=2))
print("\n")
print(classification_report(y_test, prediction))


# ### Test with XGBoost

# In[ ]:


XGBModel = XGBClassifier()
XGBModel.fit(X_train, y_train , verbose=1)

#Testing the model 
#Predicting using our  model
XGBpredictions = XGBModel.predict(X_test)


# In[ ]:


# Validation of the results
print(accuracy_score(y_test,XGBpredictions))
print("\n")
print(confusion_matrix(y_test, XGBpredictions))
print("\n")
print(fbeta_score(y_test, XGBpredictions, beta=2))
print("\n")
print(classification_report(y_test, XGBpredictions))


# ### Gaussian Naive Bayes model

# In[ ]:


GNBModel = GaussianNB()
GNBModel.fit(X_train, y_train)

#Testing the model 
#Predicting using our  model
GNBpredictions = GNBModel.predict(X_test)


# In[ ]:


# Validation of the results
print(accuracy_score(y_test,GNBpredictions))
print("\n")
print(confusion_matrix(y_test, GNBpredictions))
print("\n")
print(fbeta_score(y_test, GNBpredictions, beta=2))
print("\n")
print(classification_report(y_test, GNBpredictions))


# ## Neural Network Model

# In[ ]:


y_test_categorical = to_categorical(
    y_test, num_classes=2, dtype='float32')
y_train_categorical = to_categorical(
    y_train, num_classes=2, dtype='float32')


# In[ ]:


def nn_model(learning_rate):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(2, kernel_initializer='normal',activation='sigmoid'))

    # Compile the network :
    optimizer = Adam(learning_rate=1e-5)
    NN_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    NN_model.summary()
    return NN_model


# In[ ]:


NN_model = nn_model(1e-4)
nb_epochs = 200
NN_model.fit(X_train, y_train_categorical, epochs=nb_epochs, batch_size=32)


# In[ ]:


NNpredictions = NN_model.predict(X_test)

NN_prediction = list()
for i in range(len(NNpredictions)):
    NN_prediction.append(np.argmax(NNpredictions[i]))


# In[ ]:


# Validation of the results
print(accuracy_score(y_test, NN_prediction))
print("\n")
print(confusion_matrix(y_test, NN_prediction))
print("\n")
print(fbeta_score(y_test, NN_prediction, beta=2))
print("\n")
print(classification_report(y_test, NN_prediction))


# The Neural Network is the best model found so far.. reaching accuracy of 74%.  

# In[ ]:


# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, NNpredictions[:, 1])

lr_auc = roc_auc_score(y_test, NNpredictions[:, 1])

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--', label='No Skill: ROC AUC=%.3f' % (0.5))
plt.plot(fpr, tpr, label='Logistic: ROC AUC=%.3f' % (lr_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:


NN_model = nn_model(1e-4)
nb_epochs = 200
NN_model.fit(X_train, y_train_categorical, epochs=nb_epochs, batch_size=32)

optimizer = Adam(learning_rate=5e-5)
NN_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
nb_epochs = 200
NN_model.fit(X_train, y_train_categorical, epochs=nb_epochs, batch_size=32)


# In[ ]:


NNpredictions = NN_model.predict(X_test)

NN_prediction = list()
for i in range(len(NNpredictions)):
    NN_prediction.append(np.argmax(NNpredictions[i]))


# In[ ]:


# Validation of the results
print(accuracy_score(y_test, NN_prediction))
print("\n")
print(confusion_matrix(y_test, NN_prediction))
print("\n")
print(fbeta_score(y_test, NN_prediction, beta=2))
print("\n")
print(classification_report(y_test, NN_prediction))


# In[ ]:


# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, NNpredictions[:, 1])

lr_auc = roc_auc_score(y_test, NNpredictions[:, 1])

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--', label='No Skill: ROC AUC=%.3f' % (0.5))
plt.plot(fpr, tpr, label='Logistic: ROC AUC=%.3f' % (lr_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# ## SVM model

# In[ ]:


clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)


# In[ ]:


SVMpredictions = clf.predict(X_test)


# In[ ]:


# Validation of the results
print(accuracy_score(y_test,SVMpredictions))
print("\n")
print(confusion_matrix(y_test, SVMpredictions))
print("\n")
print(fbeta_score(y_test, SVMpredictions, beta=2))
print("\n")
print(classification_report(y_test, SVMpredictions))


# In[ ]:




