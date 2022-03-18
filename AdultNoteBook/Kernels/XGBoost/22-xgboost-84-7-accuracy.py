#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import os


# In[ ]:


file = ('/kaggle/input/adult.csv')
df = pd.read_csv(file)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


columns = ["workclass","education","marital.status","occupation","relationship","race","sex","native.country","income"]


# In[ ]:


for i in columns:
    print(df[i].value_counts(),"\n","="*50)
    


# In[ ]:


#Replacing "?" values with Mode 
df["workclass"].replace("?",df["workclass"].mode()[0],inplace = True)


# In[ ]:


#Replace 1st-4th,5th-6th,7th-8th,9th,10th to elementary school
#Replace 11th, 12th with HS-grad
sch = ["1st-4th","5th-6th","7th-8th","9th","10th"]
hsgrad = ["11th","12th"]
df["education"].replace(to_replace=sch,value = "elementary_school",inplace=True)
df["education"].replace(to_replace=hsgrad,value = "HS-grad",inplace=True)


# In[ ]:


#Replacning the marital_status into two categories married and unmarried
married = ["Married-civ-spouse","Divorced","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"]
unmarried = ["Never-married"]
df["marital.status"].replace(to_replace=married,value = "married",inplace=True)
df["marital.status"].replace(to_replace=unmarried,value = "unmarried",inplace=True)


# In[ ]:


#replacing the ? with the mode of the column
df["occupation"].replace("?",df["occupation"].mode()[0],inplace = True)


# In[ ]:


#replacing the ? with the mode of the column
df["native.country"].replace("?",df["native.country"].mode()[0],inplace = True)


# In[ ]:


df["income"].value_counts()


# In[ ]:


sns.countplot(x = "income",data = df)
#the given dataset is an imbalanced dataset since the count of each variable attributes column has a huge variance.


# In[ ]:


sns.catplot(x = "workclass",y = "capital.gain",data=df)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# In[ ]:


sns.catplot(x = "workclass",y = "hours.per.week",data=df)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# In[ ]:


sns.countplot(x = "workclass",data=df,hue = "education" )
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# In[ ]:


sns.countplot(x = "education",data=df,hue = "marital.status" )
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(data=df.corr().abs(),annot=True)


# In[ ]:


#dividing the data into input and output
x = df.drop(columns = ["income"])
y = df["income"]


# In[ ]:


df.head()


# In[ ]:


num_cols = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]
cat_cols = ["workclass","education","marital.status","occupation","relationship","race","sex","native.country"]


# In[ ]:


#converting the character variables into dummy values
dumm = pd.get_dummies(x[cat_cols])


# In[ ]:


#scaling the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data=scaler.fit_transform(x[num_cols])


# In[ ]:


#converting the data into dataframes
scaled_data = pd.DataFrame(scaled_data,columns= num_cols)


# In[ ]:


dumm.columns


# In[ ]:


#removning the unwanted columns to avoin dummy variable trap
dumm.drop(columns = ["workclass_Never-worked","education_Preschool","marital.status_unmarried","occupation_Armed-Forces","relationship_Wife","race_Other","sex_Female","native.country_Holand-Netherlands"],inplace=True)


# In[ ]:


#Combning the scaled and dummy variable data
x_inp = pd.concat([scaled_data,dumm],axis=1)


# In[ ]:


#splitting the data into train and test
x_train,x_test,y_train,y_test = train_test_split(x_inp,y,test_size = 0.25,random_state=355)


# In[ ]:


X = pd.concat([x_train,y_train],axis = 1)


# In[ ]:


#balancing the dataset using Random Over-Sampling method
class_o = X[X.income=="<=50K"]
class_1  = X[X.income==">50K"]


# In[ ]:


# class count
class_count_0, class_count_1 = X['income'].value_counts()


# In[ ]:


class_1_over = class_1.sample(class_count_0,replace = True)


# In[ ]:


class_1_over.shape


# In[ ]:


over_sampled = pd.concat([class_o,class_1_over])


# In[ ]:


over_sampled["income"].value_counts()


# In[ ]:


over_sampled_x = over_sampled.drop(columns = ["income"])
over_sampled_y = over_sampled["income"]


# In[ ]:


# Developing the model using XG Boost

model = XGBClassifier(objective='binary:logistic')
model.fit(over_sampled_x, over_sampled_y)


# In[ ]:


y_pred = model.predict(over_sampled_x)
predictions = [(value) for value in y_pred]
accuracy = accuracy_score(over_sampled_y,predictions)
accuracy


# In[ ]:


y_pred = model.predict(x_test)
predictions = [(value) for value in y_pred]
accuracy = accuracy_score(y_test,predictions)
accuracy


# In[ ]:


#Hyperparameter tuning using GridSearchCV
param_grid={
   
    'learning_rate':[1,0.5,0.1,0.01,0.001],
    'max_depth': [3,5,10,20],
    'n_estimators':[10,50,100,200]
    
}


# In[ ]:


grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid, verbose=3)


# In[ ]:


grid.fit(over_sampled_x,over_sampled_y)


# In[ ]:


grid.best_params_


# In[ ]:


new_model=XGBClassifier(learning_rate= 0.5, max_depth= 20, n_estimators= 200)
new_model.fit(over_sampled_x, over_sampled_y)


# In[ ]:


#Accuracy on test file

y_pred_test_file = new_model.predict(x_test)
predictions = [(value) for value in y_pred_test_file]
accuracy = accuracy_score(y_test,predictions)
accuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




