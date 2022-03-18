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
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)


# * RowNumber: Row Serial number
# * CustomerID: Customer Unique Identifier
# * Surname: Customer Last name
# * CreditScore: Customer Credit Score
# * Geography: Conuntry where the Customer lives
# * Gender: Customer Gender
# * Age: Customer Age
# * Tenure: Tenure with Bank
# * Balance: Account Balance
# * NumOfProducts: Number of bank products customer is using
# * HasCrCard: Has a credit card (0 = No, 1 = Yes)
# * IsActiveMember: Is an active member (0 = No, 1 = Yes)
# * EstimatedSalary: Estimated Salary
# * Exited: Exited Bank (0 = No, 1 = Yes)

# In[ ]:


df=pd.read_csv('../input/analysis-of-banking-data-model/Bank_churn_modelling.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


# We drop the 2 columns in the data
df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True,axis=1)
df.head(5)


# In[ ]:


#Separating churn and non churn customers
churn= df[df["Exited"] == 1]
not_churn= df[df["Exited"] == 0]


# In[ ]:


target_col = ["Exited"]
cat_cols   = df.nunique()[df.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in df.columns if x not in cat_cols + target_col]


# In[ ]:


# number of men are more than women
df['Gender'].value_counts()


# In[ ]:


#the highest one belongs to tenure number 2
df['Tenure'].value_counts()


# In[ ]:


# the high portion is for exited 0 with the percent
df['Exited'].value_counts(normalize = True)*100


# In[ ]:


df['Geography'].value_counts()


# In[ ]:


# we want to use some filters for better undrestanding
df[(df['Age']>=50)&(df['EstimatedSalary']>100000)]


# In[ ]:


df[(df['Gender']=='Female')&(df['Exited']==1)&(df['HasCrCard']==0)]


# In[ ]:


df['Geography'].unique()
Geography_Gender=pd.crosstab(df['Geography'],df['Gender'])
Geography_Gender


# In[ ]:


df['HasCrCard'].unique()
HasCrCard_Age=pd.crosstab(df['HasCrCard'],df['Age'])
HasCrCard_Age


# In[ ]:


df['Exited'].unique()
Exited_IsActiveMember=pd.crosstab(df['Exited'],df['IsActiveMember'])
Exited_IsActiveMember


# In[ ]:


df.describe()


# In[ ]:


#middle of the point of the number set
df.median()


# In[ ]:


# sum of the numbers and divide to number by amount of the numbers
df.mean()


# In[ ]:


# we use the groupby with sum, count for Geography with Estimated salary
df_grouped_sum=df.groupby('Gender',as_index=False)['EstimatedSalary'].agg('sum').rename(columns={'EstimatedSalary':'EstimatedSalary_Sum'})
df_grouped_cnt=df.groupby('Gender',as_index=False)['EstimatedSalary'].agg('count').rename(columns={'EstimatedSalary':'EstimatedSalary_Cnt'})

#Merge the 2 groups with each other
df_grouped_Salary=df_grouped_sum.merge(df_grouped_cnt,left_on='Gender',right_on='Gender',how='inner')

#Calcuate the average Estimate salary for each Country
df_grouped_Salary.loc[:,'AVG_EstimatedSalary'] = df_grouped_Salary['EstimatedSalary_Sum'] /df_grouped_Salary['EstimatedSalary_Cnt']

df_grouped_Salary.sort_values('EstimatedSalary_Sum',ascending=False)


# In[ ]:


# like the previous one we use groupby with sum and count for tenure with balance
df_grouped_sum=df.groupby('Tenure',as_index=False)['Balance'].agg('sum').rename(columns={'Balance':'Balance_Sum'})
df_grouped_cnt=df.groupby('Tenure',as_index=False)['Balance'].agg('count').rename(columns={'Balance':'Balance_Cnt'})
#merge the 2 groups with each other
df_grouped_Balance=df_grouped_sum.merge(df_grouped_cnt,left_on='Tenure',right_on='Tenure',how='inner')

#calculate the average balance with each tenure
df_grouped_Balance.loc[:,'AVG_Balance'] = df_grouped_Balance['Balance_Sum'] /df_grouped_Balance['Balance_Cnt']

df_grouped_Balance.sort_values('Balance_Sum',ascending=False)


# # we use the visualization part with different plots

# In[ ]:


plt.rcParams['figure.figsize']=(10,5)
df['Exited'].value_counts().sort_values(ascending=False).plot.bar(color='red')
plt.xlabel('number of the exited costumers')
plt.ylabel('count')
plt.xticks(rotation=50)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize']=(10,5)
df['Gender'].value_counts().sort_values(ascending=False).plot.pie(y='Gender',autopct="%0.1f%%")
plt.title('number of the people by the gender')
plt.axis('off')
plt.show()


# In[ ]:


sns.catplot('Gender','EstimatedSalary',data=df,kind='box',color='red')
plt.show()


# In[ ]:


sns.relplot('Age','Balance',data=df, kind='line',ci=None)
plt.show()


# In[ ]:


sns.scatterplot(x='Age',y='CreditScore',data=df,color='orange')
plt.show()


# In[ ]:


sns.jointplot('Age','EstimatedSalary',data=df,kind='kde',color='pink')
plt.show()


# In[ ]:


sns.countplot(data=df, x='Exited',palette='Set3')
plt.title('number of the clients who exited from the bank')
plt.show()


# In[ ]:


#as you see, the most exited part(0) is for france 
fig,ax=plt.subplots(figsize=(20,5))
sns.countplot(df['Geography'],hue=df['Exited'],ax=ax)
plt.xlabel('Abundance of Geography')
plt.ylabel('counts')
plt.xticks(rotation=40)
plt.show()


# In[ ]:


del_corr=df.corr()
f,ax=plt.subplots(figsize=(10,5))

#we want to use heatmap
sns.heatmap(del_corr,annot=True,cmap='inferno')
plt.show()


# * in the heatmap, cells with 1.0 has the highest correlation and cells near 1 also has the highest correlation.

# In[ ]:


plt.figure(figsize=(10,5))
sns.histplot(df['Age'],kde=True,color='orange')


# In[ ]:


import plotly.tools as tls
def plot_pie(column):
    df1= go.Pie(values=churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "Churn Customers",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6
                   ) 
    
    df2=go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Non churn customers" 
                   )

    
    
    layout= go.Layout(dict(title = column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "churn customers",
                                                font = dict(size = 14),
                                                showarrow = False,
                                                x = .20, y = .5),
                                           dict(text = "Non churn customers",
                                                font = dict(size = 14),
                                                showarrow = False,
                                                x = .83,y = .4
                                               )
                                          ]
                           )
                      )
        
    
    df = [df1,df2]
    fig  = go.Figure(data = df,layout = layout)
    py.iplot(fig)
    
#for all categorical columns plot pie
#for i in columns :
  #plot_pie(i)
    


# In[ ]:


plot_pie(cat_cols[0])


# *  as you see , the most portion of the churn costumers for Germany(40%) and Spain has just 20.3% and for non churn costumers France has the highest portion.**

# In[ ]:


plot_pie(cat_cols[1])


# * the highest portion for churn costumer is for female that almost 56% and for non churn costumers is for Male with almost 58%

# In[ ]:


plot_pie(cat_cols[4])


#  the output shows that in churn costumers the high portion which the member is not active and for non churn costumers the most portion is 55.5 which  the member is active 

# In[ ]:


plot_pie(cat_cols[3])


# * the output shows that in churn costumers and in non churn costumers  most of them have credit card( 70% and 70.7%)

# # we want to use histogarm for analyzing the continuous variables

# In[ ]:


def histogram(column):
    df1=go.Histogram(x=churn[column],
                     name='Churn Costumers',
                     histnorm="percent",
                     marker=dict(line=dict(width=.3,
                                           color='black'
                                          
                                          )
                                ),
                     
                    opacity=.8
                    )
    
    df2=go.Histogram(x=not_churn[column],
                     name="Non Churn Costumer",
                     histnorm="percent",
                     marker=dict(line=dict(width=.3,
                                           color='black'
                                          
                                          )
                                ),
                     opacity=.8
                    )
    data=[df1,df2]
    layout=go.Layout(dict(title=column+"Distribution in costumer attrition",
                          plot_bgcolor="rgb(243,243,243)",
                          paper_bgcolor="rgb(243,243,243)",
                          xaxis=dict(gridcolor="rgb(255,255,255)",
                                     title=column,
                                     zerolinewidth=1,
                                     ticklen=5,
                                     gridwidth=2
                                    ),
                          yaxis=dict(gridcolor="rgb(255,255,255)",
                                     title="percent",
                                     zerolinewidth=1,
                                     ticklen=5,
                                     gridwidth=2
                                    
                                    ),
                         )
                    )
    fig=go.Figure(data=data,layout=layout)
    py.iplot(fig)

    


# In[ ]:


#it shows that the costumers with the age 46 has the highest percet and churn.
histogram(num_cols[2])


# In[ ]:


#as you see, the tenure numebr 1 has the highest portion and churn.
histogram(num_cols[3])


# In[ ]:


# in this graph,the balance with 107.5k has the highest portion and churn 
histogram(num_cols[4])


# In[ ]:


histogram(num_cols[3])


# **we want to prepare the data for modelling and dived to tarin and test and use the fiture engineering which has the most important feature**

# In[ ]:


#Preparation the data
list=['Geography','Gender']
df=pd.get_dummies(df,columns=list,prefix=list)


# In[ ]:


df.head(5)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#we want to find out which feature has the most crucial and important
x=df.drop('Exited',axis=1)
y=df.Exited
feature_label=x.columns
model=RandomForestClassifier(n_estimators = 10000)
model.fit(x,y)
importances=model.feature_importances_
indices=np.argsort(importances)[::-1]
for i in range (x.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, feature_label[i], importances[indices[i]]))


# In[ ]:


plt.title('importance of features')
plt.bar(range(x.shape[1]),importances[indices],color='orange',align='center')
plt.xticks(range(x.shape[1]),feature_label,rotation=60)
plt.show()
#As you see, Creditscore is the most important feature among another features


# In[ ]:


#train and test 
y = df["Exited"]
x = df.drop(["Exited","Geography_Germany", "Geography_Spain", "Gender_Male", "HasCrCard","IsActiveMember","CreditScore","Age","Balance","EstimatedSalary"], axis = 1)


# In[ ]:


# Train-Test Separation
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.30, 
                                                    random_state=12345)


# In[ ]:


from imblearn.combine import SMOTETomek
model=SMOTETomek()
#Oversample training  data
model.fit(x_train,y_train)
#Validate the data
model.fit(x_test,y_test)


# In[ ]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[ ]:


# now we want to make some models
models=[]
models.append(('LR',LogisticRegression(random_state=12345)))
models.append(('DT',DecisionTreeClassifier(random_state=(12345))))
models.append(('NN',MLPClassifier(random_state=(12345))))
models.append(('SVM',SVC(random_state=(12345))))
models.append(('RF',RandomForestClassifier(random_state=(12345))))
models.append(('GB',GradientBoostingClassifier(random_state=(12345))))
models.append(('KN', KNeighborsClassifier()))
#valuate
result=[]
name=[]


# In[ ]:


#now we want to compare the models with each other
for name, model in models:
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)
    accuracy=accuracy_score(y_test,predictions)
    msg = "%s: (%f)" % (name, accuracy)
    print(msg)

