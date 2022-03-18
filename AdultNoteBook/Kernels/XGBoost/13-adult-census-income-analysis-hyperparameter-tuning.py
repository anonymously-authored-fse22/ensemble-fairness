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


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import LabelBinarizer

# Modelling Libraries
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation & CV Libraries
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import classification_report, f1_score, plot_roc_curve
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold


# In[ ]:


df= pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.head()


# In[ ]:


df.shape


# Dataframe has 32561 rows and 15 columns

# In[ ]:


df.info()


# In[ ]:


df.describe(include='all').head()


# In[ ]:


df.isna().sum()


# There are no null values

# Droping fnlwgt columns as it's not important for exploratory data analysis.

# In[ ]:


df.drop('fnlwgt', axis=1, inplace=True)
df.replace({'?':'Unknown'}, inplace=True)


# # Splitting The Data.

# Split the data into two parts that are related to each other for EDA.

# In[ ]:


work_col = ['workclass','education','education.num','occupation','capital.gain','capital.loss','hours.per.week','income']

dempgraphic_col = ['age','marital.status','relationship','race','sex','native.country']


# In[ ]:


work_col =df[work_col]
dempgraphic_col = df[dempgraphic_col]


# # ****Work Related Column Analysis****

# First Lets do univariate analysis of Work related features.

# In[ ]:


pay = work_col['income'].value_counts()
fig = px.bar(x=pay.index, y=pay, title='Total Income Distribution', text=(work_col['income'].value_counts()/len(work_col['income'])*100))
fig['layout'].update(height=500, width=500)
fig.update_traces(textposition='outside',texttemplate='%{text:.4s}', marker_color=['pink','plum'])
fig.show()


# Income <=50k is almost 76%, Income >50k is 24%. There seems to be slight imbalance in data.

# In[ ]:


trace1 = go.Bar(x=work_col['workclass'].value_counts().index, y=work_col['workclass'].value_counts(), 
                text=(work_col['workclass'].value_counts()/len(work_col['workclass'])*100), 
                marker=dict(color=work_col['workclass'].value_counts(), colorscale='earth'))

trace2 = go.Bar(x=work_col['education'].value_counts().index, y=work_col['education'].value_counts(), 
                text=(work_col['education'].value_counts()/len(work_col['education'])*100),
               marker=dict(color=work_col['education'].value_counts(), colorscale='earth'))

trace3 = go.Bar(x=work_col['occupation'].value_counts().index, y=work_col['occupation'].value_counts(), 
                text=(work_col['occupation'].value_counts()/len(work_col['occupation'])*100),
               marker=dict(color=work_col['occupation'].value_counts(), colorscale='earth'))


fig = make_subplots(rows=2, cols=2, specs=[[{'type':'bar'},{'type':'bar'}],
                                          [{'type':'bar'},None]],
                   subplot_titles=('Work Class Distribution','Education Distribution','Occupation Distribution',
                                  'Hours Per Week Distribution'))
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
fig.append_trace(trace3,2,1)

fig['layout'].update(height=1100, width=1200,title='Work Related Feature Analysis')

fig.update_traces(textposition='outside',texttemplate='%{text:.4s}%')
fig.show()


# According to data around 69.7% people are working privately.
# 
# Around 32.2% where high school graduates, while 22.39% are collage graduates.
# 
# Prof-specialty, Craft-repair, and Exec-management occuaption are higher in count than other occupations.

# In[ ]:


plt.figure(figsize=(25,8))
sns.countplot(x=work_col['hours.per.week'])
plt.title('Hours Per Week', fontsize=30)
plt.xlabel('Hours', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()


# Most people do 40 Hours per week.

# In[ ]:


ax=work_col.groupby('income')[['capital.gain','capital.loss']].agg(['min','max','mean'])
ax.style.background_gradient(cmap="CMRmap_r")


# Bivariate Analysis of Work related Features

# In[ ]:


fig = px.histogram(x=work_col['workclass'], color=work_col['income'],color_discrete_sequence=['grey','yellow'], height=400, width=700, title='Work Class VS Income',
                  labels={'Work':'Work'})
fig.show()

fig = px.histogram(x=work_col['occupation'], color=work_col['income'],color_discrete_sequence=['grey','plum'], height=400, width=700, title='Occupation VS Income')
fig.show()

fig = px.histogram(x=work_col['education'], color=work_col['income'], color_discrete_sequence=['grey','orange'], height=400, width=700, title='Education VS Income')
fig.show()


# People doing private jobs have higher rate of earning >=50k aswell as <50k.
# 
# Rate of earning >=50k is higher in Exec-managerial, Prof-specialty occupation.
# 
# Bachelors degree holder have higher chance of earning >=50k. Masters, Doctorate degrees have lower total count but there rate of earning >=50k
# is alot higher.

# In[ ]:


fig = px.histogram(x=work_col['hours.per.week'], 
                   color=work_col['income'], 
                   height=600, 
                   width=1000,log_y=True,
                  template='ggplot2')

fig.update_layout(paper_bgcolor='rgb(248, 248, 255)',
     plot_bgcolor='rgb(248, 248, 255)',
     showlegend=False,)
fig.show()


# I have used Log scale for better visualization.
# 
# 

# # Demographic Column Analysis

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=dempgraphic_col['age'],
                          xbins=dict(
                          start=17,
                          end=90,
                          size=1),
                          opacity=1))

fig.update_layout(title_text='Age Distribution',
                 xaxis_title='Age',
                 yaxis_title='Count',
                 bargap=0.05,
                 xaxis={'showgrid':False},
                 yaxis={'showgrid':False},
                 template='seaborn',
                 height=600,
                 width=1000)

fig.update_layout(paper_bgcolor='rgb(248, 248, 255)',
     plot_bgcolor='rgb(248, 248, 255)',
     showlegend=False,)
fig.show()


# In[ ]:


colors=['mediumturquoise','lightgreen','seagreen',"rgb(114, 78, 145)",'palegreen','olive','gold','darkorange']

traces1 = go.Pie(values=dempgraphic_col['marital.status'].value_counts(), labels=dempgraphic_col['marital.status'].value_counts().index, marker_colors=['mediumturquoise','lightgreen','seagreen',"rgb(114, 78, 145)",'palegreen','olive'])

traces2 = go.Pie(values=dempgraphic_col['relationship'].value_counts(), labels=dempgraphic_col['relationship'].value_counts().index, marker_colors=['lightcyan','cyan','royalblue','darkblue','steelblue','lightblue'])

traces3 = go.Pie(values=dempgraphic_col['race'].value_counts(), labels=dempgraphic_col['race'].value_counts().index,marker_colors=['pink','plum','coral','salmon'])

traces4 = go.Pie(values=dempgraphic_col['sex'].value_counts(), labels=dempgraphic_col['sex'].value_counts().index, marker_colors=['gold','darkorange'])

fig = make_subplots(rows=2, cols =2, specs=[[{'type':'domain'}, {'type':'domain'}],
                                           [{'type':'domain'},{'type':'domain'}]],
                   subplot_titles=('Marital Status Distribution', 'Relationship Distribution','Race Distribution','Gender Distribution'))

fig.append_trace(traces1,1,1)
fig.append_trace(traces2,1,2)
fig.append_trace(traces3,2,1)
fig.append_trace(traces4,2,2)

fig['layout'].update(height=1000, 
                     title='Demographic Columns Analysis', titlefont_size=20,
                     paper_bgcolor='rgb(248, 248, 255)',
                     plot_bgcolor='rgb(248, 248, 255)',
                     showlegend=False,)

fig.update_traces(hole=.4, pull=[0,0,0.2,0,0], hoverinfo='label+percent', marker_line=dict(color='black', width=2),)

fig.show()

46% of  people are married, whereas 32.8% never married.

Count of husband working is alot higher than wife.

Count of White race is higher than other races.

Male count is double than that of female.
# In[ ]:


fig = px.bar(x=dempgraphic_col['native.country'].value_counts().index, y=dempgraphic_col['native.country'].value_counts(),log_y=True,
             text=(dempgraphic_col['native.country'].value_counts()/len(dempgraphic_col['native.country'])*100))

fig.update_traces(textposition='outside', texttemplate='%{text:.3s}%')
fig['layout'].update(height=500, width=1500,title='Country Count')
fig.show()


# US comprise 89.6% of total data, while only 10% is shared by other countries combined.

# 
# 

# In[ ]:


fig = px.histogram(x=dempgraphic_col['marital.status'], color=df['income'],color_discrete_sequence=['navy','lightblue'], height=400, width=700, title='Marital Status VS Income',
                  labels={'Work':'Work'})
fig.show()

fig = px.histogram(x=dempgraphic_col['relationship'], color=df['income'],color_discrete_sequence=['darkorange','gold'], height=400, width=700, title='Relationship VS Income')
fig.show()

fig = px.histogram(x=dempgraphic_col['sex'], color=df['income'], color_discrete_sequence=['maroon','palevioletred'], height=400, width=600, title='Gender VS Income')
fig.show()

fig = px.histogram(x=dempgraphic_col['race'], color=df['income'], color_discrete_sequence=['pink','peachpuff'], height=400, width=700, title='Race VS Income')
fig.show()


# Rate of unmarried people earning >=50k is high, while count of married people earning >50k is high.
# 
# 

# In[ ]:


fig = px.histogram(x=dempgraphic_col['native.country'], 
                   color=df['income'],log_y=True,
                  width=900)
fig.update_layout(paper_bgcolor='rgb(248, 248, 255)',
     plot_bgcolor='rgb(248, 248, 255)',
     showlegend=False,)
fig.show()


# In[ ]:


fig = px.histogram(x=dempgraphic_col['age'], 
                   color=df['income'],
                  height=500,
                  width=800,
                  template='ggplot2',
                  nbins=100)

fig.update_layout(paper_bgcolor='rgb(248, 248, 255)',
     plot_bgcolor='rgb(248, 248, 255)',
     showlegend=False,)
fig.show()


# Label encoding all the Categorical features.
# 
# people earning  >=50k range between all ages.

# In[ ]:


transformer = ColumnTransformer([
    ('one hot', OneHotEncoder(drop = 'first'), ['relationship', 'race', 'sex']),
    ('binary', ce.BinaryEncoder(), ['workclass', 'marital.status', 'occupation', 'native.country'])],
    remainder = 'passthrough')


# In[ ]:


x = df.drop(['income','education'],axis=1)
y = np.where(df['income'] == '>50K', 1, 0)

print(x.shape)
print(y.shape)


# In[ ]:


x = transformer.fit_transform(x)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=100)


# In[ ]:


models = [('LR',LogisticRegression(max_iter=10000)), ('SVC', SVC()),
         ('DT', DecisionTreeClassifier()), ('RF',RandomForestClassifier()),
         ('KNN',KNeighborsClassifier(n_neighbors=10)), ('GNB',GaussianNB()),
         ('GBC',GradientBoostingClassifier()), ('ADA', AdaBoostClassifier()),
         ('XGB', XGBClassifier())]
results = []
names = []
final_Score =[]

for name,model in models:
    model.fit(x_train,y_train)
    model_results = model.predict(x_test)
    score = accuracy_score(y_test, model_results)
    results.append(score)
    names.append(name)
    final_Score.append((name,score))
    
final_Score.sort(key=lambda k:k[1],reverse=True)


# In[ ]:


final_Score


# In[ ]:


random_gbc ={'learning_rate':[0.0001,0.001,0.01,0.1],
            'n_estimators':[100,200,500,1000],
            'max_features':['sqrt','log2'],
            'max_depth':list(range(11))
                            }

random_xgb = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25 ] ,
            "max_depth"        : [ 3, 4, 5, 6, 8, 10],
            "min_child_weight" : [ 1, 3, 5, 7 ],
            "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
            'eval_metric':['mlogloss']}

random_rf = {'n_estimators':[100,200,500,800,1000],
            'max_features':['auto','sqrt','log2'],
            'max_depth':list(range(1,11)),
            'criterion':['gini','entropy']}


# # GBC Classifier Hyperparameter tuning

# In[ ]:


score = []
gbc_rs = RandomizedSearchCV(estimator= GradientBoostingClassifier(), param_distributions=random_gbc, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
                           n_iter=10,n_jobs=-1, verbose=2)

gbc_rs.fit(x_train, y_train)
gbc_pred = gbc_rs.best_estimator_.predict(x_test)
gbc_best_score = accuracy_score(y_test, gbc_pred)
score.append(['GBC', dict(gbc_rs.best_params_), gbc_best_score])


# # XGB Classifier Hyperparameter tuning

# In[ ]:


xgb_rs = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=random_xgb, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
                           n_iter=10, n_jobs=-1, verbose=2)

xgb_rs.fit(x_train,y_train)
xgb_pred = xgb_rs.best_estimator_.predict(x_test)
xgb_best_score = accuracy_score(y_test, xgb_pred)

score.append(['XGB', dict(xgb_rs.best_params_), xgb_best_score])


# # RandomForest Classifier Hyperparameter tuning

# In[ ]:


rf_rs = RandomizedSearchCV(estimator = RandomForestClassifier(),param_distributions= random_rf, cv= RepeatedStratifiedKFold(n_repeats=5, n_splits=2),
                          n_iter=10, n_jobs=-1, verbose=2)

rf_rs.fit(x_train, y_train)
rf_pred = rf_rs.best_estimator_.predict(x_test)
rf_best_score = accuracy_score(y_test, rf_pred)
score.append(['RandomForest', dict(rf_rs.best_params_), rf_best_score])


# In[ ]:


score = pd.DataFrame(score,columns=['Model','Parameters','Score'])
score


# In[ ]:




