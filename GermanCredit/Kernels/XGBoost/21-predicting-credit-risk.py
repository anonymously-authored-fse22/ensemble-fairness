#!/usr/bin/env python
# coding: utf-8

# <h1>Welcome to my Kernel ! </h1>
# 
# <h2>I have used the inital code which does data exploration and feature engineering on the German Credit Risk to understand their distribuitions and patterns. Following this I have performed my own LightGBM training using Optuna for learning purpose</h2>
# 

# Copied from https://www.kaggle.com/kabure/<br>

# # Tables of Content:
# 
# **1. [Introduction](#Introduction)** <br>
#     - Info's about datasets
# **2. [Librarys](#Librarys)** <br>
#     - Importing Librarys
#     - Importing Dataset
# **3. [Knowning the data](#Known)** <br>
#     - 3.1 Looking the Type of Data
#     - 3.2 Shape of data
#     - 3.3 Null Numbers
#     - 3.4 Unique values
#     - 3.5 The first rows of our dataset
# **4. [Exploring some Variables](#Explorations)** <br>
#     - 4.1 Ploting some graphical and descriptive informations
# **5. [Correlation of data](#Correlation)** <br>
# 	- 5.1 Correlation Data
# **6. [Preprocess](#Preprocessing)** <br>
# 	- 6.1 Importing Librarys
# 	- 6.2 Setting X and Y
#     - 6.3 Spliting the X and Y in train and test 
# **7. 1 [Model 1](#Modelling 1)** <br>
#     - 7.1.1 Random Forest 
#     - 7.1.2 Score values
#     - 7.1.3 Cross Validation 
# **7. 2 [Model 2](#Modelling 2)** <br>
#     - 7.2.1 Logistic Regression 
#     - 7.2.2 Score values
#     - 7.2.3 Cross Validation 
#     - 7.2.4 ROC Curve

# <a id="Introduction"></a> <br>
# 

# # **1. Introduction:** 
# <h2>Context</h2>
# The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes. The link to the original dataset can be found below.
# 
# <h2>Content</h2>
# It is almost impossible to understand the original dataset due to its complicated system of categories and symbols. Thus, I wrote a small Python script to convert it into a readable CSV file. Several columns are simply ignored, because in my opinion either they are not important or their descriptions are obscure. The selected attributes are:
# 
# <b>Age </b>(numeric)<br>
# <b>Sex </b>(text: male, female)<br>
# <b>Job </b>(numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)<br>
# <b>Housing</b> (text: own, rent, or free)<br>
# <b>Saving accounts</b> (text - little, moderate, quite rich, rich)<br>
# <b>Checking account </b>(numeric, in DM - Deutsch Mark)<br>
# <b>Credit amount</b> (numeric, in DM)<br>
# <b>Duration</b> (numeric, in month)<br>
# <b>Purpose</b>(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others<br>
# <b>Risk </b> (Value target - Good or Bad Risk)<br>

# <a id="Librarys"></a> <br>
# # **2. Librarys:** 
# - Importing Librarys
# - Importing Dataset

# In[ ]:


#Load the librarys
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn

#Importing the data
df_credit = pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv",index_col=0)


# <a id="Known"></a> <br>
# # **3. First Look at the data:** 
# - Looking the Type of Data
# - Null Numbers
# - Unique values
# - The first rows of our dataset

# In[ ]:


#Searching for Missings,type of data and also known the shape of data
print(df_credit.info())


# In[ ]:


#Looking unique values
print(df_credit.nunique())
#Looking the data
print(df_credit.head())


# # **4. Some explorations:** <a id="Explorations"></a> <br>
# 
# - Starting by distribuition of column Age.
# - Some Seaborn graphical
# - Columns crossing
# 
# 

# <h2>Let's start looking through target variable and their distribuition</h2>

# In[ ]:


# it's a library that we work with plotly
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
from collections import Counter # To do counter of some features

trace0 = go.Bar(
            x = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().index.values,
            y = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().values,
            name='Good credit'
    )

trace1 = go.Bar(
            x = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().index.values,
            y = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().values,
            name='Bad credit'
    )

data = [trace0, trace1]

layout = go.Layout(
    
)

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Risk Variable'
    ),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')


# I will try implement some interactive visuals in my Kernels, this will be the first, inspired in Alexader's Kernel and I will also continue implementing plotly and bokeh in my Kerne

# In[ ]:


df_good = df_credit.loc[df_credit["Risk"] == 'good']['Age'].values.tolist()
df_bad = df_credit.loc[df_credit["Risk"] == 'bad']['Age'].values.tolist()
df_age = df_credit['Age'].values.tolist()

#First plot
trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)
#Second plot
trace1 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)
#Third plot
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# In[ ]:


df_good = df_credit[df_credit["Risk"] == 'good']
df_bad = df_credit[df_credit["Risk"] == 'bad']

fig, ax = plt.subplots(nrows=2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

g1 = sns.distplot(df_good["Age"], ax=ax[0], 
             color="g")
g1 = sns.distplot(df_bad["Age"], ax=ax[0], 
             color='r')
g1.set_title("Age Distribuition", fontsize=15)
g1.set_xlabel("Age")
g1.set_xlabel("Frequency")

g2 = sns.countplot(x="Age",data=df_credit, 
              palette="hls", ax=ax[1], 
              hue = "Risk")
g2.set_title("Age Counting by Risk", fontsize=15)
g2.set_xlabel("Age")
g2.set_xlabel("Count")
plt.show()


# <h2>Creating an categorical variable to handle with the Age variable </h2>

# In[ ]:


#Let's look the Credit Amount column
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)


df_good = df_credit[df_credit["Risk"] == 'good']
df_bad = df_credit[df_credit["Risk"] == 'bad']


# In[ ]:


trace0 = go.Box(
    y=df_good["Credit amount"],
    x=df_good["Age_cat"],
    name='Good credit',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_bad['Credit amount'],
    x=df_bad['Age_cat'],
    name='Bad credit',
    marker=dict(
        color='#FF4136'
    )
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount (US Dollar)',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')


# Interesting distribuition

# <h2>I will now Look the distribuition of Housing own and rent by Risk</h2>
# 

# In[ ]:


#First plot
trace0 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'good']["Housing"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'good']["Housing"].value_counts().values,
    name='Good credit'
)

#Second plot
trace1 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'bad']["Housing"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'bad']["Housing"].value_counts().values,
    name="Bad Credit"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Housing Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Housing-Grouped')


# we can see that the own and good risk have a high correlation

# <h3>Distribuition of Credit Amount by Housing</h3>

# In[ ]:


fig = {
    "data": [
        {
            "type": 'violin',
            "x": df_good['Housing'],
            "y": df_good['Credit amount'],
            "legendgroup": 'Good Credit',
            "scalegroup": 'No',
            "name": 'Good Credit',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "x": df_bad['Housing'],
            "y": df_bad['Credit amount'],
            "legendgroup": 'Bad Credit',
            "scalegroup": 'No',
            "name": 'Bad Credit',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'green'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


py.iplot(fig, filename = 'violin/split', validate = False)


# Interesting moviments! Highest values come from category "free" and we have a different distribuition by Risk

# <h2>Looking the diference by Sex</h2>

# In[ ]:


#First plot
trace0 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'good']["Sex"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'good']["Sex"].value_counts().values,
    name='Good credit'
)

#First plot 2
trace1 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'bad']["Sex"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'bad']["Sex"].value_counts().values,
    name="Bad Credit"
)

#Second plot
trace2 = go.Box(
    x = df_credit[df_credit["Risk"]== 'good']["Sex"],
    y = df_credit[df_credit["Risk"]== 'good']["Credit amount"],
    name=trace0.name
)

#Second plot 2
trace3 = go.Box(
    x = df_credit[df_credit["Risk"]== 'bad']["Sex"],
    y = df_credit[df_credit["Risk"]== 'bad']["Credit amount"],
    name=trace1.name
)

data = [trace0, trace1, trace2,trace3]


fig = tls.make_subplots(rows=1, cols=2, 
                        subplot_titles=('Sex Count', 'Credit Amount by Sex'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 2)

fig['layout'].update(height=400, width=800, title='Sex Distribuition', boxmode='group')
py.iplot(fig, filename='sex-subplot')


# <b> How can I set the boxplots in different places? how can I use the same legend to both graphs?</b>

# I will create categories of Age and look the distribuition of Credit Amount by Risk...
# 

# I will do some explorations through the Job
# - Distribuition
# - Crossed by Credit amount
# - Crossed by Age

# In[ ]:


#First plot
trace0 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'good']["Job"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'good']["Job"].value_counts().values,
    name='Good credit Distribuition'
)

#Second plot
trace1 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'bad']["Job"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'bad']["Job"].value_counts().values,
    name="Bad Credit Distribuition"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Job Distribuition'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')


# In[ ]:


trace0 = go.Box(
    x=df_good["Job"],
    y=df_good["Credit amount"],
    name='Good credit'
)

trace1 = go.Box(
    x=df_bad['Job'],
    y=df_bad['Credit amount'],
    name='Bad credit'
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount distribuition by Job'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')


# In[ ]:



fig = {
    "data": [
        {
            "type": 'violin',
            "x": df_good['Job'],
            "y": df_good['Age'],
            "legendgroup": 'Good Credit',
            "scalegroup": 'No',
            "name": 'Good Credit',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "x": df_bad['Job'],
            "y": df_bad['Age'],
            "legendgroup": 'Bad Credit',
            "scalegroup": 'No',
            "name": 'Bad Credit',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'green'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


py.iplot(fig, filename = 'Age-Housing', validate = False)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12), nrows=2)

g1 = sns.boxplot(x="Job", y="Credit amount", data=df_credit, 
            palette="hls", ax=ax[0], hue="Risk")
g1.set_title("Credit Amount by Job", fontsize=15)
g1.set_xlabel("Job Reference", fontsize=12)
g1.set_ylabel("Credit Amount", fontsize=12)

g2 = sns.violinplot(x="Job", y="Age", data=df_credit, ax=ax[1],  
               hue="Risk", split=True, palette="hls")
g2.set_title("Job Type reference x Age", fontsize=15)
g2.set_xlabel("Job Reference", fontsize=12)
g2.set_ylabel("Age", fontsize=12)

plt.subplots_adjust(hspace = 0.4,top = 0.9)

plt.show()


# Looking the distribuition of Credit Amont

# In[ ]:


import plotly.figure_factory as ff

import numpy as np

# Add histogram data
x1 = np.log(df_good['Credit amount']) 
x2 = np.log(df_bad["Credit amount"])

# Group data together
hist_data = [x1, x2]

group_labels = ['Good Credit', 'Bad Credit']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

# Plot!
py.iplot(fig, filename='Distplot with Multiple Datasets')


# In[ ]:


#Ploting the good and bad dataframes in distplot
plt.figure(figsize = (8,5))

g= sns.distplot(df_good['Credit amount'], color='r')
g = sns.distplot(df_bad["Credit amount"], color='g')
g.set_title("Credit Amount Frequency distribuition", fontsize=15)
plt.show()


# Distruibution of Saving accounts by Risk

# In[ ]:


from plotly import tools
import numpy as np
import plotly.graph_objs as go

count_good = go.Bar(
    x = df_good["Saving accounts"].value_counts().index.values,
    y = df_good["Saving accounts"].value_counts().values,
    name='Good credit'
)
count_bad = go.Bar(
    x = df_bad["Saving accounts"].value_counts().index.values,
    y = df_bad["Saving accounts"].value_counts().values,
    name='Bad credit'
)


box_1 = go.Box(
    x=df_good["Saving accounts"],
    y=df_good["Credit amount"],
    name='Good credit'
)
box_2 = go.Box(
    x=df_bad["Saving accounts"],
    y=df_bad["Credit amount"],
    name='Bad credit'
)

scat_1 = go.Box(
    x=df_good["Saving accounts"],
    y=df_good["Age"],
    name='Good credit'
)
scat_2 = go.Box(
    x=df_bad["Saving accounts"],
    y=df_bad["Age"],
    name='Bad credit'
)

data = [scat_1, scat_2, box_1, box_2, count_good, count_bad]

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Count Saving Accounts','Credit Amount by Savings Acc', 
                                          'Age by Saving accounts'))

fig.append_trace(count_good, 1, 1)
fig.append_trace(count_bad, 1, 1)

fig.append_trace(box_2, 1, 2)
fig.append_trace(box_1, 1, 2)

fig.append_trace(scat_1, 2, 1)
fig.append_trace(scat_2, 2, 1)



fig['layout'].update(height=700, width=800, title='Saving Accounts Exploration', boxmode='group')

py.iplot(fig, filename='combined-savings')


# How can I better configure the legends?  I am trying to substitute the graph below, so how can I use the violinplot on subplots of plotly?

# In[ ]:


print("Description of Distribuition Saving accounts by Risk:  ")
print(pd.crosstab(df_credit["Saving accounts"],df_credit.Risk))

fig, ax = plt.subplots(3,1, figsize=(12,12))
g = sns.countplot(x="Saving accounts", data=df_credit, palette="hls", 
              ax=ax[0],hue="Risk")
g.set_title("Saving Accounts Count", fontsize=15)
g.set_xlabel("Saving Accounts type", fontsize=12)
g.set_ylabel("Count", fontsize=12)

g1 = sns.violinplot(x="Saving accounts", y="Job", data=df_credit, palette="hls", 
               hue = "Risk", ax=ax[1],split=True)
g1.set_title("Saving Accounts by Job", fontsize=15)
g1.set_xlabel("Savings Accounts type", fontsize=12)
g1.set_ylabel("Job", fontsize=12)

g = sns.boxplot(x="Saving accounts", y="Credit amount", data=df_credit, ax=ax[2],
            hue = "Risk",palette="hls")
g2.set_title("Saving Accounts by Credit Amount", fontsize=15)
g2.set_xlabel("Savings Accounts type", fontsize=12)
g2.set_ylabel("Credit Amount(US)", fontsize=12)

plt.subplots_adjust(hspace = 0.4,top = 0.9)

plt.show()


# Pretty and interesting distribution...

# In[ ]:


print("Values describe: ")
print(pd.crosstab(df_credit.Purpose, df_credit.Risk))

plt.figure(figsize = (14,12))

plt.subplot(221)
g = sns.countplot(x="Purpose", data=df_credit, 
              palette="hls", hue = "Risk")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)
g.set_title("Purposes Count", fontsize=20)

plt.subplot(222)
g1 = sns.violinplot(x="Purpose", y="Age", data=df_credit, 
                    palette="hls", hue = "Risk",split=True)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Purposes by Age", fontsize=20)

plt.subplot(212)
g2 = sns.boxplot(x="Purpose", y="Credit amount", data=df_credit, 
               palette="hls", hue = "Risk")
g2.set_xlabel("Purposes", fontsize=12)
g2.set_ylabel("Credit Amount", fontsize=12)
g2.set_title("Credit Amount distribuition by Purposes", fontsize=20)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()


# Duration of the loans distribuition and density

# In[ ]:


plt.figure(figsize = (12,14))

g= plt.subplot(311)
g = sns.countplot(x="Duration", data=df_credit, 
              palette="hls",  hue = "Risk")
g.set_xlabel("Duration Distribuition", fontsize=12)
g.set_ylabel("Count", fontsize=12)
g.set_title("Duration Count", fontsize=20)

g1 = plt.subplot(312)
g1 = sns.pointplot(x="Duration", y ="Credit amount",data=df_credit,
                   hue="Risk", palette="hls")
g1.set_xlabel("Duration", fontsize=12)
g1.set_ylabel("Credit Amount(US)", fontsize=12)
g1.set_title("Credit Amount distribuition by Duration", fontsize=20)

g2 = plt.subplot(313)
g2 = sns.distplot(df_good["Duration"], color='g')
g2 = sns.distplot(df_bad["Duration"], color='r')
g2.set_xlabel("Duration (month)", fontsize=12)
g2.set_ylabel("Frequency", fontsize=12)
g2.set_title("Duration Frequency x good and bad Credit", fontsize=20)

plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)

plt.show()


# 
# Interesting, we can see that the highest duration have the high amounts. <br>
# The highest density is between [12 ~ 18 ~ 24] months<br>
# It all make sense.
# 

# <h2> Checking Account variable </h2>

# First, let's look the distribuition 

# In[ ]:


#First plot
trace0 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'good']["Checking account"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'good']["Checking account"].value_counts().values,
    name='Good credit Distribuition' 
    
)

#Second plot
trace1 = go.Bar(
    x = df_credit[df_credit["Risk"]== 'bad']["Checking account"].value_counts().index.values,
    y = df_credit[df_credit["Risk"]== 'bad']["Checking account"].value_counts().values,
    name="Bad Credit Distribuition"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Checking accounts Distribuition',
    xaxis=dict(title='Checking accounts name'),
    yaxis=dict(title='Count'),
    barmode='group'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename = 'Age-ba', validate = False)


# Now, we will verify the values through Checking Accounts

# In[ ]:


df_good = df_credit[df_credit["Risk"] == 'good']
df_bad = df_credit[df_credit["Risk"] == 'bad']

trace0 = go.Box(
    y=df_good["Credit amount"],
    x=df_good["Checking account"],
    name='Good credit',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_bad['Credit amount'],
    x=df_bad['Checking account'],
    name='Bad credit',
    marker=dict(
        color='#FF4136'
    )
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Cheking distribuition'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')


# The old plot that I am trying to substitute with interactive plots

# In[ ]:


print("Total values of the most missing variable: ")
print(df_credit.groupby("Checking account")["Checking account"].count())

plt.figure(figsize = (12,10))

g = plt.subplot(221)
g = sns.countplot(x="Checking account", data=df_credit, 
              palette="hls", hue="Risk")
g.set_xlabel("Checking Account", fontsize=12)
g.set_ylabel("Count", fontsize=12)
g.set_title("Checking Account Counting by Risk", fontsize=20)

g1 = plt.subplot(222)
g1 = sns.violinplot(x="Checking account", y="Age", data=df_credit, palette="hls", hue = "Risk",split=True)
g1.set_xlabel("Checking Account", fontsize=12)
g1.set_ylabel("Age", fontsize=12)
g1.set_title("Age by Checking Account", fontsize=20)

g2 = plt.subplot(212)
g2 = sns.boxplot(x="Checking account",y="Credit amount", data=df_credit,hue='Risk',palette="hls")
g2.set_xlabel("Checking Account", fontsize=12)
g2.set_ylabel("Credit Amount(US)", fontsize=12)
g2.set_title("Credit Amount by Cheking Account", fontsize=20)

plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top = 0.9)

plt.show()
plt.show()


# Crosstab session and anothers to explore our data by another metrics a little deep

# In[ ]:


print(pd.crosstab(df_credit.Sex, df_credit.Job))


# In[ ]:


plt.figure(figsize = (10,6))

g = sns.violinplot(x="Housing",y="Job",data=df_credit,
                   hue="Risk", palette="hls",split=True)
g.set_xlabel("Housing", fontsize=12)
g.set_ylabel("Job", fontsize=12)
g.set_title("Housing x Job - Dist", fontsize=20)

plt.show()


# In[ ]:


print(pd.crosstab(df_credit["Checking account"],df_credit.Sex))


# In[ ]:


date_int = ["Purpose", 'Sex']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_credit[date_int[0]], df_credit[date_int[1]]).style.background_gradient(cmap = cm)


# In[ ]:


date_int = ["Purpose", 'Sex']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_credit[date_int[0]], df_credit[date_int[1]]).style.background_gradient(cmap = cm)


# ## Looking the total of values in each categorical feature

# In[ ]:


print("Purpose : ",df_credit.Purpose.unique())
print("Sex : ",df_credit.Sex.unique())
print("Housing : ",df_credit.Housing.unique())
print("Saving accounts : ",df_credit['Saving accounts'].unique())
print("Risk : ",df_credit['Risk'].unique())
print("Checking account : ",df_credit['Checking account'].unique())
print("Aget_cat : ",df_credit['Age_cat'].unique())


# ## Let's do some feature engineering on this values and create variable Dummies of the values

# In[ ]:


def one_hot_encoder(df, nan_as_category = False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# ## Transforming the data into Dummy variables

# In[ ]:


df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

#Purpose to Dummies Variable
df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#Sex feature in dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
# Housing get dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
# Housing get Saving Accounts
df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
# Housing get Risk
df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
# Housing get Checking Account
df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
# Housing get Age categorical
df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)


# ## Deleting the old features

# In[ ]:


#Excluding the missing columns
del df_credit["Saving accounts"]
del df_credit["Checking account"]
del df_credit["Purpose"]
del df_credit["Sex"]
del df_credit["Housing"]
del df_credit["Age_cat"]
del df_credit["Risk"]
del df_credit['Risk_good']


# 

# # **5. Correlation:** <a id="Correlation"></a> <br>
# - Looking at the data correlation
# <h1>Looking at the correlation of the data

# In[ ]:


plt.figure(figsize=(14,12))
sns.heatmap(df_credit.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True,  linecolor='white', annot=True)
plt.show()


# # **6. Preprocessing:** <a id="Preprocessing"></a> <br>
# - Importing ML librarys
# - Setting X and y variables to the prediction
# - Splitting Data
# 

# In[ ]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[ ]:


df_credit['Credit amount'] = np.log(df_credit['Credit amount'])


# In[ ]:


#Creating the X and y variables
X = df_credit.drop('Risk_bad', 1).values
y = df_credit["Risk_bad"].values

# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)


# In[ ]:


# # to feed the random state
# seed = 7

# # prepare models
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('RF', RandomForestClassifier()))
# models.append(('SVM', SVC(gamma='auto')))
# models.append(('XGB', XGBClassifier()))

# # evaluate each model in turn
# results = []
# names = []
# scoring = 'recall'

# for name, model in models:
#         kfold = KFold(n_splits=10, random_state=seed)
#         cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
#         results.append(cv_results)
#         names.append(name)
#         msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#         print(msg)
        
# # boxplot algorithm comparison
# fig = plt.figure(figsize=(11,6))
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()


# Very interesting. Almost all models shows a low value to recall. 
# 
# We can observe that our best results was with CART, NB and XGBoost. <br>
# I will implement some models and try to do a simple Tunning on them

# # **7.1 Model 1 :** <a id="Modelling 1"></a> <br>
# - Using Random Forest to predictict the credit score 
# - Some of Validation Parameters

# In[ ]:


# #Seting the Hyper Parameters
# param_grid = {"max_depth": [3,5, 7, 10,None],
#               "n_estimators":[3,5,10,25,50,150],
#               "max_features": [4,7,15,20]}

# #Creating the classifier
# model = RandomForestClassifier(random_state=2)

# grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
# grid_search.fit(X_train, y_train)


# In[ ]:


# print(grid_search.best_score_)
# print(grid_search.best_params_)


# In[ ]:


# rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)

# #trainning with the best params
# rf.fit(X_train, y_train)


# In[ ]:


# #Testing the model 
# #Predicting using our  model
# y_pred = rf.predict(X_test)

# # Verificaar os resultados obtidos
# print(accuracy_score(y_test,y_pred))
# print("\n")
# print(confusion_matrix(y_test, y_pred))
# print("\n")
# print(fbeta_score(y_test, y_pred, beta=2))


# # **7.2 Model 2:** <a id="Modelling 2"></a> <br>

# In[ ]:


# from sklearn.utils import resample
# from sklearn.metrics import roc_curve


# In[ ]:


# # Criando o classificador logreg
# GNB = GaussianNB()

# # Fitting with train data
# model = GNB.fit(X_train, y_train)


# In[ ]:


# # Printing the Training Score
# print("Training score data: ")
# print(model.score(X_train, y_train))


# In[ ]:


# y_pred = model.predict(X_test)

# print(accuracy_score(y_test,y_pred))
# print("\n")
# print(confusion_matrix(y_test, y_pred))
# print("\n")
# print(classification_report(y_test, y_pred))


# With the Gaussian Model we got a best recall. 

# ## Let's verify the ROC curve

# In[ ]:


# #Predicting proba
# y_pred_prob = model.predict_proba(X_test)[:,1]

# # Generate ROC curve values: fpr, tpr, thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# # Plot ROC curve
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()


# In[ ]:


# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import FeatureUnion
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest


# In[ ]:


# features = []
# features.append(('pca', PCA(n_components=2)))
# features.append(('select_best', SelectKBest(k=6)))
# feature_union = FeatureUnion(features)
# # create pipeline
# estimators = []
# estimators.append(('feature_union', feature_union))
# estimators.append(('logistic', GaussianNB()))
# model = Pipeline(estimators)
# # evaluate pipeline
# seed = 7
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(model, X_train, y_train, cv=kfold)
# print(results.mean())


# In[ ]:


# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(accuracy_score(y_test,y_pred))
# print("\n")
# print(confusion_matrix(y_test, y_pred))
# print("\n")
# print(fbeta_score(y_test, y_pred, beta=2))


# # Hyperparameter optimization with cross-validation
# 

# Using optuna to perform initial hyperparamter optimisation
# 
# The following hyperparamters are tuned using the implementation
# 
# * feature_fraction
# * num_leaves
# * bagging_fraction and bagging_freq
# * feature_fraction (again)
# * regularization factors (i.e. 'lambda_l1' and 'lambda_l2')
# * min_child_samples

# In[ ]:


#Creating the X and y variables
# Defined earlier
# X = df_credit.drop('Risk_bad', 1).values
# y = df_credit["Risk_bad"].values


# In[ ]:


X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ## SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


y_train.shape


# In[ ]:


np.unique(y_train, return_counts=True)


# In[ ]:


oversample = SMOTE()
os_X,os_y =oversample.fit_resample(X_train, y_train)


# In[ ]:


np.unique(os_y, return_counts=True)


# In[ ]:


import optuna.integration.lightgbm as lgb
import optuna
from sklearn.model_selection import RepeatedKFold


# In[ ]:


rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# select another metric?
# binary error is inversely proportional to accuracy
params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",                
        "seed": 42
    }

# minimize the error - maximise the accuracy
study_tuner = optuna.create_study(direction='minimize')
# dtrain = lgb.Dataset(X_train, label=y_train)
# Passing SMOTE data
dtrain = lgb.Dataset(os_X, label=os_y)


# Suppress information only outputs - otherwise optuna is 
# quite verbose, which can be nice, but takes up a lot of space
optuna.logging.set_verbosity(optuna.logging.WARNING) 

# Run optuna LightGBMTunerCV tuning of LightGBM with cross-validation

# learning rate callback was initially implemented - why is it hardcoded so?
# training time reduced considerably when using callback (Took more than 2 hrs to train only feature fraction and num leaves)- the session failed after that
# so tried to comment out the callback. Now, it took few mins to run the first two hyperparameters that it trains - validation score more or less similar. 
# using callback - is it important?

# Possibly it is not important - https://github.com/Microsoft/LightGBM/issues/1841
# However, keep in mind that shrinking the learning rate over increasing boosting rounds could lead to tremendously high overfitting; 
# so I guess that you will rarely want to decrease the learning rate over time 
# (as opposed to in deep learning, where the lower bias in the model across epochs helps mitigating the overfit caused by the shrinkage) quoting "julioasotodv"
tuner = lgb.LightGBMTunerCV(params, 
                            dtrain, 
                            study=study_tuner,
                            verbose_eval=False,                            
                            early_stopping_rounds=250,
                            time_budget=19800, # Time budget of 5 hours, we will not really need it
                            seed = 42,
                            folds=rkf,
                            num_boost_round=10000,
#                             callbacks=[lgb.reset_parameter(learning_rate = [0.005]*200 + [0.001]*9800) ] #[0.1]*5 + [0.05]*15 + [0.01]*45 + 
                           )

tuner.run()


# In[ ]:


print(tuner.best_params)
# Classification error
print(tuner.best_score)
# Or expressed as accuracy
print(1.0-tuner.best_score)


# **W/O SMOTE**
# 
# tuner.best_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42, 'feature_pre_filter': False, 'lambda_l1': 4.311120696175597e-06, 'lambda_l2': 0.052582244425462714, 'num_leaves': 13, 'feature_fraction': 0.45199999999999996, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20}
# 
# Logloss error: tuner.best_score = 0.5280193431880829

# **With SMOTE**
# 
# tune.best_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42, 'feature_pre_filter': False, 'lambda_l1': 2.8205516919422897e-07, 'lambda_l2': 5.737854219621876e-07, 'num_leaves': 34, 'feature_fraction': 0.6, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20}
# 
# Logloss error: tuner.best_score = 0.3925921826901682
# 

# In[ ]:


# Set-up a temporary set of best parameters that we will use as a starting point below.
# Note that optuna will complain about values on the edge of the search space, so we move 
# such values a tiny little bit inside the search space.
tmp_best_params = tuner.best_params
if tmp_best_params['feature_fraction']==1:
    tmp_best_params['feature_fraction']=1.0-1e-9
if tmp_best_params['feature_fraction']==0:
    tmp_best_params['feature_fraction']=1e-9
if tmp_best_params['bagging_fraction']==1:
    tmp_best_params['bagging_fraction']=1.0-1e-9
if tmp_best_params['bagging_fraction']==0:
    tmp_best_params['bagging_fraction']=1e-9  


# In[ ]:





# ## Training and optimizing after getting a best parameters using optuna implementation

# In[ ]:


# import lightgbm as lgb
# dtrain = lgb.Dataset(X, label=y)

# # We will track how many training rounds we needed for our best score.
# # We will use that number of rounds later.
# best_score = 999
# training_rounds = 10000

# # Declare how we evaluate how good a set of hyperparameters are, i.e.
# # declare an objective function.
# def objective(trial):
#     # Specify a search space using distributions across plausible values of hyperparameters.
#     param = {
#         "objective": "binary",
#         "metric": "binary_error",
#         "verbosity": -1,
#         "boosting_type": "gbdt",                
#         "seed": 42,
#         'learning_rate':0.1,
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'num_leaves': trial.suggest_int('num_leaves', 2, 512),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
#         'seed': 1979,
#         'feature_pre_filter':False
#     }
    
#     # Run LightGBM for the hyperparameter values
#     lgbcv = lgb.cv(param,
#                    dtrain,
# #                    categorical_feature=ids_of_categorical,
#                    folds=rkf,
#                    verbose_eval=False,                   
#                    early_stopping_rounds=250,                   
#                    num_boost_round=10000  
# #                    callbacks=[lgb.reset_parameter(learning_rate = [0.005]*200 + [0.001]*9800) ]
#                   )
    
#     cv_score = lgbcv['binary_error-mean'][-1] + lgbcv['binary_error-stdv'][-1]
#     if cv_score<best_score:
#         training_rounds = len( list(lgbcv.values())[0] )
    
#     # Return metric of interest
#     return cv_score

# # Suppress information only outputs - otherwise optuna is 
# # quite verbose, which can be nice, but takes up a lot of space
# optuna.logging.set_verbosity(optuna.logging.WARNING) 

# # We search for another 4 hours (3600 s are an hours, so timeout=14400).
# # We could instead do e.g. n_trials=1000, to try 1000 hyperparameters chosen 
# # by optuna or set neither timeout or n_trials so that we keep going until 
# # the user interrupts ("Cancel run").
# study = optuna.create_study(direction='minimize')  
# study.enqueue_trial(tmp_best_params)
# study.optimize(objective, timeout=14400) 


# # Hyperparameter Visualisation

# In[ ]:


# optuna.visualization.plot_optimization_history(study)


# In[ ]:


# optuna.visualization.plot_parallel_coordinate(study)


# In[ ]:


# optuna.visualization.plot_param_importances(study)


# In[ ]:


# optuna.visualization.plot_slice(study)


# In[ ]:


# print(study.best_params)


# {'lambda_l1': 0.3216393371762933, 'lambda_l2': 0.020172055755509347, 'num_leaves': 501, 'feature_fraction': 0.7259695210423447, 'bagging_fraction': 0.9563525530698169, 'bagging_freq': 1, 'min_child_samples': 15}
# 

# In[ ]:


# # Classification error
# print(study.best_value)
# # Or expressed as accuracy
# print(1.0-study.best_value)


# In[ ]:





# ## Trying to improve the score

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import lightgbm as lgb\n# dtrain = lgb.Dataset(X_train, label=y_train)\ndtrain = lgb.Dataset(os_X, label=os_y)\n\nrkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)\n\n# We will track how many training rounds we needed for our best score.\n# We will use that number of rounds later.\nbest_score = 99999\ntraining_rounds = 10000\n\n# Declare how we evaluate how good a set of hyperparameters are, i.e.\n# declare an objective function.\n\ndef objective(trial):\n    # Specify a search space using distributions across plausible values of hyperparameters.\n    param = {\n        "objective": "binary",\n        "metric": "binary_logloss",#using similar to baseline\n        "verbosity": -1,\n        \'learning_rate\':0.01,\n        "boosting_type": "gbdt",   \n        \'num_boost_round\': trial.suggest_int(\'num_boost_round\', 1, 10000),\n        \'max_depth\': trial.suggest_int(\'max_depth\', 1, 200),\n        \'lambda_l1\': trial.suggest_loguniform(\'lambda_l1\', 1e-8, 100.0),\n        \'lambda_l2\': trial.suggest_loguniform(\'lambda_l2\', 1e-8, 100.0),\n        \'num_leaves\': trial.suggest_int(\'num_leaves\', 2, 256),\n        \'feature_fraction\': trial.suggest_uniform(\'feature_fraction\', 0.1, 1.0),\n        \'bagging_fraction\': trial.suggest_uniform(\'bagging_fraction\', 0.1, 1.0),\n        \'bagging_freq\': trial.suggest_int(\'bagging_freq\', 0, 15),\n        \'min_child_samples\': trial.suggest_int(\'min_child_samples\', 1, 100),\n        \'seed\': 1979,\n        \'feature_pre_filter\':False\n    }\n    \n    # Run LightGBM for the hyperparameter values\n    lgbcv = lgb.cv(param,\n                   dtrain,\n#                    categorical_feature=ids_of_categorical,\n                   folds=rkf,\n                   verbose_eval=False,    \n                   early_stopping_rounds=250,                   \n                  )\n    \n#     cv_score = lgbcv[\'binary_error-mean\'][-1] + lgbcv[\'binary_error-stdv\'][-1]\n#     if cv_score<best_score:\n#         training_rounds = len( list(lgbcv.values())[0] )\n\n# Using similar optimisation\n      # Return the minimum log-loss found from cv\n    cv_score = pd.Series(lgbcv[\'binary_logloss-mean\']).min()\n    \n#     # The index of the minimum log-loss value is the number of trees\n    optimal_trees = pd.Series(lgbcv[\'binary_logloss-mean\']).idxmin()\n\n#     cv_score = lgbcv[\'binary_logloss-mean\'][-1] + lgbcv[\'binary_logloss-stdv\'][-1]\n    if cv_score<best_score:\n        training_rounds = len( list(lgbcv.values())[0] )\n    \n#     # Add the number of trees to the Trial stage\n#     trial.set_user_attr("optimal_trees", optimal_trees)\n    \n    # Return metric of interest\n    return cv_score')


# In[ ]:


# Suppress information only outputs - otherwise optuna is 
# quite verbose, which can be nice, but takes up a lot of space
# optuna.logging.set_verbosity(optuna.logging.WARNING) 

# We search for another 4 hours (3600 s are an hours, so timeout=14400).
# We could instead do e.g. n_trials=1000, to try 1000 hyperparameters chosen 
# by optuna or set neither timeout or n_trials so that we keep going until 
# the user interrupts ("Cancel run").
study1 = optuna.create_study(direction='minimize')  
study1.enqueue_trial(tmp_best_params)

# We search for another 3 hours (3600 s are an hours, so timeout=10800).

study1.optimize(objective, timeout=10800)
# study1.optimize(objective, n_trials=300)


# In[ ]:


# Print optimal trees
# print('optimal number of trees: ', study1.best_trial.user_attrs['optimal_trees'])

print('best logloss: ', study1.best_trial.values[0])

# Print best params
study1.best_trial.params


# WITH SMOTE: 
# 
# {'num_boost_round': 5385,
#  'max_depth': 95,
#  'lambda_l1': 0.0016864295177581943,
#  'lambda_l2': 0.01064782356317584,
#  'num_leaves': 192,
#  'feature_fraction': 0.489464930792113,
#  'bagging_fraction': 0.5723988870610315,
#  'bagging_freq': 8,
#  'min_child_samples': 1}
#  
#  
#  best logloss:  0.38668516640141903
# 

# optimal number of trees:  32
# best logloss:  0.5821837794857733
# 
# 
# 
# {'lambda_l1': 0.027557048144281192,
#  'lambda_l2': 6.733793502288289,
#  'num_leaves': 300,
#  'feature_fraction': 0.9995093343010514,
#  'bagging_fraction': 0.6997723787350977,
#  'bagging_freq': 1,
#  'min_child_samples': 3}

# In[ ]:


optuna.visualization.plot_param_importances(study1)


# In[ ]:


optuna.visualization.plot_parallel_coordinate(study1)


# 'num_boost_round': 5385,
#  'max_depth': 95,
#  'lambda_l1': 0.0016864295177581943,
#  'lambda_l2': 0.01064782356317584,
#  'num_leaves': 192,
#  'feature_fraction': 0.489464930792113,
#  'bagging_fraction': 0.5723988870610315,
#  'bagging_freq': 8,
#  'min_child_samples': 1

# In[ ]:


# Retrain model on all of X_train with optimal parameters to return a model object

dtrain = lgb.Dataset(os_X, label=os_y)

param = {
    "objective": "binary",
    "metric": "binary_error", #binary_logloss
    "verbosity": -1,
    "boosting_type": "gbdt",                
    "seed": 42,
    'lambda_l1': study1.best_trial.params['lambda_l1'],
    'lambda_l2': study1.best_trial.params['lambda_l2'],
    'num_leaves': study1.best_trial.params['num_leaves'],
    'num_boost_round': study1.best_trial.params['num_boost_round'],
    'max_depth': study1.best_trial.params['max_depth'],
    'feature_fraction': study1.best_trial.params['feature_fraction'],
    'bagging_fraction': study1.best_trial.params['bagging_fraction'],
    'bagging_freq': study1.best_trial.params['bagging_freq'],
    'min_child_samples': study1.best_trial.params['min_child_samples'],
    'learning_rate':0.1,
    'feature_pre_filter':'false'
}
    
clf = lgb.train(train_set=dtrain,
          params=param,
#           num_boost_round=study1.best_trial.user_attrs['optimal_trees']
         )


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import log_loss, accuracy_score


test_metric = log_loss(y_test, y_pred) #log_loss
cross_val_metric = study1.best_trial.values[0]

print('Log-loss from cross validation on training data: ', cross_val_metric)
print('Log-loss from unseen test data: ', test_metric)


# In[ ]:


# SMOTE Results in major overfitting!!!!!

