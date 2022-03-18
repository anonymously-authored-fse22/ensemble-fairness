#!/usr/bin/env python
# coding: utf-8

# # Adult Income Prediction and Data Exploration

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import urllib

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./datasets/AdultIncome/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# We upload our dataset

# In[ ]:


adult_income = pd.read_csv("../input/adult-census-income/adult.csv")


# We have a quick look at the table:

# In[ ]:


adult_income.head()


# At first sight, the table seems to have null values. The `education.num` and `education` fields are the same, one is categorical and the other is numerical. Let's have a statistical look at the numerical values.

# In[ ]:


adult_income.describe()


# There might be some outliers in all numerical values. 

# In[ ]:


adult_income.head()


# ### What do we want to predict?

# Our main goal is to predict if a person, given some certain features, has a high salary or not (A salary is considered high if it's above 50,000$ per year). This is contained in the `income` target

# In[ ]:





# ### Exploring null values

# In[ ]:


adult_income = adult_income.replace('?', np.NaN)


# In[ ]:


adult_income.isna().sum()


# As we observe, `workclass`, `occupation` and `native.country`.

# #### workclass

# The `workclass` feature is categorical. So we'll replace the null values setting the label `Unknown`.

# In[ ]:


adult_income['workclass'] = adult_income['workclass'].replace(np.NaN, 'Unknown')


# In[ ]:


adult_income['workclass'].isna().sum()


# In[ ]:


adult_income[adult_income['workclass'] == 'Unknown']['workclass'].count()


# #### occupation

# The `occupation` feature is categorical. So we'll replace the null values setting the label `Other`.

# In[ ]:


adult_income['occupation'] = adult_income['occupation'].replace(np.NaN, 'Other')


# In[ ]:


adult_income[adult_income['occupation'] == 'Other']['occupation'].count()


# #### Native Country

# The `native.country` feature is categorical. So we'll also replace the null values setting the label `Other`.

# In[ ]:


adult_income['native.country'] = adult_income['native.country'].replace(np.NaN, 'Other')


# In[ ]:


adult_income[adult_income['native.country'] == 'Other']['native.country'].count()


# Now there are no null values

# In[ ]:


adult_income.isna().sum()


# ### Auxiliar functions

# Before analyzing and exploring our dataset, We will create a auxiliar function to plot charts with certain parameters. 

# In[ ]:


from matplotlib.ticker import FuncFormatter


def plot_features_income(data, column, type_names, size=(20, 10)):
    fig, ax = plt.subplots(figsize=size)
    barWidth = 0.25
    bars1 = list()
    bars2 = list()
    for col in type_names:
        dt = data[data[column] == col]
        count_up = dt[dt['income'] == '>50K']['income'].count()
        count_down = dt[dt['income'] == '<=50K']['income'].count()
        bars1.append(count_up)
        bars2.append(count_down)
    
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1] 
    
    rects1 = plt.bar(r1, bars1, color='gold', width=barWidth, edgecolor='white', label='More than 50K $')
    rects2 = plt.bar(r2, bars2, color='tomato', width=barWidth, edgecolor='white', label='Less or Equal than 50K $')
    
    plt.xlabel(column, fontweight='bold')
    plt.ylabel('Income per number of people', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], type_names, rotation=30)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.4)
    
    heights_1 = list()
    for rect in rects1:
        height = rect.get_height()
        heights_1.append(height)
        
    heights_2 = list()
    for rect in rects2:
        height = rect.get_height()
        heights_2.append(height)
    
    count = 0
    for rect in rects1:
        h1 = heights_1[count]
        h2 = heights_2[count]
        ptg = (h1 / (h1 + h2)) * 100
        ax.text(rect.get_x() + rect.get_width()/2., 0.99*h1,
            '%d' % int(ptg) + "%", ha='center')
        count = count + 1
    
    count = 0
    for rect in rects2:
        h1 = heights_1[count]
        h2 = heights_2[count]
        ptg = (h2 / (h1 + h2)) * 100
        ax.text(rect.get_x() + rect.get_width()/2., h2,
            '%d' % int(ptg) + "%", ha='center', va='bottom')
        count = count + 1    
        
    
    plt.tight_layout()
    plt.legend()
    plt.show()
    


# In[ ]:





# # Data Exploration

# In[ ]:


adult_income.dtypes


# ## Categorical features

# We will first analyze our categorical features.

# ### workclass

# The `workclass` feature represents the kind of profession a person has. Let's see the relation between this feature and the `income` feature.

# In[ ]:


workclass_types = adult_income.workclass.unique()


# In[ ]:


workclass_types


# In[ ]:


plt.figure(figsize=(8, 8))
adult_income['workclass'].value_counts().plot.pie(autopct='%1.1f%%')


# We see that **60%** of people registered in the census work in the private sector. The rest is distribuited among between self-employement and public sector. We have a **5.6%** of jobs that are unknown. Now we we'll have a look at people earning more thatn 50,000$ depending on workclass.

# In[ ]:



plot_features_income(data=adult_income, column='workclass', type_names=workclass_types)


# For every workclass, except self-employement, there are more people earning below 50,000\\$ than people earning more than 50,000\\$. 
# Private sector holds most of the jobs, having the majority of them a salary below 50,000$. Now let's have a closer look high paid and non-high paid jobs.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['workclass'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title(' More Than 50K $ per year according to workclass')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['workclass'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to workclass')
ax[1].set_ylabel('')


# The observations in the **high salary chart** we draw is:
# * 63.3% of high paid jobs can be found in the private sector
# * 17.1% are self employed jobs
# * 2.4% are Unknown jobs
# * The rest are Goverment or civil servant jobs

# The observations in the **low salary chart** we draw is:
# * Most of the salaries under 50,000$ are in the private sector.
# * The rest of percentages are similar to the ones in the high salary sector.

# ### Education

# Let's have a look at the education feature. 

# In[ ]:


plt.figure(figsize=(10, 10))
adult_income['education'].value_counts().plot.pie(autopct='%1.1f%%')


# We see that people's education scale in the census is very distributed.

# In[ ]:


plt.figure(figsize=(20, 10))
education_types = adult_income.education.unique()
plot_features_income(data=adult_income, column='education', type_names=education_types)


# The charts plot some expectable information. We can see that most people who are school professors and most people holding a Master degree or a PhD earn more than 50,000\\$ per year. It's interesting that the 41\% of people owning a bachelor's degree tend to earn more than 50,000\\$ a year. The observations we can draw here is that people who went to college and have professional degree tend to earn more than 50,000\\$ per year. 

# Now, if we look at the charts below, among people earning more than 50,000\\$ grouped by education we can see that half of the people have, at least, a college degree or are high school graduates (HS-grad). On the other hand, the other pie chart presents a similar distribuition but, as we saw in the previous charts, we can see that people earning a Master degree or a PhD tend to earn more than 50,000\\$.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['education'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('More Than 50K $ per year according to education')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['education'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to education')
ax[1].set_ylabel('')


# 

# In[ ]:





# ### Marital status

# In[ ]:


plt.figure(figsize=(10, 10))
adult_income['marital.status'].value_counts().plot.pie(autopct='%1.1f%%')


# The 46% of the people in the census are married, the 32% is single and the 13.6% is divorced.

# 

# In[ ]:


plt.figure(figsize=(20, 10))
marital_types = adult_income['marital.status'].unique()
plot_features_income(data=adult_income, column='marital.status', type_names=marital_types)


# This is a very telling chart. As we can see, almost half of people who are married earn more than 50,000\\$, most people who are separated, divorced or single earn less than 50,000\\$. Now let's separate the groups by people who earn more than 50,000\\$ and less than 50,000\\$.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['marital.status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('More Than 50K $ per year according to marital status')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['marital.status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to marital status')
ax[1].set_ylabel('')


# Most people earning more than 50,000\\$ are married in a 85%, while they only represent a 33.5% of people earning less than 50,000\\$. A very interesing fact is that people who earn less than 50,000\\$ are either single or divorced, in other words, don't have partner.

# ### occupation

# We are taking a look at what kind of jobs have influence on salaries.

# In[ ]:


plt.figure(figsize=(8, 8))
adult_income['occupation'].value_counts().plot.pie(autopct='%1.1f%%')


# In[ ]:


plt.figure(figsize=(20, 10))
occupation_types = adult_income['occupation'].unique()
plot_features_income(data=adult_income, column='occupation', type_names=occupation_types)


# 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['occupation'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('More Than 50K $ per year according to occupation fields')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['occupation'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to occupation fields')
ax[1].set_ylabel('')


# We can see that most well paid jobs are related to Executive Managers, specialized preoffesors, techology engineers and protection services.

# ### Relationship

# In[ ]:


plt.figure(figsize=(8, 8))
adult_income['relationship'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')


# In[ ]:


plt.figure(figsize=(20, 10))
relationships_types = adult_income['relationship'].unique()
plot_features_income(data=adult_income, column='relationship', type_names=relationships_types)


# An interesting fact is that 44% of people earning more than 50,000\\$ are married men, but it's even more interesting that the percentage of married women earning 50,000\\$ is slightly higher. Let's divide the information by groups of people who earn more and less than 50,000\\$.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['relationship'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('More Than 50K $ per year according to relationship status')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['relationship'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to relationship status')
ax[1].set_ylabel('')


# The pie charts show that, in general, most of people earning more than 50,000$ are married men. On the other pie charts the information is much more distribuited.  

# ### Race

# In[ ]:


plt.figure(figsize=(8, 8))
adult_income['race'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')


# In[ ]:


plt.figure(figsize=(20, 10))
race_types = adult_income['race'].unique()
plot_features_income(data=adult_income, column='race', type_names=race_types)


# 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['race'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('More Than 50K $ per year according to race')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['race'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to race')
ax[1].set_ylabel('')


# Statistically, there are more asians and whites earning more than 50,000$ than other races. 

# ### Sex

# In[ ]:


plt.figure(figsize=(18, 8))
sns.countplot(adult_income['sex'], order = ['Male', 'Female'])


# The census registers more men than women. 

# In[ ]:


plt.figure(figsize=(20, 10))
race_types = adult_income['sex'].unique()
plot_features_income(data=adult_income, column='sex', type_names=race_types)


# The chart show that 30% of men earn more than 50,000\\$ while only 10% of women surpass that amount. In other words, there are 200% more men than women earning above 50,000 \\$.

# ### Native Country

# In[ ]:


plt.figure(figsize=(20, 10))
country_types = adult_income['native.country'].unique()
plot_features_income(data=adult_income, column='native.country', type_names=country_types)


# In[ ]:


plt.figure(figsize=(20, 10))
country_types = ['Mexico', 'Philippines', 'Germany', 'El-Salvador']
plot_features_income(data=adult_income, column='native.country', type_names=country_types)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
adult_income[adult_income['income'] == '>50K']['native.country'].value_counts().plot.pie(autopct='', ax=ax[0])
ax[0].set_title('More Than 50K $ per year according to nationality')
ax[0].set_ylabel('')
adult_income[adult_income['income'] == '<=50K']['native.country'].value_counts().plot.pie(autopct='', ax=ax[1])
ax[1].set_title('Less or equal Than 50K $ per year according to nationality')
ax[1].set_ylabel('')


# In[ ]:





# ## Numerical Analysis

# ### Age

# Now we'll take a lot at the age distribuition of the census. 

# In[ ]:


plt.figure(figsize=(20,10))
plt.grid()
sns.distplot(adult_income['age'])


# The age distribuition collected in the census is concentrated among from 20 y/o to the 50 y/o interval. 

# In[ ]:


plt.figure(figsize=(20, 10))
plt.xticks(rotation=45)
sns.countplot(adult_income['age'], hue=adult_income['income'], palette=['dodgerblue', 'gold'])


# This is very interesting plot. As age grows, there are more people earning more than 50,000\\$, so we can say that, generally, income is correllated to age. 

# ### Hours per week

# In[ ]:


plt.figure(figsize=(20,10))
plt.grid()
sns.distplot(adult_income['hours.per.week'])


# The plot shows that most people in the census work 40 hours per week. Now, we'd like to know the hours per week distribuition of the people earning more than 50,000\\$. 

# Normally, people who earn more than 50,000\\$ per year have a 40 hours/week rutine. There are also a lot working for 45, 50 and 60 hours/week. 

# # Multivariable analysis

# After analysing each variable, we will apply a multivariable analysis combining several variables and correlations.

# ## correlations

# In[ ]:


numerical_dt = list(adult_income.select_dtypes(include=['float64', 'int64']).columns)


# In[ ]:


numerical_dt


# In[ ]:


numerical_dt = np.asarray(numerical_dt)


# In[ ]:


numerical_dt


# In[ ]:


num_dt = adult_income.loc[:, numerical_dt]


# In[ ]:


num_dt = num_dt.drop(columns='education.num')


# In[ ]:


cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr_matrix.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix, 
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns)


# The hitmap shows no evident high correlation cases among the numerical variables.

# ## Analysis based on gender and age

# After analyzing each of every feature we realized men to earn more than women, so we decided execute a better analysis on this field, so that we can draw some useful informations.

# ### Gender and workclass

# We're going to have a look at the relations between gender and workclass and occupations, and what kind of jobs women mostly occupy in the census. 

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(20, 10))
plt.figure(figsize=(20, 10))
sns.countplot(adult_income['workclass'], hue=adult_income['sex'], ax=axs[1], palette=['pink', 'dodgerblue'], order=adult_income[adult_income['sex'] == 'Female']['workclass'].value_counts().index)
sns.countplot(adult_income['occupation'], hue=adult_income['sex'], ax=axs[0], palette=['pink', 'dodgerblue'], order=adult_income[adult_income['sex'] == 'Female']['occupation'].value_counts().index)
plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)
plt.show()


# Most women occupy the jobs related to clerical administration, cleaning services and other services, but jobs related to professor speciality, business and sales, engineering, technology, transport, protection service and primary sector are mostly occupied by men. It's also interesting to see that most gender gap in private sector and self employement is bigger than in other sectors.

# ### Gender, Hours per week and Income

# Let's see if there's any relationship between hours per week and income divided by gender. 

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(25, 8))
plt.xticks(rotation=45)
sns.violinplot(adult_income['sex'], adult_income['hours.per.week'], hue=adult_income['income'], palette=['gold', 'dodgerblue'], ax=ax[0])
sns.stripplot(adult_income['sex'], adult_income['hours.per.week'], hue=adult_income['income'], palette=['skyblue', 'tomato'], ax=ax[1])
ax[0].grid(True)
ax[1].grid(True)


# The charts show that men work more for hours than women. The left chart show that, regardless of the income, there are more women working for less than men and the men chart is more distribuited above 40 hours per week. The right chart shows that men working more hours tend to earn more than 50,000\\$. We see a concentration of red dots among the 40 and 60 hours/week interval. On the other hand, this concentration doesn't appear women side. Even though the hours per week gap between men and women is not so big, it's clear that there's no correlation between hours per week and income when it comes to women. 

# ### Age, gender and Hours per week

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(30, 8))
plt.xticks(rotation=45)
sns.lineplot(adult_income['age'], adult_income['hours.per.week'], hue=adult_income['income'], palette=['tomato', 'dodgerblue'], ax=ax[0])
sns.lineplot(adult_income['age'], adult_income['hours.per.week'], hue=adult_income['sex'], palette=['tomato', 'dodgerblue'], ax=ax[1])
ax[0].grid(True)
ax[0].title.set_text("Age and Hours per week divided by Income")
ax[1].grid(True)
ax[0].title.set_text("Age and Hours per week divided by Gender")


# We see a very interesting trend in chart above. Let's take a loot at the left chart first. As the age grows, there are more people earning more than 50,000\\$ but work for more hours. In both cases, as age reaches the 60 year old, people tend to work for less hours but the number of people earning more than 50K increases. What's funny is that people who earn a lot start working for more hours when as they start turning 80.
# 
# The right chart shows very similar line paths. Men tend work for more hours than women, but as they get closer the standard retirement age, men and women work for the similar number of hours. What's very bizare, is that women who are 80 and 90 are the one working for more hours than the rest of ages. 

# ## Final observations and Conclusion after the Data Exploration

# We analyzed and explored all the features of the dataset and their particularites, we want to summerise all the interesting facts we discovered and could help us predict whether a person earns more or less than 50,000 \\$. The interesting observations we drew are:
# * **Workclas and occupations**
#     * The 55\% of self employed people work are self-employed
#     * The 63.3\% of the total people in the census earning more than 50,000\\$ work in the private sector and the 71\% of the total people in the census earning under 50,000\\$ work in the private sector too. 
#     * If we focus only in the private sector, the 26\% earn more than 50,000\\$.
#     * The jobs were we can find more people earning above 50,000\\$ are executive managers, protection services, college professors, engineering and jobs related to technology who are mostly occupied by men.
# 
# 
# * **Education**
#     * It's interesting that the 73\% of the Professors, 74\% of PhDs, the 55\% of people owning a Master Degree and the 40\% of Bachelors bachelors earn above 50,000\\$.
#     * We this information we can conclude that owning at least a college degree will increase your probabilities to earn 50,000 \\$/year.
#     
#     
# * **Gender, Marital Status and relationship**
#     * The 85% of total people in the census earning more than 50,000\\$ are married.
#     * The 44\% of people who are married earn more than 50,000\\$.
#     * The 44\% of husbands earen more than 50,000\\$.
#     * The 47\% of Wifes earn more than 50,000\\$.
#     * According to this info, being maried increases the probability of earning above 50,000\\$.
#     
#     
# * **Other interesting information**
#     * The salary is directly related to age. The older people get, the more the surpass the 50,000\\$ line.
#     * Men work for more hourse than women in all ages but as they both get closer to the 60's they tend to work for similiar amount of hours per week.
#     * People earning more than 50,000\\$ per year tend to work for more hours too.
#     * Men working for more than 40 hours per week tend to earn above 50,000\\$ but women don't follow this trend and there's no correlation between hours per week and income when it comes to females. 
# 
# 
# * So we could say that a person who's likely to earn above 50.000\\$/year is a person who:
#     * Is male whose age is between 30 or over.
#     * Married
#     * Whose job is related to bussines, engineering, college profesor, protection services, technical or IT field.
#     * Holds a master degree or a Phd.
#     * Works for more than 40 hours per week.
#     * Is American, Asian or European.

# 

# ## Data Cleaning and Formatting

# Now that we've performed out data exploration and have drawn some assumptions, it's time to clean the data, format it and erase those rows and columns who are useless or could noise during our learning process. 

# In[ ]:


adult_income_prep = adult_income.copy()


# ### Outliers anomaly

# Outliers can be very harmful for our learning models and can cause noise that can create distorsions in our predictions. We'll create an auxiliar function to erase the outliers in each numerical feature.

# In[ ]:


def treat_outliers(data, column, upper=False, lower=False):
    Q1=adult_income_prep[column].quantile(0.25)
    Q3=adult_income_prep[column].quantile(0.75)
    IQR=Q3-Q1
    print(Q1)
    print(Q3)
    print(IQR)
    U_threshold = Q3+1.5*IQR
    #print(L_threshold, U_threshold)
    if upper: 
        adult_income_prep[column] = adult_income_prep[adult_income_prep[column] < U_threshold]
    if lower:
        adult_income_prep[column] = adult_income_prep[adult_income_prep[column] >= U_threshold]


# #### Checking outliers in the age feature

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['age'])
plt.grid()


# We found outliers in our chart, so we'll erase them.

# In[ ]:


treat_outliers(data=adult_income_prep, column='age', upper=True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['age'])
plt.grid()


# In[ ]:


treat_outliers(data=adult_income_prep, column='age', upper=True)


# Let's check out now

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['age'])
plt.grid()


# There are still to rows which age column contains an outlier.

# In[ ]:


treat_outliers(data=adult_income_prep, column='age', upper=True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['age'])
plt.grid()


# Now it's OK.

# In[ ]:





# #### Removing outliers of final Weight 

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['fnlwgt'])
plt.grid()


# In[ ]:


treat_outliers(data=adult_income_prep, column='fnlwgt', upper=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['fnlwgt'])
plt.grid()


# #### Checking outliers in Capital Gain and Loss

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['capital.gain'])
plt.grid()


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['capital.loss'])
plt.grid()


# We realize `capital.gain` and `capital.loss` will disturb our learning process as they don't give any useful information either.

# In[ ]:


adult_income_prep = adult_income_prep.drop(columns=['capital.gain', 'capital.loss'])


# In[ ]:





# #### Checking outliers of Hours per week

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['hours.per.week'])
plt.grid()


# There are outliers, we must remove them.

# In[ ]:


treat_outliers(data=adult_income_prep, column='hours.per.week', upper=True, lower=True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=adult_income_prep, x=adult_income_prep['hours.per.week'])
plt.grid()


# Now it's alright. Let's see how our dataset is now.

# In[ ]:


adult_income_prep.head()


# We found new null values in the `age` and `fnlwgt` column. We have to fill it the median value.

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[ ]:


adult_income_num = adult_income_prep[['age', 'fnlwgt', 'hours.per.week']]


# In[ ]:


adult_income_num.head()


# In[ ]:


imputer.fit(adult_income_num)


# In[ ]:


imputer.statistics_


# In[ ]:


X = imputer.transform(adult_income_num)


# In[ ]:


X


# In[ ]:


adult_tr = pd.DataFrame(X, columns=adult_income_num.columns)


# In[ ]:


adult_tr


# In[ ]:


adult_income_prep['age'] = adult_tr['age']
adult_income_prep['fnlwgt'] = adult_tr['fnlwgt']
adult_income_prep['hours.per.week'] = adult_tr['hours.per.week']


# In[ ]:


adult_income_prep.head()


# Alright, no null values now. Now let's change the income values by 1 and 0.

# In[ ]:


adult_income_prep['income'] = adult_income_prep['income'].replace('<=50K', 0)
adult_income_prep['income'] = adult_income_prep['income'].replace('>50K', 1)


# We'll erase the `education` feature because it's the same as `education.num`.

# In[ ]:


adult_income_prep = adult_income_prep.drop(columns='education')


# #### Category Encoding

# During our learning process, we can use non-numerical values, so it's better to encode our non-numerical features.

# In[ ]:


adult_income_prep.workclass = adult_income_prep.workclass.astype('category').cat.codes
adult_income_prep['marital.status'] = adult_income_prep['marital.status'].astype('category').cat.codes
adult_income_prep['occupation'] = adult_income_prep['occupation'].astype('category').cat.codes
adult_income_prep['relationship'] = adult_income_prep['relationship'].astype('category').cat.codes
adult_income_prep['race'] = adult_income_prep['race'].astype('category').cat.codes
adult_income_prep['sex'] = adult_income_prep['sex'].astype('category').cat.codes
adult_income_prep['native.country'] = adult_income_prep['native.country'].astype('category').cat.codes


# In[ ]:


adult_income_prep.head()


# Now our dataset is ready for training.

# # Training and Comparing models

# In[ ]:


np.random.seed(1234)


# We prepare out dataset and divide it into subsets.

# In[ ]:


y = adult_income_prep['income']
X_prepared = adult_income_prep.drop(columns='income')


# We import the `sklearn` library we need to partition the dataset into training and testing subsets.

# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X_prepared, y, random_state = 0)


# We will use a crossvalidation to search for the best hyperparameters.

# In[ ]:


from sklearn.model_selection import cross_val_score


# We'll have to dictionaries containing the Mean Absolute Error and the accuracy value of each algorithm.

# In[ ]:


MAE = dict()
Acc = dict()


# ## Traditional ML Techniques: Logistic regression

# We will perform a crossvalidated logistic regression to our dataset. From the Logistic Regression we will extract the coeficients/features who have a better or a worse influence on the prediction.

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log_model = LogisticRegression()


# In[ ]:


score = cross_val_score(log_model, X_prepared, y, scoring="neg_mean_absolute_error", cv=10)


# In[ ]:


print("MAE score mean:\n", np.abs(score).mean())


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = [
    {'C': [0.001,0.01,0.1,1,10,100]},
]
grid_search = GridSearchCV(log_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_X, train_y)


# In[ ]:


grid_search.best_params_


# In[ ]:


log_model = LogisticRegression(C=100, random_state=0)


# In[ ]:


log_model.fit(train_X, train_y)


# In[ ]:


val_predictions = log_model.predict(val_X)


# In[ ]:


columns = adult_income_prep.drop(columns='income').columns
coefs = log_model.coef_[0]
print("Features - Coefs")
for index in range(len(coefs)):
    print(columns[index], ":", coefs[index])
    


# It's pretty interesting to see what the logistic regression reveals.
# * Education, relationship, gender and race are the features which most positively have an impact on income
# * The hours per week and the final weight have a negative impact on income

# Now, let's calculate the mean absolute error (MAE).

# In[ ]:


from sklearn.metrics import mean_absolute_error
lm_mae = mean_absolute_error(val_y, val_predictions)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


lm_acc = accuracy_score(val_y, val_predictions)
MAE['lm'] = lm_mae
Acc['lm'] = lm_acc


# In[ ]:


print("The mae is", lm_mae)


# In[ ]:


print("The accuracy is", lm_acc * 100, "%")


# ## Modern ML techniques

# We've performed a training and testing process using a traditional ML technique which was the Logistic Regression. Now, we'll use some modern classifers which are:
# * Random Forests
# * K Nearest Neighbours
# * Gradient Boosting Machine
# * Naive Bayes
# 
# For all of them we'll perform a crossvaliation to detect the best hyperparameters. 

# ### Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_model = RandomForestClassifier()
grid_search = GridSearchCV(forest_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_X, train_y)


# In[ ]:


grid_search.best_params_


# In[ ]:


rf_model = RandomForestClassifier(max_features=2, n_estimators=30, random_state=0)


# In[ ]:


rf_model.fit(train_X, train_y)


# In[ ]:


val_predictions = rf_model.predict(val_X)


# In[ ]:


rf_mae = mean_absolute_error(val_y, val_predictions)


# In[ ]:


rf_mae


# In[ ]:


rf_acc = accuracy_score(val_y, val_predictions)


# In[ ]:


rf_acc


# In[ ]:


MAE['rf'] = rf_mae
Acc['rf'] = rf_acc


# ### Gradient Boosting Machine

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbm_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=0)

param_grid = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}

grid_search = GridSearchCV(gbm_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_X, train_y)


# In[ ]:


grid_search.best_params_


# In[ ]:


gbm_model = GradientBoostingClassifier(max_depth=7, min_samples_split=800, random_state=0)


# In[ ]:


gbm_mae = mean_absolute_error(val_y, val_predictions)


# In[ ]:


gbm_mae


# In[ ]:


gbm_acc = accuracy_score(val_y, val_predictions)


# In[ ]:


gbm_acc


# In[ ]:


MAE['gbm'] = gbm_mae
Acc['gbm'] = gbm_acc


# ### K-Nearest Neighbours

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as KNN


# In[ ]:


KNN


# In[ ]:


knn_model = KNN()

param_grid = {'n_neighbors':range(5,10,1)}

grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_X, train_y)


# In[ ]:


knn_params = grid_search.best_params_
knn_params


# In[ ]:


knn_model = KNN(n_neighbors=8)


# In[ ]:


knn_model.fit(train_X, train_y)


# In[ ]:


val_predictions = knn_model.predict(val_X)


# In[ ]:


knn_mae = mean_absolute_error(val_y, val_predictions)


# In[ ]:


knn_mae


# In[ ]:


knn_acc = accuracy_score(val_y, val_predictions)


# In[ ]:


knn_acc


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


GNB = GaussianNB()


# In[ ]:


GNB.fit(train_X, train_y)


# In[ ]:


val_predictions = GNB.predict(val_X)


# In[ ]:


GNB_mae = mean_absolute_error(val_y, val_predictions)


# In[ ]:


GNB_mae


# In[ ]:


GNB_acc = accuracy_score(val_y, val_predictions)


# In[ ]:


GNB_acc


# In[ ]:


MAE['gnb'] = GNB_mae
Acc['gnb'] = GNB_acc


# In[ ]:


MAE['knn'] = knn_mae
Acc['knn'] = knn_acc


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
ax[0].plot(list(MAE.keys()), list(MAE.values()))
ax[0].set_title("Mean Absolute Error")
ax[0].grid()
ax[1].bar(list(Acc.keys()), list(Acc.values()))
ax[1].set_title("Accuracy score")


# Apparently the Random Forest Classifier is the best compared to the rest due the time Gradient Boosting needs to perform the training and testing with a 81.36% accuracy. 

# In[ ]:




