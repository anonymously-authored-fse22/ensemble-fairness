#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing

# The goal of this project is to correctly identify if a client will subscribe to a term deposit. Identified by the y column. 
# 
# There a lot of graphs as I wanted to have a clear visual of everything, some arent relevant at all and I will clean up later!

# Start by importing the relevant libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/bank-marketing/bank-additional-full.csv', sep = ';')


# In[ ]:


df.head()


# # Exploratory Data Analysis

# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.describe()


# For some numerical variables the median is far from the mean, this means that it is not normally distributed.

# # Bar plots

# In[ ]:


#barplot of job with bars split by education
#pivot table to get jobs as index and education as column. count of 'age' are the values
df_pivot = pd.pivot_table(df,columns='education', index='job', aggfunc='count',values='age')
df_pivot.plot(kind='bar',stacked=True, figsize=(15,10))
plt.title('Count of people in each job split by education level')
plt.ylabel('Count')
plt.xlabel('Job')


# In[ ]:


#works but not entirely clean will fix later
#pie chart function to see further breakdown of any column by anyother column
def pie_chart():
    while True:
        try:
            column = input('Which Column? \n{}'.format(df.select_dtypes('object').columns))
            cat = input('Which category? \n{}'.format(df[column].unique()))
            split_by = input('Which category do you want to split on? \n{}'.format(df.select_dtypes('object').columns.drop(column)))
            break
        except:
            print("column or category doesn't exist")
            continue
    if column in df.select_dtypes('object').columns:
        df_pivot = pd.pivot_table(df,columns = split_by, index = column, aggfunc='count',values='age')
        df_pivot = df_pivot.fillna(0)
        plt.figure(figsize=(10,10))
        plt.pie(df_pivot.loc[cat], labels = df_pivot.loc[cat].index, autopct = '%1.2f%%')
        plt.title('{}: The number of people in {} split by {}'.format(column,cat,split_by))
    else:
        print('Not Categorical column')
        column = input('Please choose from the columns \n{}'.format(df.select_dtypes('object').columns))


# In[ ]:


#bar chart of jobs by personal loan
pd.pivot_table(df,columns='loan', index='job', aggfunc='count',values='age').plot(kind='bar', stacked = True, figsize=(15,5))


# In[ ]:


#bar chart of jobs by yes or no
pd.pivot_table(df,columns='y', index='job', aggfunc='count',values='age').plot(kind='bar', stacked = True, figsize=(15,5))


# Pretty surprising that there is a small proportion of unemployed clients with personal loans compared to the other jobs. Was expecting unemployed to have a few personal loans

# In[ ]:


df_pivot


# In[ ]:


#no. of people with housing loan
sns.countplot(df['housing'], order = ['yes','no','unknown'])


# In[ ]:


#split by target variable
#pivot table.sorting by no column.plotting as bar chart
pd.pivot_table(df,columns='y', index='housing', aggfunc='count',values='age').sort_values(['no'], ascending=False).plot(kind='bar', stacked = True, figsize=(15,10) )


# Proportion of target variable doesn't change depending on housing loan or not

# In[ ]:


sns.countplot(df['loan'], order=['no','yes','unknown'])


# In[ ]:


pd.pivot_table(df,columns='y', index='loan', aggfunc='count',values='age').sort_values(['no'], ascending=False).plot(kind='bar', stacked = True,figsize=(15,10) )


# Personal loan does not seem to affect whether or not they subscribed to a term deposit

# In[ ]:


df_pivot_poutcome = pd.pivot_table(df,columns='y', index='poutcome', aggfunc='count',values='age')
df_pivot_poutcome.sort_values(['no'], ascending=False).plot(kind='bar', stacked = True, figsize=(15,10) )
plt.title('Count of outcome of the last campaign')


# In[ ]:


df_pivot_poutcome.loc['success']


# In[ ]:


success_perc = round(100 * df_pivot_poutcome.loc['success'][1]/sum(df_pivot_poutcome.loc['success']),1)
failure_perc = round(100 * df_pivot_poutcome.loc['failure'][1]/sum(df_pivot_poutcome.loc['failure']),1)
nonexistent_perc = round(100 * df_pivot_poutcome.loc['nonexistent'][1]/sum(df_pivot_poutcome.loc['nonexistent']),1)


# In[ ]:


print('For a successful last campaign {}% subscribed to a term deposit'.format(success_perc))
print('For a failed last campaign {}% subscribed to a term deposit'.format(failure_perc))
print('For a nonexistent last campaign {}% subscribed to a term deposit'.format(nonexistent_perc))


# Succesful last campaigns are important in determining whether they will subscribe to a term deposit. Where the outcome of the last campaign has been successful a higher proportion of people have subscribed to a term deposit

# In[ ]:


pd.pivot_table(df,columns='y', index='month', aggfunc='count',values='age').sort_values(['no'], ascending=False).plot(kind='bar', stacked = True,figsize=(15,10) )


# In[ ]:


months = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


# In[ ]:


def absolute_value(val,month):
    a = round(val*sum(df[df['month']==month].y.value_counts())/100,0)
    return a


# In[ ]:


fig, axs = plt.subplots(2,5,figsize=(20,8))
#creating dictionary of numbers and plot axes for the for loop
#making manually is probably easier to follow
plot_dict=dict(zip(range(0,10),[axs[0,0],axs[0,1],axs[0,2],axs[0,3],axs[0,4],
                     axs[1,0],axs[1,1], axs[1,2], axs[1,3], axs[1,4]]))

#enumerate returns tuple of month and a number(idx)
#change to autopct = '%1.2f%%' for percentage
for idx, month in enumerate(months):
    #gets axes from dict and plots on it
    plot_dict.get(idx).pie(df[df['month']==month].y.value_counts(),
             labels=df[df['month']==month].y.value_counts().index, autopct=lambda val: absolute_value(val,month))
    plot_dict.get(idx).set_title(month)


# Quite a big descrepancy in proportion of people subscribed to term deposits for certain last contact months. For these last contact months did anything significant happen? Why do a lot of people last contacted in these months subscribe to term deposits?

# Discrepancy could be explained by the low numbers

# Initially it looked like the last contacted month may have an affect on whether or not they subscribe to a term deposit but the low sample size for these months possibly explains the difference.

# In[ ]:


pd.pivot_table(df,columns='y', index='day_of_week', aggfunc='count',values='age').sort_values(['no'], ascending=False).plot(kind='bar', stacked = True,figsize=(15,10) )


# Day of week has little affect

# This analysis has shown that outcome of last campaign and last contact month of the year may have an affect whether or not they have subscribed to a term deposit.

# # Distribution of numerical variables

# In[ ]:


#histogram of age split by putput variable
df['age'].hist(by = df.y, bins = 30, figsize=(12,5))


# In[ ]:


df['duration'].hist(bins=100, figsize=(10,7))


# In[ ]:


df['duration'].hist(bins=100, figsize=(10,7), by=df['y'])


# In[ ]:


#histogram of contacts performed during campaign
df['campaign'].hist(figsize=(12,5), bins=30, by =df['y'])


# In[ ]:


df['nr.employed'].hist(figsize=(12,5), by=df['y'])


# May actually be ordinal categorical variable and not ratio

# In[ ]:


df['nr.employed'].unique()


# In[ ]:


df['emp.var.rate'].unique()


# In[ ]:


df['cons.conf.idx'].unique()


# In[ ]:


df['cons.price.idx'].unique()


# In[ ]:


df['euribor3m'].hist(bins=6)


# Again this looks like this may instead be a categorical variable

# In[ ]:


df['euribor3m'].unique()


# There are a lot of unique values which could mean it is not categorical

# Gap in split could be due to target variable

# In[ ]:


df['euribor3m'].hist(by=df['y'],figsize=(12,5))


# Although splitting by target variable doesnt completely separate the distributions. For the yes column there is a higher number of lower values compared to higher values. This means the euribor 3 column could provide some useful info. 

# Euribor is the interest rate at which credit institutions lend money to each other. Euribor 3m is actually an average of the rates at which European banks lend money to each other over a 3 month period. https://www.bankinter.com/banca/en/faqs/mortgages/what-is-euribor-and-how-does-it-affect-me

# This histogram shows that a lot of people subscribe to a term deposit when the euribor rate is low. 

# # Boxplots of Categorical variables 

# possibly useful information from client data

# In[ ]:


#Boxplot of Job by age
df.boxplot('age','job',figsize=(15,5))
plt.ylabel('Age')
plt.xlabel('Job')


# As expected students have the lowest median age and the retired have the highest

# In[ ]:


df.boxplot('age','marital',figsize=(15,10))
plt.ylabel('Age')
plt.xlabel('Marital Status')


# Again expected, single people have the lowest median age with divorce being highest

# Distribution as it pertains to the target variable

# In[ ]:


df.boxplot('age','y',figsize=(10,5))
plt.ylabel('Age')
plt.xlabel('y')


# In[ ]:


df.boxplot('duration','y',figsize=(10,5))


# Those that did not subscribe to a term loan have a lower median call duration this is explained in the bank-additional-names.txt file

# In[ ]:


df.boxplot('euribor3m','y',figsize=(10,5))


# Median at diferent points, Euribor 3m is also helpful in distinguishing whether or not they will subscribe

# In[ ]:


df.boxplot('cons.price.idx','y',figsize=(10,5))


# In[ ]:


df.boxplot('nr.employed','y',figsize=(10,5))


# This analysis shows that Consumer confidence index, Euribor 3 month rate and number of employees may also affect whether or not they will subscribe to a term loan as the bosplots show different medians for each target variable. 

# # Missing values (Unknown)

# In[ ]:


#checking number of unknowns in each column
(df=='unknown').sum()


# Job, Marital Status, Education, default, housing and loan all have unknown values. Can either be MCAR, MNAR or MAR. Missing completely at random means that the variable that is missing has nothing to do with the data. could be missing due to equipment failure. MAR means it is missing due to another variable being measured. MNAR is due to the variable that is being studied. 

# # Filling unknown with mode

# In[ ]:


df.job.mode()[0]


# In[ ]:


def fillmode(dataframe,columns):
    for i in columns:
        df[i] = df[i].apply(lambda x:df[i].mode()[0] if x=='unknown' else x)


# In[ ]:


fillmode(df,df.columns)


# In[ ]:


#rechecking the unknowns in each column
(df=='unknown').sum()


# There are now zero unknowns

# Checking for NAs in numerical columns

# In[ ]:


df.isnull().sum()


# No NAs and no unknowns

# # Transforming categorical features

# Need domain knowledge on whether the categorical features are nominal or ordinal

# In[ ]:


df_cat = df.select_dtypes('object')


# In[ ]:


df_cat.columns


# from previous analysis month and poutcome are important. Fom domain knowledge I think that Job, Education, housing and loan will have some affect.

# In[ ]:


df_cat.head()


# month and day of week are ordinal everything else is nominal. Ordinal variables must be labelled in order, nominal variables can be labelled randomly. 

# In[ ]:


df_cat['job'].nunique()


# # Labelling nominal variables

# In[ ]:


#importing encoder
from sklearn.preprocessing import LabelEncoder


# In[ ]:


jle = LabelEncoder()
#creating new columns with labels
df_cat['jobLabel']=jle.fit_transform(df_cat['job'])
df_cat['educationLabel']=jle.fit_transform(df_cat['education'])
df_cat['poutcomeLabel']=jle.fit_transform(df_cat['poutcome'])


# In[ ]:


df_cat.head()


# # Labelling binary ('yes','no') variables

# In[ ]:


#creating binary map
bin_map = {'yes': 1, 'no': 0}

#creating new labelled columns
df_cat['defaultLabel']=df_cat['default'].map(bin_map)
df_cat['housingLabel']=df_cat['housing'].map(bin_map)
df_cat['loanLabel']=df_cat['loan'].map(bin_map)
df_cat['yLabel']=df_cat['y'].map(bin_map)


# In[ ]:


df_cat.head()


# In[ ]:


#checking correlation of binary labels to see which is most correlated with output
df_cat[['housingLabel','loanLabel','defaultLabel','yLabel']].corr()


# None are correlated with y, housing is most correlated

# # Labelling ordinal features

# In[ ]:


df_cat['month'].unique()


# In[ ]:


month_map = {'mar': 3,
            'apr':4,
            'may':5,
            'jun':6,
            'jul':7,
            'aug':8,
            'sep':9,
            'oct':10,
            'nov':11,
            'dec':12}


# In[ ]:


df_cat['month'] = df['month'].map(month_map)


# In[ ]:


df_cat['day_of_week'].unique()


# In[ ]:


day_map = {'mon':1,
          'tue':2,
          'wed':3,
          'thu':4,
          'fri':5}


# In[ ]:


df_cat['dayLabel'] = df_cat['day_of_week'].map(day_map)


# In[ ]:


df_cat.head()


# In[ ]:


#removing loan and default columns as these were the least correlated  
df_cat_labels = df_cat[['month','jobLabel',
       'educationLabel','poutcomeLabel',
       'housingLabel', 'dayLabel','yLabel']]


# In[ ]:


df_cat_labels.columns


# In[ ]:


#renaming month column
df_cat_labels.columns=['monthLabel', 'jobLabel', 'educationLabel', 'poutcomeLabel',
       'housingLabel', 'dayLabel','yLabel']


# # Feature Selection using Chi-Squared test

# In[ ]:


#numerical variables
df_num = df.select_dtypes(['int64','float64'])


# In[ ]:


df_num.head()


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


# In[ ]:


selector = SelectKBest(chi2, k=8)

#fitting the selector but dropping negative columns
selector.fit(df_num.drop(['cons.conf.idx','emp.var.rate'],axis=1), df['y'])

#creating dataframe of features and their p values from lowest to highest, low p values are most correlated with output
pd.DataFrame({'Features':df_num.drop(['cons.conf.idx','emp.var.rate'],axis=1).columns, 'P-Value':selector.pvalues_}).sort_values(by='P-Value')


# According to this chi-squared test duration, pdays, previous, euribor3m and nr.employed are the most important numerical features

# # Training and testing

# In[ ]:


#checking correlation of numerical variables
plt.figure(figsize=(12,10))
sns.heatmap(df_num.corr(), annot=True)


# Highly correlated features represent the same information making them redundant. euribor3m, emp.var.rate and nr.employed are all highly correlated with one another. Will only be going forward with euribor3m and the rest will be removed.

# In[ ]:


df_num.columns


# In[ ]:


df_cat_labels.columns


# In[ ]:


#joining the 2 dataframes to create dataset
df2 = pd.concat([df_num[['duration','campaign', 'pdays', 'previous',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m']],df_cat_labels],axis=1)


# In[ ]:


df2.head()


# In[ ]:


df2.yLabel.value_counts()


# In[ ]:


100 * 36548/(36548+4640)


# Predicting all as no will give 89% accuracy. Therefore accuracy isn't the best metric when determining performance for this task. To recap the objective of the classification, we are trying to predict who will subscribe a term deposit so the bank can direct their marketing campaign. In my opinion I think they would want to minimise the amount of people they market to that will not subscribe. This is what the model predicts as yes but it actually no (False Positives for the yes (1) column). Therefore the most relevant metric is the precision of the yes column.

# # Train, test split

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df2.columns


# In[ ]:


X = df2[['duration', 'campaign', 'pdays', 'previous', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'monthLabel', 'jobLabel',
       'educationLabel', 'poutcomeLabel', 'housingLabel', 'dayLabel']]
y=df2['yLabel']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=101)


# # Random Forest

# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))
print('Accuracy:',accuracy_score(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


# # Cross validation

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[ ]:


skf = StratifiedKFold(n_splits=5,shuffle=True)


# In[ ]:


rfc_scores = cross_val_score(rfc, X, y, cv=skf)


# In[ ]:


print('Cross validation accuracy for Random Forest Classifier is:', rfc_scores.mean())


# In[ ]:


rfc_scores


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


neighbors = range(5,100,5)


# In[ ]:


acc = []
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n, metric='manhattan')
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test,pred))


# In[ ]:


plt.plot(neighbors, acc)
plt.title('Acuracy vs no. of neighbors')
plt.xlabel('No. of neighbors')
plt.ylabel('Accuracy')


# Accuracy peaks at around 40 neighbours

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=40, metric='manhattan')


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred))
print('Accuracy:',accuracy_score(pred,y_test))
print(classification_report(y_test,pred))


# In[ ]:


# Cross validation


# In[ ]:


knn_scores = cross_val_score(knn, X, y, cv=skf)


# In[ ]:


print('Cross validation accuracy for Neigherest neighbours classifier is:', knn_scores.mean())


# In[ ]:


knn_scores


# Nearest neighbour algorithm has best precision score for 1 class(0.67) so this is currently the best model. 
# 
# If the goal of the task is to limit the number of people they think will not subscribe but actually will the metric to use will be the recall of the 1 class. In this second case the best algorithm is the Random Forest. The Random Forest is also the most accurate.
# 
# Still need to optimise Random forest parameters e.g. n_estimators, max_depth etc. and will look at other classifiers. Unknowns were imputed with the mode may also see how removing unknowns entirely affects the task.
