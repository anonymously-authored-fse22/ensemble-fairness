#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing with Machine Learning

# ## Introduction

# 
# Marketing to potential customers have traditionally been conducted through phone calls and emails but with machine learning we can use algorithms to calculate the people with best chance of buying a bank term deposit! A Portuguese banking institution conducted direct marketing campaigns which this data set is based off on. More than one contact to a client was required, in order to know if the product (bank term deposit) was subscribed by a client or not. Our goal is to predict if a client will subscribe to the bank term deposit (yes/no), which is done by classification!
# 
# The marketing campaigns dataset contains 21 columns including the output (y) – end result. I am going to discard the output column and use the remaining columns to find the most relatable independent variables (x) – features, that will predict if a customer will subscribe to a bank deposit or not. Let’s get started!

# ## Dataset

# ### Input variables:
# 
# 1 - age (numeric)
# 
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# 
# ### Related with the last contact of the current campaign:
# 
# 8 - contact: contact communication type (categorical: 'cellular','telephone') 
# 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# ### Other attributes:
# 
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# 
# ### Social and economic context attributes:
# 
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
# 
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# 
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# ### Output variable (desired target):
# 
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# 
# Source:https://archive.ics.uci.edu/ml/machine-learning-databases/00222/ 
# 
# Dataset has 40,000+ rows of data.

# # Project Definition

# The classification goal is to predict if a client will subscribe to the bank term deposit (yes/no).
# 

# # Data Exploration

# I started by importing the pandas package which is used for manupulation of data. Then, I loaded the dataset into the dataframe (df).

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/bank-additional-full.csv', sep=';')


# In[ ]:


df.info()


# In[ ]:


df[list(df.columns)[6:]].head()


# In[ ]:


df.groupby('y').size()


# I created a column called OUTPUT_LABEL which is going to represent 0 for the negative class and 1 for the positive class based on the bank marketing data set.

# In[ ]:


df['OUTPUT_LABEL'] = (df.y == 'yes').astype('int')


# The prevalence of the positive class is calculated here...

# In[ ]:


def calc_prevalence(y_actual):
    # this function calculates the prevalence of the positive class (label = 1)
    return (sum(y_actual)/len(y_actual))


# In[ ]:


print('prevalence of the positive class: %.3f'%calc_prevalence(df['OUTPUT_LABEL'].values))


# The prevlence of the positive class is 11.3% which means that the proportion of people who agreed to a term deposit (positive class) compared to the people who did not is 11.3%.

# ## Exploring the data set and unique values

# Pandas doesn't allow you to see all the columns at once, so we will look at them in groups of 10.

# In[ ]:


df[list(df.columns)[:10]].head()


# In[ ]:


df[list(df.columns)[10:]].head()


# In[ ]:


df.info()


# In[ ]:


print('Number of columns:',len(df.columns))


# In[ ]:


# for each column
for a in list(df.columns):
    
    # get a list of unique values
    n = df[a].unique()
    
    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(n)<30:
        print(a)
        print(n)
    else:
        print(a + ': ' +str(len(n)) + ' unique values')


# ### Key Observations:

# - From the output of the code, we can see that there are roughly the same amount of categorical and numeric values in the columns.
# 
# - age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m and nr.employed are numerical variables.
# 
# - All the data inputted are non-null values, meaning that we have a value for every column.
# 
# - Output (y) has two values: "yes" and "no".
# 
# - default, housing and loan have 3 values each (yes, no and unknown).
# 
# - We are discarding duration. This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# - age has 78 unique variables, so we are going to break it down to less than or equal to 35 and greater than 35.
# 
# - euribor3m (3 month rate - daily indicator) has 316 unique variables, which is a lot and does not bring major insights into our data set, therefore we are going to discard the 3 month rate daily indicator.
# 
# 

# ## Feature Engineering

# Feature Engineering is classifying features such as numerial and categorical into groups in order to deeply section and analyze the data for results in machine learning algorithms.

# ### Numerical Features

# Breaking up age into two pasts - less than or equal to 35 and greater than 35. Because there are so many unique values for ages so I wanted to categorize them to make sense of the data. So that at the end of our analysis we can tell which customers that agree or don't agree to get a term deposit fall into which age category.

# In[ ]:


df['is_less_than_or_equal_to_35'] = (df['age'] <= 35).astype('int')
df['is_greater_than_35'] = (df['age'] > 35).astype('int')


# In[ ]:


cols_num = ['campaign', 'pdays',
       'previous', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'nr.employed','is_less_than_or_equal_to_35','is_greater_than_35']


# In[ ]:


df[cols_num].head()


# ### Graphical Representation of Numerical Features

# In[ ]:


df[cols_num].hist(column=cols_num, figsize = (16,16))


# Let's check if there are any missing values in the numerical data. 

# In[ ]:


df[cols_num].isnull().sum()


# ## Categorical Features

# Categorical variables are non-numeric data such as race and gender. To turn these non-numerical data into variables, the simplest thing is to use a technique called one-hot encoding, which will be explained below.
# 
# The first set of categorical data we will deal with are these columns:

# In[ ]:


cols_cat = ['job', 'marital', 
       'education', 'default',
       'housing', 'loan', 'contact', 'month',
       'day_of_week', 'poutcome']


# Let's check if there are any missing data

# In[ ]:


df[cols_cat].isnull().sum()


# ## One-Hot Encoding

# Now we are going to use the one-hot enconding feature. This feature creates a unique column for each entry of every categorical variable so we can deeply anlyze them. 

# In[ ]:


cols_cat = ['job', 'marital', 
       'education', 'default',
       'housing', 'loan', 'contact', 'month',
       'day_of_week', 'poutcome']
df[cols_cat]
cols_new_cat=pd.get_dummies(df[cols_cat],drop_first = True)
cols_new_cat


# ### Graphical Representation of Categorical Features

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'education', data = df[cols_cat])
ax.set_xlabel('Levels of Education', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Education', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'marital', data = df[cols_cat])
ax.set_xlabel('Marital Status', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Marital', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'job', data = df[cols_cat])
ax.set_xlabel('Types of Jobs', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Job', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'default', data = df[cols_cat])
ax.set_xlabel('Default Status', fontsize=16)
ax.set_ylabel('Number of Defaults', fontsize=16)
ax.set_title('Defaults', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'housing', data = df[cols_cat])
ax.set_xlabel('Kind of Housing', fontsize=16)
ax.set_ylabel('Housing Count', fontsize=16)
ax.set_title('Housing', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'loan', data = df[cols_cat])
ax.set_xlabel('Loan Status', fontsize=16)
ax.set_ylabel('Number of Loans', fontsize=16)
ax.set_title('Loan', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'contact', data = df[cols_cat])
ax.set_xlabel('Type of Contact', fontsize=16)
ax.set_ylabel('Number of Contacts', fontsize=16)
ax.set_title('Contacts', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'month', data = df[cols_cat])
ax.set_xlabel('Months', fontsize=16)
ax.set_ylabel('Month Count', fontsize=16)
ax.set_title('Month', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'day_of_week', data = df[cols_cat])
ax.set_xlabel('Day', fontsize=16)
ax.set_ylabel('Day Count', fontsize=16)
ax.set_title('Day of the Week', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'poutcome', data = df[cols_cat])
ax.set_xlabel('Previous Marketing Campaign Outcome', fontsize=16)
ax.set_ylabel('Number of Previous Outcomes', fontsize=16)
ax.set_title('poutcome (Previous Marketing Campaign Outcome)', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()


# In order to add the one-hot encoding columns to the dataframe, we use the concat function. axis = 1 is used to add the columns. 

# In[ ]:


df = pd.concat([df,cols_new_cat], axis = 1)


# In[ ]:


cols_all_cat=list(cols_new_cat.columns)


# In[ ]:


df[cols_all_cat]


# I removed Campaign column from the dataset because there are so many unique values and it does not make any lasting effect in the dataset for affecting the outcome of a person opening a bank term deposit

# In[ ]:


cols_new_num = ['pdays',
       'previous', 'emp.var.rate', 'cons.price.idx','cons.conf.idx','nr.employed','is_less_than_or_equal_to_35','is_greater_than_35']
df[cols_new_num].head(12)


# ### Summary of Features Engineering 

# In[ ]:


print('Total number of features:', len(cols_all_cat+cols_new_num))
print('Numerical Features:',len(cols_new_num))
print('Categorical Features:',len(cols_all_cat))


# In[ ]:


df[cols_new_num].head()


# Data check for missing values

# In[ ]:


df[cols_new_num+cols_all_cat].isnull().sum().sort_values(ascending = False)


# Good to go! No empty cells! Also, I created a new dataframe below, which includes the columns of interest. 

# In[ ]:


cols_input = cols_new_num + cols_all_cat
df_data = df[cols_input + ['OUTPUT_LABEL']]


# In[ ]:


cols_input


# In[ ]:


len(cols_input)


# In[ ]:


df_data.head(6)


# ## Building Training, Validation & Test Samples

# Training samples: these are samples from the data set used to train the model. It can be 70% of the data.
# Validation samples: these are samples used to validate or make decisions from the model. It can be 15% of the data.
# Test samples: these are samples used to measure the accuracy or performace of the model. It can be 15% of the data.

# The training (df_train_all), validation (df_valid) and test (df_test) set were created below.

# Shuffle the samples

# In[ ]:


df_data = df_data.sample(n = len(df_data), random_state = 42)
df_data = df_data.reset_index(drop = True)


# 30% of the validation and test samples:

# In[ ]:


df_valid_test=df_data.sample(frac=0.30,random_state=42)
print('Split size: %.3f'%(len(df_valid_test)/len(df_data)))


# Split into test and validation samples by 50% which makes 15% of test and 15% of validation samples.

# In[ ]:


df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
df_valid = df_valid_test.drop(df_test.index)


# Use the rest of the data for the training samples

# In[ ]:


df_train_all=df_data.drop(df_valid_test.index)


# In[ ]:


# check the prevalence of each 
print('Test prevalence(n = %d):%.3f'%(len(df_test),calc_prevalence(df_test.OUTPUT_LABEL.values)))
print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))
print('Train all prevalence(n = %d):%.3f'%(len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))


# We need to balance the data set because if we use the training data as the predictive model the accuracy is going to be very high because we haven't caught any of the y output which states whether a person will buy a term deposit or not. There are more negatives than positive so the predictive models assigns negatives to much of the samples. Creating a balance sheet will allow 50% of the samples to be both positive and negative.

# In[ ]:


# split the training data into positive and negative
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# merge the balanced data
df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

# shuffle the order of training samples 
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

print('Train balanced prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))


# All 4 dataframes were saved intto csv and the cols_input

# In[ ]:


df_train_all.to_csv('df_train_all.csv',index=False)
df_train.to_csv('df_train.csv',index=False)
df_valid.to_csv('df_valid.csv',index=False)
df_test.to_csv('df_test.csv',index=False)


# Saving cols_input too with a package called pickle

# In[ ]:


import pickle
pickle.dump(cols_input, open('cols_input.sav', 'wb'))


# Any missing values were filled with the mean value

# In[ ]:


def fill_my_missing(df, df_mean, col2use):
    # This function fills the missing values

    # check the columns are present
    for c in col2use:
        assert c in df.columns, c + ' not in df'
        assert c in df_mean.col.values, c+ 'not in df_mean'
    
    # replace the mean 
    for c in col2use:
        mean_value = df_mean.loc[df_mean.col == c,'mean_val'].values[0]
        df[c] = df[c].fillna(mean_value)
    return df


# The mean value from the training data:

# In[ ]:


df_mean = df_train_all[cols_input].mean(axis = 0)
# save the means
df_mean.to_csv('df_mean.csv',index=True)


# Loaded the means

# In[ ]:


df_mean_in = pd.read_csv('df_mean.csv', names =['col','mean_val'])
df_mean_in.head()


# In[ ]:


df_train_all = fill_my_missing(df_train_all, df_mean_in, cols_input)
df_train = fill_my_missing(df_train, df_mean_in, cols_input)
df_valid = fill_my_missing(df_valid, df_mean_in, cols_input)


# In[ ]:


# create the X and y matrices
X_train = df_train[cols_input].values
X_train_all = df_train_all[cols_input].values
X_valid = df_valid[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values

print('Training All shapes:',X_train_all.shape)
print('Training shapes:',X_train.shape, y_train.shape)
print('Validation shapes:',X_valid.shape, y_valid.shape)


# Created a scalar, saveed it, and scaled the X matrices

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
scaler.fit(X_train_all)


# In[ ]:


scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))


# In[ ]:


# load it back
scaler = pickle.load(open(scalerfile, 'rb'))


# In[ ]:


# transform our data matrices
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)


# ## Model Selection 

# This section allows us to test various  machine learning algorithm to see how our independent variables accurately predit our dependent y output variable.

# In[ ]:


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('specificity:%.3f'%specificity)
    print('prevalence:%.3f'%calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity 


# Since we balanced our training data, let's set our threshold at 0.5 to label a predicted sample as positive. 

# In[ ]:


thresh = 0.5


# ## Model Selection: baseline models

# ### K nearest neighbors (KNN)

# K Nearest Neighbors looks at the k closest datapoints and probability sample that has positive labels. It is easy to implement, and you don't need an assumption for the data structure. KNN is also good for multivariate analysis.

# Training and evaluating KNN performance:

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 100)
knn.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = knn.predict_proba(X_train_tf)[:,1]
y_valid_preds = knn.predict_proba(X_valid_tf)[:,1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall,     knn_train_precision, knn_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
knn_valid_auc, knn_valid_accuracy, knn_valid_recall,     knn_valid_precision, knn_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# Using K Nearest Neighbors and dividing the data into training and validation samples, I was able to get an AUC of 79.4% which catches 59.2% of potential customers using a threshold of 0.5 for the training set, which is good.

# ### Logistic Regression

# Logsitic regression uses a line (Sigmoid function) in the form of an "S" to predict if the dependent variable is true or false based on the independent variables. The "S-shaped" curve (on the line graph) will show the probability of the dependent variable occuring based on where the points of the independent variables lands on the curve. In this case, the output (y) is predicted by the numerical and categorical variables defined as "x" such as age, education and so on. Logistic regresssion is best used for classifying samples.

# Training and evaluating the logistic regression performance:

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state = 42)
lr.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = lr.predict_proba(X_train_tf)[:,1]
y_valid_preds = lr.predict_proba(X_valid_tf)[:,1]

print('Logistic Regression')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall,     lr_train_precision, lr_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall,     lr_valid_precision, lr_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# ### Stochastic Gradient Descent

# Stochastic Gradient Descent analyzes various sections of the data instead of the data as a whole and predicts the output using the independent variables. Stochastic Gradient Descent is faster than logistic regression in the sense that it doesn't run the whole dataset but instead looks at different parts of the dataset.

# Training and evaluating Stochastic Gradient Descent model performance:

# In[ ]:


from sklearn.linear_model import SGDClassifier
sgdc=SGDClassifier(loss = 'log',alpha = 0.1,random_state = 42)
sgdc.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = sgdc.predict_proba(X_train_tf)[:,1]
y_valid_preds = sgdc.predict_proba(X_valid_tf)[:,1]

print('Stochastic Gradient Descent')
print('Training:')
sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, sgdc_train_precision, sgdc_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, sgdc_valid_precision, sgdc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# ### Naive Bayes

# Naive Bayes assumes that all variables in the dataset are independent of each other. Meaning that there are no dependent variables or output. This algorithm uses Bayes rule which calculated the probability of an event related to previous knowledge of the variables converning the event. This won't really work in this case since we have an output of the bank customers who will get a bank deposit. This process is better for tasks such as image processing.

# Training and evaluating Naive Bayes model performance:

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = nb.predict_proba(X_train_tf)[:,1]
y_valid_preds = nb.predict_proba(X_valid_tf)[:,1]

print('Naive Bayes')
print('Training:')
nb_train_auc, nb_train_accuracy, nb_train_recall, nb_train_precision, nb_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
nb_valid_auc, nb_valid_accuracy, nb_valid_recall, nb_valid_precision, nb_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# ### Decision Tree Classifier

# Decision trees works through the data to decide if one action occurs, what will then be the result of a "yes" and a "no". It works each data making the decision of which path to take based on the answer. Because of this decision making process, this algorithm has no assumptions about the structure of the data, but instead decides on the path to take through each decision the algorithm performs.

# Training and evaluating Decision Tree model performance:

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 10, random_state = 42)
tree.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = tree.predict_proba(X_train_tf)[:,1]
y_valid_preds = tree.predict_proba(X_valid_tf)[:,1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# ### Random Forest

# Random forest works like a decision tree algorithm but it performs various decision tree analysis on the dataset as a whole. That is, it is the bigger version of the decision tree; a forest is bigger than a tree, you can think of it that way. Random forest takes random samples of trees and it works faster than the decision tree algorithm. 

# Training and evaluating Random Forest model performance:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth = 6, random_state = 42)
rf.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = rf.predict_proba(X_train_tf)[:,1]
y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, rf_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, rf_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# ### Gradient Boosting Classifier

# Boosting is a technique that builds a new decision tree algorithm that focuses on the errors on the dataset to fix them. This learns the whole model in other to fix it and improve the prediction of the model. Aside from being related to decision trees, it also relates to gradient descent algorithm as the name signifies. Gradient boosting analyzes different parts of the dataset and builds trees that focuses and corrects those errors. The XGBoost library is also the determining factor in winning a lot of Kaggle data science competitions.

# Training and evaluating Gradient Boosting model performance:

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)


# In[ ]:


y_train_preds = gbc.predict_proba(X_train_tf)[:,1]
y_valid_preds = gbc.predict_proba(X_valid_tf)[:,1]

print('Gradient Boosting Classifier')
print('Training:')
gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, gbc_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, gbc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)


# ## Analyze results baseline models

# Let's make a dataframe with these results and plot the outcomes using a package called seaborn.

# In[ ]:


df_results = pd.DataFrame({'classifier':['KNN','KNN','LR','LR','SGD','SGD','NB','NB','DT','DT','RF','RF','GB','GB'],
                           'data_set':['train','valid']*7,
                          'auc':[knn_train_auc, knn_valid_auc,lr_train_auc,lr_valid_auc,sgdc_train_auc,sgdc_valid_auc,nb_train_auc,nb_valid_auc,tree_train_auc,tree_valid_auc,rf_train_auc,rf_valid_auc,gbc_train_auc,gbc_valid_auc,],
                          'accuracy':[knn_train_accuracy, knn_valid_accuracy,lr_train_accuracy,lr_valid_accuracy,sgdc_train_accuracy,sgdc_valid_accuracy,nb_train_accuracy,nb_valid_accuracy,tree_train_accuracy,tree_valid_accuracy,rf_train_accuracy,rf_valid_accuracy,gbc_train_accuracy,gbc_valid_accuracy,],
                          'recall':[knn_train_recall, knn_valid_recall,lr_train_recall,lr_valid_recall,sgdc_train_recall,sgdc_valid_recall,nb_train_recall,nb_valid_recall,tree_train_recall,tree_valid_recall,rf_train_recall,rf_valid_recall,gbc_train_recall,gbc_valid_recall,],
                          'precision':[knn_train_precision, knn_valid_precision,lr_train_precision,lr_valid_precision,sgdc_train_precision,sgdc_valid_precision,nb_train_precision,nb_valid_precision,tree_train_precision,tree_valid_precision,rf_train_precision,rf_valid_precision,gbc_train_precision,gbc_valid_precision,],
                          'specificity':[knn_train_specificity, knn_valid_specificity,lr_train_specificity,lr_valid_specificity,sgdc_train_specificity,sgdc_valid_specificity,nb_train_specificity,nb_valid_specificity,tree_train_specificity,tree_valid_specificity,rf_train_specificity,rf_valid_specificity,gbc_train_specificity,gbc_valid_specificity,]})


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")


# I picked AUC (area under the ROC curve) as a performance indicator. The reason I chose this over other indicators such as precision and accuracy is that it measures the relationshio between true positives and false positives in our data in order to derive a score that depicts that. Also, AUC is widely used and an easier metric to compare many models with.

# All the algorithms have pretty much the same AUC, but the ones that stood out our decision tree (DT) and gradient boosting (GB). I would choose gradient boosting as the best metric to use because it has a higher auc (0.874) than the other algorithms. At a threshold of 0.5, an auc of 0.874 is good as it signifies that it is more than just a random guess towards a positive class and it is close to 1 which is perfect.

# In[ ]:


ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)
ax.set_xlabel('Classifier',fontsize = 15)
ax.set_ylabel('AUC', fontsize = 15)
ax.tick_params(labelsize=15)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)
plt.show()


# ## Learning Curves

# Gradient Descent has the best AUC score (0.796) for the validation model and the learning curve for the model will be displayed below.

# In[ ]:


import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = 'roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# The Stochastic Gradient Descent model with max_depth = 20, resulting in high variance

# In[ ]:


from sklearn.linear_model import SGDClassifier
title = "Learning Curves (Stochastic Gradient Descent)"
# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
estimator = RandomForestClassifier(max_depth = 20, random_state = 42)
plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

plt.show()


# ### Variance and Bias

# My model has high bias because the training and cross-validation score are close to each other which shows there are near in numbers. The model has high variance because the training and cross-validation scores show a lot of samples which are close to each other. So the model has both high bias and variance.

# ## Feature Importance

# This section focuses on the importance of the different features generated and in the dataframe. Depending on the importance score of some features, we can focus on higher importance scores to see if the AUC score (performance) of the model will improve.

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state = 42)
lr.fit(X_train_tf, y_train)


# In[ ]:


feature_importances = pd.DataFrame(lr.coef_[0],
                                   index = cols_input,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)


# In[ ]:


feature_importances.head()


# In[ ]:


num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Logistic Regression')
plt.yticks(ylocs, feature_labels)
plt.show()


# In[ ]:


values_to_plot = feature_importances.iloc[-num:].values.ravel()
feature_labels = list(feature_importances.iloc[-num:].index)

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Logistic Regression')
plt.yticks(ylocs, feature_labels)
plt.show()


# I realized the features that have more positive impact on the predictive outcomes of the model. nr.employed, mon_mar, and cons.price are very crucial as their importance score is higher than other nummerical variables. emp.var.rate has a high negative importance score. One way to look at it is I can remove the other columns from my dataset and maybe I can achieve a higher auc score.

# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth = 6, random_state = 42)
rf.fit(X_train_tf, y_train)


# In[ ]:


feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = cols_input,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)
feature_importances.head()


# In[ ]:


num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Random Forest')
plt.yticks(ylocs, feature_labels)
plt.show()


# ### Other Algorithm Feature Importance Scores

# ### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)


# In[ ]:


feature_importances = pd.DataFrame(gbc.feature_importances_,
                                   index = cols_input,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)
feature_importances.head()


# I realized the features that have more positive impact on the predictive outcomes of the model. nr.employed, emp.var.rate and cons.price.idx are very crucial as their importance score is higher than other nummerical variables. One way to look at it is I can remove the other columns from my dataset and maybe I can achieve a higher auc score.

# In[ ]:


num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Gradient Boosting Classifier')
plt.yticks(ylocs, feature_labels)
plt.show()


# ### Decision Trees

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 10, random_state = 42)
tree.fit(X_train_tf, y_train)


# In[ ]:


feature_importances = pd.DataFrame(tree.feature_importances_,
                                   index = cols_input,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)
feature_importances.head()


# In[ ]:


num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Decision Trees')
plt.yticks(ylocs, feature_labels)
plt.show()


# After looking at the importance of each features, I tried removing some columns with a lower importance from the model to see if my AUC will increase but it only decreased. Therefore, I'm leaving my features as it is.

# ## Hyperparameter tuning

# A hyperparameter is a tool used in machine learning in order to estimate the model parameters (used in tuning a predictive modeling problem).  Hyperparameters are used in various machine learning algorithms. 

# In[ ]:


# train a model for each max_depth in a list. Store the auc for the training and validation set

# max depths
max_depths = np.arange(2,20,2)

train_aucs = np.zeros(len(max_depths))
valid_aucs = np.zeros(len(max_depths))

for jj in range(len(max_depths)):
    max_depth = max_depths[jj]

    # fit model
    rf=RandomForestClassifier(n_estimators = 100, max_depth = max_depth, random_state = 42)
    rf.fit(X_train_tf, y_train)        
    # get predictions
    y_train_preds = rf.predict_proba(X_train_tf)[:,1]
    y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]

    # calculate auc
    auc_train = roc_auc_score(y_train, y_train_preds)
    auc_valid = roc_auc_score(y_valid, y_valid_preds)

    # save aucs
    train_aucs[jj] = auc_train
    valid_aucs[jj] = auc_valid


# n_estimators is a hyperparameter in the RandomForestClassifier that depending on the numbers of estimators entered the model can be overfitted, good compromise or underfitted. n_estimators is used for fine tuning the models in order to fit the training data. max_depth is also another hyperparameter; it controls the depth of the machine learning algorithm model.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(max_depths, train_aucs,'o-',label = 'train')
plt.plot(max_depths, valid_aucs,'o-',label = 'valid')

plt.xlabel('max_depth')
plt.ylabel('AUC')
plt.legend()
plt.show()


# Using RandomizedSearchCV to optimize a few of the baseline models. GradientBoosting Classifier may take a while so one might need to adjust the number of iterations or specific parameters. If this takes too long on the computer, one can take it out.

# In[ ]:


rf.get_params()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# number of trees
n_estimators = range(200,1000,200)
# maximum number of features to use at each split
max_features = ['auto','sqrt']
# maximum depth of the tree
max_depth = range(2,20,2)
# minimum number of samples to split a node
min_samples_split = range(2,10,2)
# criterion for evaluating a split
criterion = ['gini','entropy']

# random grid

random_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'max_depth':max_depth,
              'min_samples_split':min_samples_split,
              'criterion':criterion}

print(random_grid)


# In[ ]:


from sklearn.metrics import make_scorer, roc_auc_score
auc_scoring = make_scorer(roc_auc_score)


# In[ ]:


# create a baseline model
rf = RandomForestClassifier()

# create the randomized search cross-validation
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 20, cv = 2, 
                               scoring=auc_scoring,verbose = 1, random_state = 42)


# In[ ]:


import time
# fit the random search model (this will take a few minutes)
t1 = time.time()
rf_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2-t1)


# See the best parameters

# In[ ]:


rf_random.best_params_


# In[ ]:


rf=RandomForestClassifier(max_depth = 6, random_state = 42)
rf.fit(X_train_tf, y_train)

y_train_preds = rf.predict_proba(X_train_tf)[:,1]
y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]

thresh = 0.5

print('Baseline Random Forest')
rf_train_base_auc = roc_auc_score(y_train, y_train_preds)
rf_valid_base_auc = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f'%(rf_train_base_auc))
print('Validation AUC:%.3f'%(rf_valid_base_auc))

print('Optimized Random Forest')
y_train_preds_random = rf_random.best_estimator_.predict_proba(X_train_tf)[:,1]
y_valid_preds_random = rf_random.best_estimator_.predict_proba(X_valid_tf)[:,1]

rf_train_opt_auc = roc_auc_score(y_train, y_train_preds_random)
rf_valid_opt_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f'%(rf_train_opt_auc))
print('Validation AUC:%.3f'%(rf_valid_opt_auc))


# Optimized SGDClassifier

# In[ ]:


from sklearn.linear_model import SGDClassifier
sgdc=SGDClassifier(loss = 'log',alpha = 0.1,random_state = 42)
sgdc.fit(X_train_tf, y_train)


# In[ ]:


penalty = ['none','l2','l1']
max_iter = range(200,1000,200)
alpha = [0.001,0.003,0.01,0.03,0.1,0.3]
random_grid_sgdc = {'penalty':penalty,
              'max_iter':max_iter,
              'alpha':alpha}
# create the randomized search cross-validation
sgdc_random = RandomizedSearchCV(estimator = sgdc, param_distributions = random_grid_sgdc, n_iter = 20, cv = 2, scoring=auc_scoring,verbose = 0, random_state = 42)

t1 = time.time()
sgdc_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2-t1)


# In[ ]:


sgdc_random.best_params_


# In[ ]:


y_train_preds = sgdc.predict_proba(X_train_tf)[:,1]
y_valid_preds = sgdc.predict_proba(X_valid_tf)[:,1]

thresh = 0.5

print('Baseline sgdc')
sgdc_train_base_auc = roc_auc_score(y_train, y_train_preds)
sgdc_valid_base_auc = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f'%(sgdc_train_base_auc))
print('Validation AUC:%.3f'%(sgdc_valid_base_auc))

print('Optimized sgdc')
y_train_preds_random = sgdc_random.best_estimator_.predict_proba(X_train_tf)[:,1]
y_valid_preds_random = sgdc_random.best_estimator_.predict_proba(X_valid_tf)[:,1]
sgdc_train_opt_auc = roc_auc_score(y_train, y_train_preds_random)
sgdc_valid_opt_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f'%(sgdc_train_opt_auc))
print('Validation AUC:%.3f'%(sgdc_valid_opt_auc))


# Optimized Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)


# In[ ]:


# number of trees
n_estimators = range(50,200,50)

# maximum depth of the tree
max_depth = range(1,5,1)

# learning rate
learning_rate = [0.001,0.01,0.1]

# random grid

random_grid_gbc = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'learning_rate':learning_rate}

# create the randomized search cross-validation
gbc_random = RandomizedSearchCV(estimator = gbc, param_distributions = random_grid_gbc, n_iter = 20, cv = 2, scoring=auc_scoring,verbose = 0, random_state = 42)

t1 = time.time()
gbc_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2-t1)


# In[ ]:


gbc_random.best_params_


# In[ ]:


y_train_preds = gbc.predict_proba(X_train_tf)[:,1]
y_valid_preds = gbc.predict_proba(X_valid_tf)[:,1]

thresh = 0.5

print('Baseline gbc')
gbc_train_base_auc = roc_auc_score(y_train, y_train_preds)
gbc_valid_base_auc = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f'%(gbc_train_base_auc))
print('Validation AUC:%.3f'%(gbc_valid_base_auc))
print('Optimized gbc')
y_train_preds_random = gbc_random.best_estimator_.predict_proba(X_train_tf)[:,1]
y_valid_preds_random = gbc_random.best_estimator_.predict_proba(X_valid_tf)[:,1]
gbc_train_opt_auc = roc_auc_score(y_train, y_train_preds_random)
gbc_valid_opt_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f'%(gbc_train_opt_auc))
print('Validation AUC:%.3f'%(gbc_valid_opt_auc))


# Analyzing the 3 results

# In[ ]:


df_results = pd.DataFrame({'classifier':['SGD','SGD','RF','RF','GB','GB'],
                           'data_set':['baseline','optimized']*3,
                          'auc':[sgdc_valid_base_auc,sgdc_valid_opt_auc,
                                 rf_valid_base_auc,rf_valid_opt_auc,
                                 gbc_valid_base_auc,gbc_valid_opt_auc],
                          })


# In[ ]:


df_results


# Comparing the performance of the optimized models to the baseline models. 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)
ax.set_xlabel('Classifier',fontsize = 15)
ax.set_ylabel('AUC', fontsize = 15)
ax.tick_params(labelsize=15)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)

plt.show()


# ## Picking the best model

# I picked Gradient Boosting optimized version as my best model because the optimized version has a higher auc metric than the baseline models of Stochastic Gradient Descent and Random Forest. Gradient Boosting's AUC score also tells me that most of my data are predicted positives which has a good chance of occuring and can be used to make strategic decisions for management.

# In[ ]:


pickle.dump(gbc_random.best_estimator_, open('best_classifier.pkl', 'wb'),protocol = 4)


# # Model Evaluation

# Below is the evaluation of the performance of the best model on the training, validation and test sets. I also created an ROC curve.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[ ]:


# load the model, columns, mean values, and scaler
best_model = pickle.load(open('best_classifier.pkl','rb'))
cols_input = pickle.load(open('cols_input.sav','rb'))
df_mean_in = pd.read_csv('df_mean.csv', names =['col','mean_val'])
scaler = pickle.load(open('scaler.sav', 'rb'))


# In[ ]:


# load the data
df_train = pd.read_csv('df_train.csv')
df_valid= pd.read_csv('df_valid.csv')
df_test= pd.read_csv('df_test.csv')


# In[ ]:


# fill missing
df_train = fill_my_missing(df_train, df_mean_in, cols_input)
df_valid = fill_my_missing(df_valid, df_mean_in, cols_input)
df_test = fill_my_missing(df_test, df_mean_in, cols_input)

# create X and y matrices
X_train = df_train[cols_input].values
X_valid = df_valid[cols_input].values
X_test = df_test[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values
y_test = df_test['OUTPUT_LABEL'].values

# transform our data matrices 
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
X_test_tf = scaler.transform(X_test)


# Prediction possibilities

# In[ ]:


y_train_preds = best_model.predict_proba(X_train_tf)[:,1]
y_valid_preds = best_model.predict_proba(X_valid_tf)[:,1]
y_test_preds = best_model.predict_proba(X_test_tf)[:,1]


# Evaluating performances

# In[ ]:


thresh = .5

print('Training:')
train_auc, train_accuracy, train_recall, train_precision, train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
valid_auc, valid_accuracy, valid_recall, valid_precision, valid_specificity = print_report(y_valid,y_valid_preds, thresh)
print('Test:')
test_auc, test_accuracy, test_recall, test_precision, test_specificity = print_report(y_test,y_test_preds, thresh)


# ### The ROC Curve

# In[ ]:


from sklearn.metrics import roc_curve 

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-',label ='Valid AUC:%.3f'%auc_valid)
plt.plot(fpr_test, tpr_test, 'g-',label ='Test AUC:%.3f'%auc_test)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Our data shows that the results for the training, validation and testing data sets are skewed towards the true positive rate and above the treshold of 0.5 which is great because for example the test AUC of 0.791 means that the models predicts that 79.1% of the customers agree to open a bank term deposit and in actuallity the customers opens a bank term deposit.

# In[ ]:


df_data


# # Conclusion

# - Our AUC for the test is 0.791 which means that we are 79.1% certain of the customers opening a bank deposit compared to human prediction of 11.4%
# 
# - A precision of 0.37 divided by a prevalence of 0.11gives us 3.36, which means that the machine learning model helps us 3 times better than randomly guessing
# 
# - We should focus on targeting customers with high consumer confidence index and consumer price index and other high importance features as  they are paramount to the performance of the model and in turn the business
# 
# - Therefore, we save time and money in knowing the kinds of people we should call and that will lead to more customers and revenue
