#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income Classification
# 
# In this notebook, we will perform exploratory data analysis and explore various classification algorithms to determine whether a person makes over $50K a year (i.e. perform binary classification). Moreover, topics like features selection, cross-validation, model assessment and evaluation will be performed.
# 
# ## Description of Dataset
# The dataset has** 32561 observations** (rows) and **15 features** (columns), and was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: (**(AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0))**. Let's take a closer look at each of the variables.
# 
# 1. **Continuous variables**
# 
#     * **age** - Age of an individual.
#     * **fnlwgt** - The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population. People with similar demographic characteristics should have similar weights. 
#     * **education.num** - Level of education (represented as integer)
#     * **capital.gain **
#     * **capital.loss**
#     * **hours.per.week ** - Individual's working hour per week
# 
# 
# 2. **Categorical variables**
# 
#     * **workclass** - ['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc', 'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked']
#     * **education** - ['HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate', 'Prof-school', 'Bachelors', 'Masters', '11th', 'Assoc-acdm', 'Assoc-voc', '1st-4th', '5th-6th', '12th', '9th', 'Preschool']
#     * **marital.status** - ['Widowed', 'Divorced', 'Separated', 'Never-married', 'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
#     * **occupation** - ['Exec-managerial', 'Machine-op-inspct', 'Prof-specialty',
#        'Other-service', 'Adm-clerical', 'Craft-repair',
#        'Transport-moving', 'Handlers-cleaners', 'Sales',
#        'Farming-fishing', 'Tech-support', 'Protective-serv',
#        'Armed-Forces', 'Priv-house-serv']
#     * **relationship** - ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative', 'Husband', 'Wife']
#     * **race** - ['White', 'Black', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo']
#     * **sex** - ['Female', 'Male']
#     * **native.country** - ['United-States', '?', 'Mexico', 'Greece', 'Vietnam', 'China',
#        'Taiwan', 'India', 'Philippines', 'Trinadad&Tobago', 'Canada',
#        'South', 'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran',
#        'England', 'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba',
#        'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic',
#        'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala',
#        'Jamaica', 'Ecuador', 'France', 'Yugoslavia', 'Scotland',
#        'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)']

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files = list()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# ## A quick look at the data

# In[ ]:


income_census_df = pd.read_csv(files[0])
income_census_df.head()


# ## Let's further investigate, are there any null-values?
# 
# There are no null-values in the dataset, however, columns **workclass** and **occupation** use **"?"** to represent missing records. We will take care of these values going forward.

# In[ ]:


income_census_df.info()
income_census_df.isnull().sum()


# ## Next, let's perform EDA
# 
# Before that, 
# 
# 1. Let's impute the missing values represented by **"?"** for columns occupation and workclass to **Unknown**.
# 2. Clean the data to obtain feature values that can be easily understood.

# In[ ]:


income_census_df['occupation'].replace({'?' : 'Unknown'}, inplace=True)
income_census_df['workclass'].replace({'?' : 'Unknown'}, inplace=True)

income_census_df.head()


# #### Cleaning Marital Status column

# In[ ]:


income_census_df['marital.status'].replace({'Married-civ-spouse' : 'Married' ,
                                            'Divorced' : 'Separated', 
                                            'Married-AF-spouse' : 'Married' , 
                                            'Married-spouse-absent':'Separated'}, inplace = True)
plt.figure(figsize=(10,4))
sns.countplot(income_census_df['marital.status'])


# #### Cleaning education column

# In[ ]:


income_census_df['education'].replace({'HS-grad':'HighSchool', 
                                       'Some-college':'College', 
                                       'Bachelors' : 'University', 
                                       'Masters' : 'University',
                                       'Assoc-voc' : 'College', 
                                       'Assoc-acdm':'College',
                                       'Prof-school' : 'University', 
                                       'Doctorate' : 'University', 
                                       '11th' : 'Dropout',
                                       '10th' : 'Dropout',
                                       '7th-8th' : 'Dropout',
                                       '9th' : 'Dropout', 
                                       '12th' : 'Dropout',
                                       '5th-6th': 'Dropout',
                                       '1st-4th': 'Dropout',
                                       'Preschool':'Dropout'}, inplace = True)

plt.figure(figsize=(10,4))
sns.countplot(income_census_df['education'])
plt.xlabel('Education')


# ### 1. What is the average working hours per week for each gender in different occupations?

# In[ ]:


avg_hours_occ = income_census_df.groupby(['sex','occupation'])['hours.per.week'].mean().reset_index()
plt.figure(figsize=(10, 7))
sns.barplot(x='occupation', y='hours.per.week', hue='sex', data=avg_hours_occ)
plt.xlabel('Occupation')
plt.ylabel('Average work hours per week')
plt.title('Mean number of hours worked by each gender for given occupation')
_ = plt.xticks(rotation=45)


# ### 2. What is the average working hours per week based on education?

# In[ ]:


avg_hours_edu = income_census_df.groupby(['education'])['hours.per.week'].mean().reset_index()
plt.figure(figsize=(7, 5))
sns.barplot(x='education', y='hours.per.week', data=avg_hours_edu)
plt.xlabel('Education')
plt.ylabel('Average work hours per week')
plt.title('Mean number of hours worked based on Education')
_ = plt.xticks(rotation=45)


# ### 3. Is there a relationship between age and average hours worked per week?

# In[ ]:


avg_hours_age = income_census_df.groupby(['sex','age'])['hours.per.week'].mean().reset_index()
plt.figure(figsize=(7, 5))
sns.scatterplot(x='age', y='hours.per.week', hue='sex', data=avg_hours_age)
plt.xlabel('Age')
plt.ylabel('Average work hours per week')
plt.title('Mean number of hours worked based on Age')
_ = plt.xticks(rotation=45)


# ### 4. Do people who have an income > 50K a year, work for more hours on average than people with income <= 50K?

# In[ ]:


avg_hours_income = income_census_df.groupby(['sex','income'])['hours.per.week'].mean().reset_index()
plt.figure(figsize=(7, 5))
sns.barplot(x='income', y='hours.per.week', hue='sex', data=avg_hours_income)
plt.xlabel('Income')
plt.ylabel('Average work hours per week')
plt.title('Mean number of hours worked based on Age')
_ = plt.xticks(rotation=45)


# ### 5. Number of male and female workers with income > 50K based on Occupation

# In[ ]:


income_df = income_census_df[['sex', 'occupation', 'income']].copy()
income_df['income_grt50K'] = income_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
income_grt50K = income_df.groupby(['sex','occupation'])['income_grt50K'].sum().reset_index()

plt.figure(figsize=(10, 7))
sns.barplot(x='occupation', y='income_grt50K', hue='sex', data=income_grt50K)
plt.xlabel('')
plt.ylabel('Number of people (income > 50K)')
plt.title('Number of male and female workers with income > 50K based on Occupation')
_ = plt.xticks(rotation=45)


# ### 6. Number of male and female workers with income > 50K based on Working class

# In[ ]:


income_df = income_census_df[['sex', 'workclass', 'income']].copy()
income_df['income_grt50K'] = income_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
income_grt50K = income_df.groupby(['sex','workclass'])['income_grt50K'].sum().reset_index()

plt.figure(figsize=(10, 7))
sns.barplot(x='workclass', y='income_grt50K', hue='sex', data=income_grt50K)
plt.xlabel('')
plt.ylabel('Number of people (income > 50K)')
plt.title('Number of male and female workers with income > 50K based on Working class')
_ = plt.xticks(rotation=45)


# ### 7. Number of male and female workers with income > 50K based on Age

# In[ ]:


income_df = income_census_df[['sex', 'age', 'income']].copy()
income_df['income_grt50K'] = income_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
income_grt50K = income_df.groupby(['sex','age'])['income_grt50K'].sum().reset_index()
income_grt50K

plt.figure(figsize=(10, 7))
ax = sns.barplot(x='age', y='income_grt50K', hue='sex', data=income_grt50K)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Age')
plt.title('Number of people (income > 50K) by Age')
_ = plt.xticks(rotation=45)


# ### 8. Number of male and female workers with income > 50K based on Education

# In[ ]:


income_df = income_census_df[['sex', 'education', 'income']].copy()
income_df['income_grt50K'] = income_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
income_grt50K = income_df.groupby(['sex','education'])['income_grt50K'].sum().reset_index()

plt.figure(figsize=(10, 7))
sns.barplot(x='education', y='income_grt50K', hue='sex', data=income_grt50K)
plt.xlabel('')
plt.ylabel('Number of people (income > 50K)')
plt.title('Number of male and female workers with income > 50K based on Education')
_ = plt.xticks(rotation=45)


# ### 9. Number of male and female workers with income <= 50K based on Education

# In[ ]:


income_df = income_census_df[['sex', 'education', 'income']].copy()
income_df['income_leq50K'] = income_df['income'].apply(lambda x: 1 if x == '<=50K' else 0)
income_leq50K = income_df.groupby(['sex','education'])['income_leq50K'].sum().reset_index()

plt.figure(figsize=(10, 7))
sns.barplot(x='education', y='income_leq50K', hue='sex', data=income_leq50K)
plt.xlabel('')
plt.ylabel('Number of people (income <= 50K)')
plt.title('Number of male and female workers with income <= 50K based on Education')
_ = plt.xticks(rotation=45)


# ### 10. Do the continuous features have correlation?

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(income_census_df.corr(), annot=True , cmap='summer')


# There is no correlation between any of the continuous predictors
# 
# 
# #### EDA is done for our dataset

# ## Prepare data for modeling

# In[ ]:


# Performing One-Hot Encoding and Label Encoding
income_census_df.head()

# Remove variables: workclass, education, marital.status, occupation, relationship, race, native.country


# In[ ]:


income_census_df = income_census_df.drop(columns=['native.country', 'marital.status', 'education'])
income_census_df.head()


# In[ ]:


# create dummy variables
def get_dummy(data):
    cat_vars = []
    df = data.copy()
    for col in df.columns:
        if (df[col].dtype.name == 'object' and col != 'income'):
            cat_vars.append(col)
    df_preprocessed = pd.get_dummies(df, prefix_sep="_", columns=cat_vars)
    return df_preprocessed


# In[ ]:


df_preprocessed = get_dummy(income_census_df)
df_preprocessed.head()


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(df_preprocessed['income'])


# The class imbalance is clearly evident from the plot above. We will apply down-sampling method to handle this imbalance in the target variable.

# In[ ]:


# Map income variable to 1 and 0
df_preprocessed['target_income'] = df_preprocessed['income'].apply(lambda x: 1 if x == '>50K' else 0)
df_preprocessed.drop(['income'], axis=1, inplace=True)
df_preprocessed.head()


# ## Standardize the Data

# In[ ]:


# Standardize the data

def standardize(df):
    X = df.drop(columns='target_income')
    y = df['target_income']
    
    ss = StandardScaler()
    X[X.columns] = ss.fit_transform(X[X.columns])
    return(X, y)


# In[ ]:


orig_X, orig_y = standardize(df_preprocessed)
orig_X.shape


# ## Downsampling

# In[ ]:


# Perform downsampling

df_majority = df_preprocessed[df_preprocessed.target_income==0]
df_minority = df_preprocessed[df_preprocessed.target_income==1]

df_majority_downsampled = resample(df_majority, 
                                   replace=False, 
                                   n_samples=len(df_minority), 
                                   random_state=123)

df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled.target_income.value_counts()


# In[ ]:


ds_X, ds_y = standardize(df_downsampled)
ds_X.shape


# ## Apply PCA

# In[ ]:


# PCA on original dataset

pca = PCA()
df_orig_pca = pca.fit_transform(orig_X)
df_cumvar = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_)*100, 
                         index = range(1, orig_X.shape[1]+1) , columns=['CumVar'])
df_cumvar[df_cumvar['CumVar'] >= 95].T
# 96% variance is explained by 34 components.


# In[ ]:


pca = PCA(n_components=34)
df_orig_pca1 = pca.fit_transform(orig_X)
orig_pca_X = pd.DataFrame(df_orig_pca1 , columns=['PC '+str(i) for i in range(1,35)])
orig_pca_X.shape


# ## Train-Test Split

# We will now use the following dfs to build our classification models:
# 
# * orig_X, orig_y       
# * orig_pca_X, orig_y
# * ds_X, ds_y  -> (ds = downsampled)

# In[ ]:


def get_train_test_split(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
    return(X_train, X_test, y_train, y_test)


# In[ ]:


orig_trainX, orig_testX, orig_trainY, orig_testY = get_train_test_split(orig_X, orig_y)

print('X_train shape', orig_trainX.shape)
print('y_train shape', orig_trainY.shape)
print('X_test  shape', orig_testX.shape)
print('y_test  shape', orig_testY.shape)


# In[ ]:


def logisticRegression(trainX, trainY, testX, testY):

    lr_model = LogisticRegression(fit_intercept=True, solver='liblinear')
    lr_model.fit(trainX, trainY)
    
    y_pred = lr_model.predict(trainX)
    thres_acc = metrics.accuracy_score(trainY, y_pred, normalize = True)
    print('Logistic Regression (Train Accuracy) : ', thres_acc)
    

    probs = lr_model.predict_proba(testX)
    fpr, tpr, t = metrics.roc_curve(testY, probs[:,1])

    y_pred = lr_model.predict(testX)
    thres_acc = metrics.accuracy_score(testY, y_pred, normalize = True)

    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'Test AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    y_true = testY.apply(lambda x: '<=50k' if x == 0 else '>50k')
    y_preds = np.where(y_pred == 0,'<=50k', '>50k')

    labels = ['<=50k', '>50k']
    cm = confusion_matrix(y_true, y_preds, labels)
    print(cm)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    print('Logistic Regression (Test Accuracy) : ', thres_acc)
    print('Precision Score: ', precision_score(testY, y_pred))
    print('Recall Score: ', recall_score(testY, y_pred))
    print('F1 Score: ', f1_score(testY, y_pred))


# ### Logistic Regression on original dataset

# In[ ]:



logisticRegression(orig_trainX, orig_trainY, orig_testX, orig_testY)


# ### Logistic Regression using Principal Components

# In[ ]:


orig_trainX, orig_testX, orig_trainY, orig_testY = get_train_test_split(orig_pca_X, orig_y)
logisticRegression(orig_trainX, orig_trainY, orig_testX, orig_testY)


# ### Logistic Regresion with Optimal Threshold

# In[ ]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(7, 5))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


# In[ ]:


# Logistic Regression with cross-validation to choose optimal threshold for binary classification

X_train = orig_trainX.reset_index().drop(columns=['index'])
y_train = orig_trainY.reset_index().drop(columns=['index'])

lr_model = LogisticRegression(fit_intercept=True, solver='liblinear')
skf = StratifiedKFold(n_splits=3)

best_accuracy = 0
opt_threshold = 0
fpr = None
tpr = None
p = None
r = None
t = None

for train, test in skf.split(X_train, y_train):
    print(train, test)
    lr_model.fit(X_train.loc[train], y_train.loc[train].values.ravel())
    probs = lr_model.predict_proba(X_train.loc[test])
    fpr, tpr, thresholds = metrics.roc_curve(y_train.loc[test], probs[:,1])
    p, r, t = metrics.precision_recall_curve(y_train.loc[test], probs[:,1])
    
    for thres in thresholds:
        y_pred = np.where(probs[:,1] > thres,1,0)
        #Apply desired utility function to y_preds, for example accuracy.
        thres_acc = metrics.accuracy_score(y_train.loc[test], y_pred)
        if thres_acc > best_accuracy:
            best_accuracy = thres_acc
            opt_threshold = thres

roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(7, 5))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'Train AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('Optimal Threshold: ', opt_threshold, ' gives Train Accuracy: ', best_accuracy)

plot_precision_recall_vs_threshold(p, r, t)


# In[ ]:


probs = lr_model.predict_proba(orig_testX)
fpr, tpr, t = metrics.roc_curve(orig_testY, probs[:,1])

roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'Test AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


y_pred = np.where(probs[:,1] > opt_threshold,1,0)
thres_acc = metrics.accuracy_score(orig_testY, y_pred, normalize = True)
print('Logistic Regression (Test Accuracy): ', thres_acc)

y_true = orig_testY.apply(lambda x: '<=50k' if x == 0 else '>50k')
y_preds = np.where(probs[:,1] > opt_threshold,'>50k','<=50k')

labels = ['<=50k', '>50k']
cm = confusion_matrix(y_true, y_preds, labels)
print(cm)
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


print('Precision Score: ', precision_score(orig_testY, y_pred))
print('Recall Score: ', recall_score(orig_testY, y_pred))
print('F1 Score: ', f1_score(orig_testY, y_pred))


# ### Logistic Regression using Downsampling

# In[ ]:


ds_trainX, ds_testX, ds_trainY, ds_testY = get_train_test_split(ds_X, ds_y)
logisticRegression(ds_trainX, ds_trainY, ds_testX, ds_testY)


# <b>Analysis</b>
# 
# We trained a logistic regression model on the original dataset, principal components, 
# optimal threshold for classification using cross-validation, and downsampling. From the above plots, 
# we can confirm that the best results were obtained with the downsampled dataset. 
# Accuracy score of 0.82, Precision score of 0.80, Recall score of 0.85, and F1 score of 0.83.
# 
# Precision: This tells when you predict something positive, how many times they were actually positive. whereas, 
# Recall: This tells out of actual positive data, how many times you predicted correctly.
# 
# Therefore, it would be right to say that logistic regression trained and tested using downsampling technique 
# gives us a relatively balanced model.

# ### Let's fit a model to find out features that have significant impact in predicting income category.

# In[ ]:


orig_trainX, orig_testX, orig_trainY, orig_testY = get_train_test_split(orig_X, orig_y)

# Training the model
model = XGBClassifier()
model_importance = model.fit(orig_trainX, orig_trainY)

# Plotting the Feature importance bar graph
plt.rcParams['figure.figsize'] = [14,12]
sns.set(style = 'darkgrid')
plot_importance(model_importance);


# ### Next, let's try a non-parametric model: Random Forest

# In[ ]:


ds_trainX, ds_testX, ds_trainY, ds_testY = get_train_test_split(ds_X, ds_y)


# Hyperparameter tuning relies more on experimental results than theory, and thus the best method to determine the optimal settings is to try many different combinations and evaluate the performance of each model.

# In[ ]:


rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# ### We will try adjusting the following set of hyperparameters:
# 
# * **n_estimators** = number of trees in the foreset
# * **max_features** = max number of features considered for splitting a node
# * **max_depth** = max number of levels in each decision tree
# * **min_samples_split** = min number of data points placed in a node before the node is split
# * **min_samples_leaf** = min number of data points allowed in a leaf node
# * **bootstrap** = method for sampling data points (with or without replacement)

# ### Hyperparameters Tuning:
# 
# We will generate a Random Hyperparameter Grid to narrow down the range of each hyperparameter using RandomizedSearchCV. The results of Random Search will be used in GridSearchCV to find the optimal parameters.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
# Fit the random search model
rf_random.fit(ds_trainX, ds_trainY)


# In[ ]:


rf_random.best_params_


# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    acc = metrics.accuracy_score(test_labels, predictions, normalize = True)
    metrics.classification_report(test_labels, predictions)
    print('Model Performance')
    print(metrics.classification_report(test_labels, predictions))
    return acc


# In[ ]:


# Evaluate Baseline Model

base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(ds_trainX, ds_trainY)
base_accuracy = evaluate(base_model, ds_testX, ds_testY)


# In[ ]:


best_random = rf_random.best_estimator_
random_acc = evaluate(best_random, ds_testX, ds_testY)


# **We achieved an improvement in accuracy of 2.46% with RandomSearchCV from the baseline model.**

# We can further improve our results by using grid search to focus on the most promising hyperparameters ranges found in the random search.

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_features': ['auto'],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(ds_trainX, ds_trainY)


# In[ ]:


grid_search.best_params_


# In[ ]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, ds_testX, ds_testY)


# It seems we have about maxed out performance

# There is not much to choose from Random Forest with GridSearchCV and RandomSearchCV. Moreover, Random Forest does not perform any better than the Logistic Regression to predict income category. 
