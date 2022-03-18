#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income
# **Predict whether income exceeds $50K/yr based on census data**  
# 
# 
# ---
# 
# 
# My objective of creating this kernel is to learn (and teach) to apply **Stacking** and see the improvement in performance
# 
# 
# **Contents:**
# 1. [Data Exploration and Visualization](#Data-Exploration-(EDA))
# 2. [Data Preprocessing](#Data-Preprocessing)
# 3. [Modeling](#Modeling-Part)
# 4. [Stacking](#Stacked-Generalization/Stacking)
# 5. [Predictions](#Predictions)
# 

# <u>Let's get Started!</u>
# 
# Importing required modules

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tqdm import tqdm


# Reading data file

# In[ ]:


data_all = pd.read_csv("/kaggle/input/adult-census-income/adult.csv")


# Take a peek at the data, see what columns are present and the data types, also check if headers are picked correctly

# In[ ]:


data_all.head()


# Hmm... looks like we have '?' character in data for missing values, we will need to replace that

# Checking how many (Rows, Columns) are present in data

# In[ ]:


data_all.shape


# replacing '?' with NaN for now, it would be easier to fill NaN later with other resonable estimates

# In[ ]:


data_all.replace('?',np.nan, inplace=True)


# Checking number of classes and class distribution, we should check if our classes are balanced or skewed, it would help us choose correct performance metric for evaluating model

# In[ ]:


pd.value_counts(data_all['income'])/data_all['income'].count()*100


# We can see that there are **two classes** the **classes are skewed.**
# 
# Will choose <b>F1-Score</b> (Precision and Recall) as the performance metric for our model

# Let's look at data characteristics in bit more detail, and look for potential outliers in the data

# In[ ]:


data_all.describe(include="all").T


# Looks like Workclass, Occupation and native.country has NaN values  
# We will use mode (most occuring) as imputing method to fill these NaNs  
# 
# **Age** seem alright, minimum is 17 and maximum is 90 (who's still working at 90?)  
# **hours.per.week** - minimum is only 1 ? We will check this.. could be an outlier if salary is >50K

# In[ ]:


data_all[(data_all['hours.per.week'] <= 4) & (data_all['income'] == '>50K')].sort_values(by='education.num')


# Well at-least people earning > 50 K are highly educated (so probably not outliers), also looks like their occupations in 'execs' or 'profs'

# let's fill all missing values with `mode`

# In[ ]:


# fill values
print(data_all['workclass'].mode())
print(data_all['occupation'].mode())
print(data_all['native.country'].mode())
data_all['workclass'].fillna(data_all['workclass'].mode()[0], inplace=True)
data_all['occupation'].fillna(data_all['occupation'].mode()[0], inplace=True)
data_all['native.country'].fillna(data_all['native.country'].mode()[0], inplace=True)


# Final check if we still have any NaN's in our data

# In[ ]:


data_all.info()


# # Data Exploration (EDA)

# Now we will do some basic **EDA**, we will visualize independent variable against dependent and try to determine some relationships. We will look out for any outliers or anything out of ordinary. We will also do some **feature engineering** steps (clubbing similar classes, thereby reducing features dimension)  
# 
# This section is a bit more detailed so please skip to [Data Preprocessing](#Data-Preprocessing) section directly if you want to

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="workclass",hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')


# **Observation 1:** Most of the employess are employed in Private sector and looks like people who are self employed (inc - i believe registered firms) are more likely to earn >50K a year  
# In general, count of people earning <=50K is more than count of people earning more

# Let's see some numbers - what percentage of people in each class earns more than 50K a year

# In[ ]:


data_all[data_all['income'] == '>50K']['workclass'].value_counts()/data_all['workclass'].value_counts()


# We will also club together some of the classes based on above information

# In[ ]:


data_all.replace({'workclass':{'Federal-gov':'fed_gov', 
                               'State-gov':'gov', 'Local-gov':'gov',
                               'Without-pay':'unemployed','Never-worked':'unemployed', 
                               'Self-emp-inc':'self-emp'}}
                 ,inplace=True)


# Occupation vs Income

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="occupation",hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')


# People at higher-up positions (Exec-managerial) have more probablity of earning more than >50K per year  
# As expected, blue-collared workforce (handlers etc..) have much less probabilty of earning >50K, some still seem to be earning > 50K but those are most probably older people

# Let's see some actual numbers

# In[ ]:


data_all[data_all['income'] == '>50K']['occupation'].value_counts()/data_all['occupation'].value_counts()


# We will again group some classes based on above numbers

# In[ ]:


data_all.replace({'occupation':{'Exec-managerial':'premium_pay', 
                                'Prof-specialty':'good_pay','Protective-serv':'good_pay', 'Tech-support':'good_pay',
                                'Craft-repair':'normal_pay', 'Sales':'normal_pay','Transport-moving':'normal_pay',
                                'Adm-clerical':'low_pay','Armed-Forces':'low_pay','Farming-fishing':'low_pay','Machine-op-inspct':'low_pay',
                                'Other-service':'poor_pay', 'Handlers-cleaners':'poor_pay', 'Priv-house-serv':'poor_pay'}},inplace=True)


# Education vs Income

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="education.num", hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')


# So proportion of people earning > 50K is more for highly educated people... only if I knew before ;)

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="marital.status",hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment='right')


# Not quite sure what to conclude.. seems like people happily married (or let's just say married) have higher chances of earning > 50K as compared to other (unhappy ?) people.. doesn't look quite intutive to me though  
# 'Never-married' people are probably younger folks and that's why they earn less

# Again, we will group some classes into one

# In[ ]:


data_all.replace({'marital.status':{'Never-married':'single','Divorced':'single','Separated':'single',
                                   'Widowed':'single','Married-spouse-absent':'single','Married-AF-spouse':'single',
                                   'Married-civ-spouse':'married'}},inplace=True)


# Gender vs Income

# In[ ]:


sns.set(rc={'figure.figsize':(4,4)})
sns.set_style("white")
sns.countplot(x="sex",hue="income", data=data_all, palette="Set3")
sns.despine()


# We can see the pay gap between genders, proportion of females earning > 50K is less as compared to males

# Do females earn more at higer age as compared to males?

# In[ ]:


sns.catplot(x="sex", y="age", kind="violin", inner="box", data=data_all[data_all['income'] == '>50K'], orient="v")


# Doesn't quite look like the case, both male and female have approximately same median age when they start to earn > 50K

# In[ ]:


sns.catplot(x="education", y="age", kind="violin", inner="box"
            , data=data_all[data_all['income'] == '>50K'], orient="v", aspect=2.5, height=5)


# We can see that less educated people earn > 50K/year at an older age as compared to others

# Is there any pay gap between people from different races ?

# In[ ]:


sns.set(rc={'figure.figsize':(6,4)})
sns.set_style("white")
sns.countplot(x="race", hue="income", data=data_all, palette="Set2")
sns.despine()


# let's see some numbers, graph isn't quite clear

# In[ ]:


data_all[data_all['income'] == '>50K']['race'].value_counts()/data_all['race'].value_counts()


# looks like some discrimination with black folks.. lets draw a graph comparing education, age, [White, black] races and income (only > 50K)

# In[ ]:


g = sns.catplot(x="education", y="age", hue="race", kind="violin", inner="quartile", split=True
            , data=data_all[(data_all['income'] == '>50K') & (data_all['race'].isin(['White','Black']))], orient="v", aspect=2.5, height=5)
g = g.set_xticklabels(rotation=60, horizontalalignment='right')


# Things kind of look alright to me.. more or less ok (with few exceptions, but that could actually be an issue iwth data too)  
# 
# Let's see if low proportion of people earning >50K is because black people are more employed in low paying occupation ?

# In[ ]:


g = sns.catplot(x="occupation", y="age", hue="race", kind="violin", inner="quartile", split=True
            , data=data_all[(data_all['race'].isin(['White','Black']))], orient="v", aspect=2, height=4)
g = g.set_xticklabels(rotation=60, horizontalalignment='right')


# This seems like the case, more of them (as compared to white) are employed in low and poor pay occupations.. so this explains it
# 
# Also, we will group together some classes

# In[ ]:


data_all.replace({'race':{'Black':'not-white', 'Asian-Pac-Islander':'not-white', 'Amer-Indian-Eskimo':'not-white'
                          ,'Other':'not-white'}}
                , inplace=True)


# relationship vs income

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
sns.countplot(x="relationship",hue="income", data=data_all, palette="Set2")
sns.despine()


# relation case looks similar (related) to 'marital.status' field.. people happily married are husband and wife (and this graph suggests the same, people who are married have more probability of earning > 50K).. we may remove one of the feature from our model

# In[ ]:


data_all.replace({'relationship':{'Husband':'family','Wife':'family','Not-in-family':'not_family','Own-child':'family',
                                  'Unmarried':'not_family','Other-relative':'not_family'}},inplace=True)


# Much awaited Age vs Income comparision

# In[ ]:


data_all['age_bins'] = pd.cut(data_all['age'], bins=4)
sns.set(rc={'figure.figsize':(6,4)})
sns.set_style("white")
sns.countplot(x="age_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# as expected, proportion of younger people earning more than 50K/year is less as compared to adult and senior people

# In[ ]:


data_all['hpw_bins'] = pd.cut(data_all['hours.per.week'], 4)
sns.set(rc={'figure.figsize':(6,4)})
sns.set_style("white")
sns.countplot(x="hpw_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# simple math.. the more you work more you would earn

# In[ ]:


data_all['cap_gain_bins'] = pd.cut(data_all['capital.gain'], [0,3000,7000,100000])
sns.set(rc={'figure.figsize':(5,3)})
sns.set_style("white")
sns.countplot(x="cap_gain_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# capital gains field is co-related with income, more gains means more income

# In[ ]:


data_all['cap_loss_bins'] = pd.cut(data_all['capital.loss'], [0,1000,5000])
sns.set(rc={'figure.figsize':(3,3)})
sns.set_style("white")
sns.countplot(x="cap_loss_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# Doesn't look like same case with capital loss though, income doesn't seem to be related with capital.loss directly

# Update native.country values.. using continent names rather than country names to club together classes

# In[ ]:


data_all.replace({'native.country':{'China': 'asia', 'Hong': 'asia', 'India': 'asia', 'Iran': 'asia', 'Cambodia': 'asia', 'Japan': 'asia', 'Laos': 'asia', 'Philippines': 'asia', 'Vietnam': 'asia', 'Taiwan': 'asia', 'Thailand': 'asia'}},inplace=True)
data_all.replace({'native.country':{'England': 'europe', 'France': 'europe', 'Germany': 'europe', 'Greece': 'europe', 'Holand-Netherlands': 'europe', 'Hungary': 'europe', 'Ireland': 'europe', 'Italy': 'europe', 'Poland': 'europe', 'Portugal': 'europe', 'Scotland': 'europe', 'Yugoslavia': 'europe'}},inplace=True)
data_all.replace({'native.country':{'Canada':'NAmerica','United-States':'NAmerica','Puerto-Rico':'NAmerica'}},inplace=True)
data_all.replace({'native.country':{'Columbia': 'SA', 'Cuba': 'SA', 'Dominican-Republic': 'SA', 'Ecuador': 'SA', 'El-Salvador': 'SA', 'Guatemala': 'SA', 'Haiti': 'SA', 'Honduras': 'SA', 'Mexico': 'SA', 'Nicaragua': 'SA', 'Outlying-US(Guam-USVI-etc)': 'SA', 'Peru': 'SA', 'Jamaica': 'SA', 'Trinadad&Tobago': 'SA'}},inplace=True)
data_all.replace({'native.country':{'South':'SA'}},inplace=True)


# In[ ]:


# Except North America
sns.set_style("white")
sns.countplot(x="native.country",hue="income", data=data_all[data_all['native.country'] != 'NAmerica'], palette="Set2")
sns.despine()


# No correlation with any country.. although proportion of people from south america region earning less than 50K is more as compared to people from other region

# In[ ]:


data_all.drop(columns=['age_bins','hpw_bins','cap_gain_bins','cap_loss_bins'],inplace=True)


# **Summarized findings**
# 1. Majority of people work in **Private**-Sector
# 2. People who are <b>self-employed </b>(self-employed-inc), or with <b>higher-education degree</b> (Prof-school, doctorate, masters) generally earn more than 50K a year,
# 3. People who are married have higher chances of earning more than >50K (may be this is related to age)
# 4. Approx 1/3rd of male earns more than 50K. For females this number is much lower
# 5. As age increases, proportion of people earning more than 50K increases
# 6. As number of work hours per week increases, proportion of people earning > 50K also increases, although people at higher-up positions tend to earn > 50K even when working for very few number of hours
# 7. More capital gain translates to more than 50K income
# 8. Proportion of black people earning > 50K is less than white people but when taking education into account, they seem to be paid fairely (with few exceptions). When compared with occupation we found that more of them are employed in low, poor pay occupations and hence proportion is less

# # Data Preprocessing
# 
# In next few steps, we will convert categorical variables to numerical, scale numerical variables, convert target variable to binary and drop 'education' variable, we will keep all other variables

# let's first convert income variable to binary 0 and 1 (as needed by model)

# In[ ]:


data_all.at[data_all[data_all['income'] == '<=50K'].index, 'income'] = 1
data_all.at[data_all[data_all['income'] == '>50K'].index, 'income'] = 0


# just check if everything got converted to 0 and 1 and class distribution is still same.. verification step

# In[ ]:


pd.value_counts(data_all['income'])/data_all['income'].count()*100


# let's **drop eduction** field as the same information is already presented by another numerical variable **education.num**

# In[ ]:


# we will not use 'education' column as we have 'education.num'
data_all.drop(columns=['education'], inplace=True)


# We will next do a train test split - 0.3 part for testing and rest for training  
# We will `stratify` our data on 'income' so that class distribution is same in both training and test

# In[ ]:


X,y = data_all.loc[:,data_all.columns != 'income'], data_all.loc[:,data_all.columns == 'income']
X_train, X_test, y_train, y_test = train_test_split(data_all.loc[:,data_all.columns != 'income']
                                                    , data_all.loc[:,data_all.columns == 'income']
                                                    , test_size=0.1, random_state=42, stratify=data_all['income'], shuffle=True)


# In[ ]:


X_train.shape, X_test.shape


# Scaling numerical variables  
# **Tip:** Make sure to use same scale for both training and testing (meaning don't fit separately on train and test sets)

# In[ ]:


#### Keep numerical
scalar_train = StandardScaler()
num_col_names = X_train.select_dtypes(include=['int64']).columns
num_col_df = X_train[num_col_names].copy()
scaled_num_col_df = scalar_train.fit_transform(num_col_df)
for i,j in enumerate(num_col_names):
    X_train[j] = scaled_num_col_df[:,i]

#---------------------------------
num_col_df_test = X_test[num_col_names].copy()
scaled_num_col_df = scalar_train.transform(num_col_df_test)
for i,j in enumerate(num_col_names):
    X_test[j] = scaled_num_col_df[:,i]


# In[ ]:


X_train.head()


# Looks like all variables scaled perfectly  
# 
# Converting rest of the variables to `category` type foe easier preprocessing in next steps

# In[ ]:


category_cols = list(set(X_train.columns.tolist()) - set(num_col_names))
for col in category_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')


# we will now create separate binary features for each workclass type, occupation type, race type, country name and marital status

# In[ ]:


X_train = pd.get_dummies(columns=category_cols, data=X_train, prefix=category_cols, prefix_sep="_",drop_first=True)

X_test = pd.get_dummies(columns=category_cols, data=X_test, prefix=category_cols, prefix_sep="_",drop_first=True)


# Let's just see what all and how manu columns are present in our final dataset which we will use for training

# In[ ]:


print(X_train.columns, X_train.shape)


# # Modeling Part
# 
# 

# In[ ]:


X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')


# Let's try logistic regression

# In[ ]:


# performance on logistic regression
log_clf = LogisticRegression(solver='liblinear', random_state=42, max_iter=500)

log_clf.fit(X_train, y_train.ravel())

test_pred = log_clf.predict(X_test)


# In[ ]:


y_test = np.asarray(y_test).astype('float32').ravel()


# In[ ]:


prec_t = precision_score(y_test, test_pred)
rec_t = recall_score(y_test, test_pred)
print("Test SK_Precision: %.3f" % prec_t)
print("Test SK_Recall: %.3f" % rec_t)
print("Test F1 Score: %.3f" % ((2*prec_t*rec_t)/(prec_t + rec_t)))
print("Test Accuracy: %.3f" % accuracy_score(y_test, test_pred))


# **Let's see if this score can be improved**

# # Stacked Generalization/Stacking  
# 
# * We will split the training data in 4 parts - train_a, train_b, train_c, train_d  
# * Fit a first-stage model on train_a+train_b+train_c combined and then predict on train_d. Repeat this step 3 more times to create predictions for aall 4 train sets.  
# * Repeat the previous step for other models as well. I have randomly selected NN, RF, ET, GBM and SVM
# * train a meta model (logistic regression) on the predictions of previous models

# In[ ]:


skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X_train, y_train)

nn = []; rf = []; et = []; gb = []; svc = []; y_1 = []

for train_index, test_index in tqdm(skf.split(X_train, y_train)):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train_1, X_test_1 = X_train[train_index], X_train[test_index]
    y_train_1, y_test_1 = y_train[train_index], y_train[test_index]
    
    # Neural network
    clf = MLPClassifier(hidden_layer_sizes=(32,16,8), random_state=42)
    clf.fit(X_train_1, y_train_1.ravel())
    train_pred = clf.predict_proba(X_test_1)
    nn.extend(list(train_pred[:,1].ravel()))
    
    # RandomForest
    clf = RandomForestClassifier(max_depth=20, random_state=42, n_estimators=100, min_samples_split=0.01)
    clf.fit(X_train_1, y_train_1.ravel())
    train_pred = clf.predict_proba(X_test_1)
    rf.extend(list(train_pred[:,1].ravel()))
    
    # ExtraTrees
    clf = ExtraTreesClassifier(max_depth=20, random_state=42, min_samples_split=0.01)
    clf.fit(X_train_1, y_train_1.ravel())
    train_pred = clf.predict_proba(X_test_1)
    et.extend(list(train_pred[:,1].ravel()))
    
    # GBM
    clf = GradientBoostingClassifier(min_samples_split=0.01, random_state=42, n_estimators=100)
    clf.fit(X_train_1, y_train_1.ravel())
    train_pred = clf.predict_proba(X_test_1)
    gb.extend(list(train_pred[:,1].ravel()))
    
    # SVM
    clf = SVC(random_state=42, C=2, kernel='linear', probability=True)
    clf.fit(X_train_1, y_train_1.ravel())
    train_pred = clf.predict_proba(X_test_1)
    svc.extend(list(train_pred[:,1].ravel()))  
    
    y_1.extend(list(y_test_1.ravel()))
    

print(len(nn), len(rf), len(et), len(gb), len(svc), len(y_1))


# In[ ]:


X_2 = pd.DataFrame.from_dict({'NN':nn, 'RF':rf, 'ET':et, 'GB':gb, 'SVC':svc, 'Y':y_1})
print(X_2.shape)
X_2.head()


# In[ ]:


cv_splits = StratifiedShuffleSplit(n_splits=5, test_size=0.3, train_size=0.7, random_state=42)

alg = LogisticRegression(solver='liblinear', random_state=42, max_iter=500)

cross_val = cross_validate(alg, 
                           X_2.loc[:, ~X_2.columns.isin(['Y'])], 
                           X_2['Y'],
                           cv  = cv_splits,
                           scoring = ['precision', 'recall', 'f1'],
                           return_train_score=True, return_estimator=False
                          )


# In[ ]:


print(cross_val['train_precision'].mean())
print(cross_val['train_recall'].mean())
print(cross_val['train_f1'].mean())
print("--")
print(cross_val['test_precision'].mean())
print(cross_val['test_recall'].mean())
print(cross_val['test_f1'].mean())


# # Predictions  
# 
# * Use the same steps for test set too
# * Train the first-stage models on whole training set and then predict on test set
# * Use meta-model to create final predictions

# In[ ]:


nnt = []; rft = []; ett = []; gbt = []; svct = []; y_1t = []

# Neural network
clf = MLPClassifier(hidden_layer_sizes=(32,16,8), random_state=42)
clf.fit(X_train, y_train.ravel())
train_pred = clf.predict_proba(X_test)
nnt.extend(list(train_pred[:,1].ravel()))

# RandomForest
clf = RandomForestClassifier(max_depth=20, random_state=42, n_estimators=100, min_samples_split=0.01)
clf.fit(X_train, y_train.ravel())
train_pred = clf.predict_proba(X_test)
rft.extend(list(train_pred[:,1].ravel()))

# ExtraTrees
clf = ExtraTreesClassifier(max_depth=20, random_state=42, min_samples_split=0.01)
clf.fit(X_train, y_train.ravel())
train_pred = clf.predict_proba(X_test)
ett.extend(list(train_pred[:,1].ravel()))

# GBM
clf = GradientBoostingClassifier(random_state=42, n_estimators=100, min_samples_split=0.01)
clf.fit(X_train, y_train.ravel())
train_pred = clf.predict_proba(X_test)
gbt.extend(list(train_pred[:,1].ravel()))

# SVM
clf = SVC(random_state=42, C=1, kernel='linear', probability=True)
clf.fit(X_train, y_train.ravel())
train_pred = clf.predict_proba(X_test)
svct.extend(list(train_pred[:,1].ravel()))


y_1t.extend(list(y_test))


print(len(nnt), len(rft), len(ett), len(gbt), len(svct), len(y_1t))


# In[ ]:


X_test_new = pd.DataFrame.from_dict({'NN':nnt, 'RF':rft, 'ET':ett, 'GB':gbt, 'SVC':svct, 'Y':y_1t})
print(X_test_new.shape)
X_test_new.head()


# In[ ]:


alg.fit(X_2.loc[:, ~X_2.columns.isin(['Y'])], X_2['Y'])
test_pred = alg.predict(X_test_new.loc[:, ~X_test_new.columns.isin(['Y'])])


# In[ ]:


prec_t = precision_score(X_test_new['Y'].values.ravel().tolist(), test_pred.ravel().tolist())
rec_t = recall_score(X_test_new['Y'].values.ravel().tolist(), test_pred.ravel().tolist())
print("Test SK_Precision: %.3f" % prec_t)
print("Test SK_Recall: %.3f" % rec_t)
print("Test F1 Score: %.3f" % ((2*prec_t*rec_t)/(prec_t + rec_t)))
print("Test Accuracy: %.3f" % accuracy_score(X_test_new['Y'].values.ravel().tolist(), test_pred.ravel().tolist()))


# **Change**  
# 
# There's about **1.4% improvement in f1 score **and **2.3% in accuracy**, we might be able to improve further by tuning classifiers

# In[ ]:




