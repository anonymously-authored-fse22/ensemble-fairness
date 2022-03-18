#!/usr/bin/env python
# coding: utf-8

# # Adult Income Prediction notebook
# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). *The prediction task is to determine whether a person makes over  USD 50K a year.*  
# This is a binary classification problem where we have to predict whether a person earns over $50k per year or not. The scoring function to optimize is accuracy. The notebook follows the following steps to analyse the data and build predictive models.
# - Data cleaning and preprocessing
# - Exploratory data analysis
# - Modelling: I have tried out different classification algorithms.
#   - Random forest
#   - XGBoost
#   - Naive Bayes
#   - Logistic regression
#   - CatBoost  
# These models were then optimized by tuning the hyper-parameters through Grid Search, keeping a close check on the cross-validation scores to prevent overfitting. Thereafter I also tried out stacking different models together to improve the accuracy but it didn't improve the accuracy sigificantly.
# - Finally I explored oversampling techniques including SMOTE (in progress).
# 
# Please feel free to suggest/comment.
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[3]:


sns.set() ##set defaults


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv("../input/adult.csv")


# In[6]:


data.shape


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


data['workclass'].value_counts()


# In[10]:


data['income'].value_counts()    


# #### Encode the target variable to binary

# In[11]:


data['income'] = data['income'].apply(lambda inc: 0 if inc == "<=50K" else 1) # Binary encoding of the target variable


# ## Exploratory analysis

# In[12]:


plt.figure(figsize=(10,5))
sns.countplot(data['income'])


# As one can see, there is considerable class imbalance in the target variable, i.e. income. This is also intuitively obvious as one expects fewer 'rich' people (earning>50k/annum) than 'not-so-rich' people (earning <50k/annum). Therefore we might need to consider over-sampling techniques in our ML model to improve our accuracy.

# In[13]:


plt.figure(figsize=(14,6))
sns.countplot(data['marital.status'])


# Those with `Never-married` and `Married-civ-spouse` labels dominate the dataset.

# In[14]:


plt.figure(figsize=(15,6))
ax=sns.barplot(x='marital.status',y='income',data=data,hue='sex')
ax.set(ylabel='Fraction of people with income > $50k')
data['marital.status'].value_counts()


# The above plot shows the the fraction of people earning more than $50k per annum, grouped by their marital status and gender. The data shows that married people have a higher %age of high-earners, compared to those who either never married or are widowed/divorced/separated. The black lines indicate 2 standard deviations (or 95\% confidence interval) in the data set. The married spouses of armed forces personnel have a much higher variation in their income compared to civil spouses because of low-number statistics.

# In[15]:


plt.figure(figsize=(12,6))
sns.countplot(data['workclass'])


# In[16]:


plt.figure(figsize=(12,6))
ax=sns.barplot('workclass', y='income', data=data, hue='sex')
ax.set(ylabel='Fraction of people with income > $50k')


# In[17]:


plt.figure(figsize=(10,8))  
sns.heatmap(data.corr(),cmap='Accent',annot=True)
#data.corr()
plt.title('Heatmap showing correlations between numerical data')


# In[18]:


plt.figure(figsize=(12,6))
sns.boxplot(x="income", y="age", data=data, hue='sex')
#data[data['income']==0]['age'].mean()


# The mean age of people earning more than 50k per annum is around 44 whereas the mean age of of those earning less than 50k per annum is 36.

# In[19]:


norm_fnl = (data["fnlwgt"] - data['fnlwgt'].mean())/data['fnlwgt'].std()
plt.figure(figsize=(8,6))
sns.boxplot(x="income", y=norm_fnl, data=data)


# As evident from the plot above, there are many outliers in the `fnlwgt` column and this feature is uncorrelated with `income`, our target variable. The correlation coefficient (which one can read from the heatmap) is -0.0095. The number of outliers, i.e. the number of records which are more than 2 s.d's away from the mean, is 1249.

# In[20]:


data[norm_fnl>2].shape


# In[21]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x='sex',y='income',data=data)
ax.set(ylabel='Fraction of people with income > $50k')


# The fraction of rich among men is significantly higher than that among women.

# In[22]:


plt.figure(figsize=(12,6))
sns.boxplot(x='income',y ='hours.per.week', hue='sex',data=data)


# On the basis of the above plot, we can only conclude that males put in slightly more hours per week than women on an average.

# In[23]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x='marital.status',y='hours.per.week',data=data,hue='sex')
ax.set(ylabel='mean hours per week')


# In[24]:


plt.figure(figsize=(10,6))
ax = sns.barplot(x='income', y='education.num',hue='sex', data=data)
ax.set(ylabel='Mean education')


# The `education.num` is label encoded such that a higher number corresponds to a higher level of education. As on would naïvely expect, people who earn more (>50k per annum) are also highly educated. The mean education level for `income=1` class is between 11 (Assoc-voc) and 12 (Assoc-acdm) whereas that for the `income=0` class is between 9 (HS-grad) and 10 (Some-college).

# In[25]:


print(data['race'].value_counts())
plt.figure(figsize=(12,6))
ax=sns.barplot(x='race',y='income',data=data)
ax.set(ylabel='Fraction of people with income > $50k')


# In[26]:


plt.figure(figsize=(12,6))
sns.jointplot(x=data['capital.gain'], y=data['capital.loss'])
#print(data[((data['capital.gain']!=0) & (data['capital.loss']!=0))].shape)


# In[27]:


plt.figure(figsize=(12,8))
sns.distplot(data[(data['capital.gain']!=0)]['capital.gain'],kde=False, rug=True)


# In[28]:


plt.figure(figsize=(12,8))
sns.distplot(data[(data['capital.loss']!=0)]['capital.loss'], kde=False,rug=True)


# In[29]:


plt.figure(figsize=(20,6))
ax=sns.barplot(x='occupation', y='income', data=data)
ax.set(ylabel='Fraction of people with income > $50k')


# In[30]:


print(data['native.country'].value_counts())
not_from_US = np.sum(data['native.country']!='United-States')
print(not_from_US, 'people are not from the United States')


# #### Convert the `native.country` feature to binary since there is a huge imbalance in this feature

# In[31]:


data['native.country'] = (data['native.country']=='United-States')*1
#data['US_or_not']=np.where(data['native.country']=='United-States',1,0)


# In[32]:


data.select_dtypes(exclude=[np.number]).head()


# In[33]:


#Replace all '?'s with NaNs.
data = data.applymap(lambda x: np.nan if x=='?' else x)


# In[34]:


data.isnull().sum(axis=0) # How many issing values are there in the dataset?


# In[35]:


data.shape[0] - data.dropna(axis=0).shape[0]   # how many rows will be removed if I remove all the NaN's?


# In[36]:


data = data.dropna(axis=0) ## Drop all the NaNs


# In[37]:


#data.education.value_counts()  # I will label-encode the education column since it is an ordinal categorical variable


# In[38]:


## This computes the fraction of people by country who earn >50k per annum
#mean_income_bycountry_df = data[['native.country','income']].groupby(['native.country']).mean().reset_index()


# In[39]:


#edu_encode_dict = {'Preschool':0,'1st-4th':1, '5th-6th':2, '7th-8th':3, '9th':4, '10th':5,
#                  '11th':6, '12th':7, 'HS-grad':8, 'Some-college':9, 'Bachelors':10, 'Masters':11, 'Assoc-voc':12, 
#                   'Assoc-acdm':13, 'Doctorate':14, 'Prof-school':15}

#data['education'] = data['education'].apply(lambda ed_level: edu_encode_dict[ed_level])


# ### One-hot encoding of the categorical columns

# In[40]:


data = pd.get_dummies(data,columns=['workclass','sex', 'marital.status',
                                    'race','relationship','occupation'],
               prefix=['workclass', 'is', 'is', 'race_is', 'relation', 'is'], drop_first=True)
### native country is ignored because that feature will be dropped later


# In[41]:


plt.figure(figsize=(20,12))
sns.heatmap(data.corr())


# In[42]:


data.select_dtypes(exclude=[np.number]).shape


# In[43]:


data.groupby('income').mean()


# In[44]:


data.shape


# In[45]:


y = data.income
X = data.drop(['income', 'education', 'native.country', 'fnlwgt'],axis=1)


# - `income` is dropped from X because it is the target variable.
# - `Education` is dropped because it is already label-encoded in `education.num`. One can notice the high correlation between `education` and `education.num` in the heatmap.
# - `native country` is dropped because it showed very little feature importance in random forest classifer.
# - `fnlwgt` is dropped because it has no correlation with `income`.

# ## Modelling
# This section explores different classification algorithms to maximise the accuracy for predicting income of a person (> 50k/yr or < 50k/yr).

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier as xgb
from sklearn import metrics


# ### Baseline model
# In the baseline model, we predict the minority class for all our train and test (or validation) examples. The resulting accuracy will serve as a benchmark for the ML models. In other words, the sophisticated ML models should have an accuracy which should at least better the baseline one.

# In[49]:


baseline_train = np.zeros(y_train.shape[0])
baseline_test = np.zeros(y_test.shape[0])
print('Accuracy on train data: %f%%' % (metrics.accuracy_score(y_train, baseline_train)))
print('Accuracy on test data: %f%%' %  (metrics.accuracy_score(y_test, baseline_test)))


# ### Random Forest classifier

# In[50]:


rfmodel = RandomForestClassifier(n_estimators=300,oob_score=True,min_samples_split=5,max_depth=10,random_state=10)
rfmodel.fit(X_train,y_train)
print(rfmodel)


# In[51]:


def show_classifier_metrics(clf, y_train=y_train,y_test=y_test, print_classification_report=True, print_confusion_matrix=True):
    print(clf)
    if print_confusion_matrix:
        print('confusion matrix of training data')
        print(metrics.confusion_matrix(y_train, clf.predict(X_train)))
        print('confusion matrix of test data')
        print(metrics.confusion_matrix(y_test, clf.predict(X_test)))
    if print_classification_report:
        print('classification report of test data')
        print(metrics.classification_report(y_test, clf.predict(X_test)))
    print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, clf.predict(X_test))*100))
    print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train, clf.predict(X_train))*100))
    print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, clf.predict(X_test))))


# In[52]:


show_classifier_metrics(rfmodel,y_train)
print('oob score = %f'% rfmodel.oob_score_)


# In[53]:


importance_list = rfmodel.feature_importances_
name_list = X_train.columns
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))
plt.figure(figsize=(20,10))
plt.barh(range(len(name_list)),importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)
plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature')
plt.show()


# ### Random forest: Grid Search and cross-validation

# In[54]:


from sklearn.model_selection import cross_val_score, GridSearchCV


# In[55]:


def grid_search(clf, parameters, X, y, n_jobs= -1, n_folds=4, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func,verbose =2)
    else:
        print('Doing grid search')
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds, verbose =1)
    gs.fit(X, y)
    print("mean test score (weighted by split size) of CV rounds: ",gs.cv_results_['mean_test_score'] )
    print ("Best parameter set", gs.best_params_, "Corresponding mean CV score",gs.best_score_)
    best = gs.best_estimator_
    return best


# In[56]:


rfmodel2 = RandomForestClassifier(min_samples_split=5,oob_score=True, n_jobs=-1,random_state=10)
parameters = {'n_estimators': [100,200,300], 'max_depth': [10,13,15,20]}
rfmodelCV = grid_search(rfmodel2, parameters,X_train,y_train)


# In[57]:


rfmodelCV.fit(X_train,y_train)
show_classifier_metrics(rfmodelCV,y_train)
print('oob score = %f'% rfmodelCV.oob_score_)


# ## XGBoost 

# In[58]:


from xgboost.sklearn import XGBClassifier


# In[59]:


param = {}
param['learning_rate'] = 0.1
param['verbosity'] = 1
param['colsample_bylevel'] = 0.9
param['colsample_bytree'] = 0.9
param['subsample'] = 0.9
param['reg_lambda']= 1.5
param['max_depth'] = 5
param['n_estimators'] = 400
param['seed']=10
xgb= XGBClassifier(**param)
xgb.fit(X_train, y_train, eval_metric=['error'], eval_set=[(X_train, y_train),(X_test, y_test)],early_stopping_rounds=40)


# In[60]:


show_classifier_metrics(xgb,y_train)


# In[61]:


importance_list = xgb.feature_importances_
name_list = X_train.columns
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))
plt.figure(figsize=(20,10))
plt.barh(range(len(name_list)),importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)
plt.xlabel('Relative Importance in XGBoost')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature')
plt.show()


# ### Grid search with cross validation: XGBoost model

# In[62]:


xgbmodel2 = XGBClassifier(seed=42)
param = {
'learning_rate': [0.1],#[0.1,0.2],
#'verbosity': [1],
'colsample_bylevel': [0.9],
'colsample_bytree': [0.9],
'subsample' : [0.9],
'n_estimators': [300],
'reg_lambda': [1.5,2,2.5],
'max_depth': [3,5,7],
 'seed': [10]   
}
xgbCV = grid_search(xgbmodel2, param,X_train,y_train)


# In[63]:


xgbCV.fit(X_train, y_train, eval_metric=['error'], eval_set=[(X_train, y_train),(X_test, y_test)],early_stopping_rounds=40)


# In[64]:


show_classifier_metrics(xgbCV,y_train)


# In[65]:


#X_test.iloc[np.where(y_test != xgbCV.predict(X_test))]


# ## Logistic regression

# In[66]:


from sklearn.linear_model import LogisticRegression


# In[67]:


param = {
'C': [3,5,10], 
'verbose': [1],
    'max_iter': [100,200,500,700]
}   
logreg = LogisticRegression(random_state=10)
logreg_grid = grid_search(logreg, param, X_train,y_train, n_folds=3)


# In[68]:


logreg_grid.fit(X_train, y_train)


# In[69]:


show_classifier_metrics(logreg_grid)


# ## Naive Bayes

# In[70]:


from sklearn.naive_bayes import GaussianNB


# In[71]:


NBmodel = GaussianNB()


# In[72]:


NBmodel.fit(X_train, y_train)


# In[73]:


NBmodel.predict(X_test)


# In[74]:


show_classifier_metrics(NBmodel,y_train)


# ### Stacked model

# In[75]:


def create_stacked_dataset(clfs,modelnames, X_train=X_train,X_test=X_test):
    X_train_stack, X_test_stack = X_train, X_test
    for clf,modelname in zip(clfs,modelnames):
        temptrain = pd.DataFrame(clf.predict(X_train),index = X_train.index,columns=[modelname+'_prediction'])
        temptest  = pd.DataFrame(clf.predict(X_test),index = X_test.index,columns=[modelname+'_prediction'])
        X_train_stack = pd.concat([X_train_stack, temptrain], axis=1)
        X_test_stack = pd.concat([X_test_stack, temptest], axis=1)
    return (X_train_stack,X_test_stack)


# In[76]:


X_train_stack,X_test_stack = create_stacked_dataset([rfmodelCV,logreg_grid,xgbCV],modelnames=['rfmodel','logreg', 'xgb'])


# In[77]:


X_train_stack.head(2)


# In[78]:


param = {}
param['learning_rate'] = 0.1
param['verbosity'] = 1
param['colsample_bylevel'] = 0.9
param['colsample_bytree'] = 0.9
param['subsample'] = 0.9
param['reg_lambda']= 1.5
param['max_depth'] = 5#10
param['n_estimators'] = 400
param['seed']=10
xgbstack= XGBClassifier(**param)
xgbstack.fit(X_train_stack, y_train, eval_metric=['error'], eval_set=[(X_train_stack, y_train),(X_test_stack, y_test)],early_stopping_rounds=30)


# In[79]:


print(metrics.classification_report(y_test, xgbstack.predict(X_test_stack)))
print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, xgbstack.predict(X_test_stack))*100))
print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train, xgbstack.predict(X_train_stack))*100))


# ### Stacked model Grid Search

# In[80]:


xgbstackCV = XGBClassifier(seed=10)
param_grid = {}
param_grid['learning_rate'] = [0.1]
param_grid['colsample_bylevel'] = [0.9]
param_grid['colsample_bytree'] = [0.9]
param_grid['subsample'] = [0.9]
param_grid['n_estimators'] = [300]
param_grid['reg_lambda']= [1.5]
param_grid['seed'] =[10]
param_grid['max_depth'] = [3,5,8,10]
xgbstackCV_grid = grid_search(xgbstackCV, param_grid,X_train_stack,y_train)


# In[81]:


xgbstackCV_grid.fit(X_train_stack, y_train, eval_metric=['error'], eval_set=[(X_train_stack, y_train),(X_test_stack, y_test)],early_stopping_rounds=30)


# In[82]:


print(metrics.classification_report(y_test, xgbstack.predict(X_test_stack)))
print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, xgbstack.predict(X_test_stack))*100))
print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train, xgbstack.predict(X_train_stack))*100))


# In[83]:


from catboost import CatBoostClassifier


# In[84]:


catb = CatBoostClassifier(learning_rate=0.3,iterations=400,verbose=0,random_seed=10,eval_metric='Accuracy',rsm=0.9)


# In[85]:


catb.fit(X_train,y_train,eval_set=[(X_train,y_train), (X_test,y_test)],early_stopping_rounds=40)


# In[86]:


show_classifier_metrics(catb)


# In[87]:


### Catboost grid search

catbCV = CatBoostClassifier(verbose=0,random_seed=10,eval_metric='Accuracy')
param_grid = {}
param_grid['learning_rate'] = [0.1]#, 0.3]
param_grid['rsm'] = [0.9]
#param_grid['subsample'] = [0.9]
param_grid['iterations'] = [200,300]
param_grid['reg_lambda']= [3] #2
param_grid['depth'] = [8,10]#5
catbCV_grid = grid_search(catbCV, param_grid,X_train,y_train)


# In[88]:


catbCV_grid.fit(X_train,y_train,eval_set=[(X_train,y_train), (X_test,y_test)],early_stopping_rounds=30)


# In[89]:


show_classifier_metrics(catbCV_grid)


# In[90]:


from imblearn.over_sampling import RandomOverSampler


# In[91]:


np.sum(y_train)/y_train.shape[0]


# In[92]:


ros = RandomOverSampler(random_state=1,sampling_strategy=0.8)


# In[93]:


X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


# In[94]:


catb_ros = CatBoostClassifier(learning_rate=0.1,iterations=400,reg_lambda=2,verbose=0,random_seed=10,eval_metric='Accuracy')


# In[95]:


catb_ros.fit(X_resampled,y_resampled,eval_set=[(X_resampled,y_resampled), (X_test,y_test)],early_stopping_rounds=40)


# In[96]:


print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, catb_ros.predict(X_test))*100))
print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_resampled, catb_ros.predict(X_resampled))*100))
print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, catb_ros.predict(X_test))))


# ### SMOTE

# In[97]:


from imblearn.over_sampling import SMOTE


# In[98]:


smt = SMOTE(random_state=10,sampling_strategy=0.7)
X_train_smt, y_train_smt = smt.fit_sample(X_train, y_train)


# In[99]:


y_train.value_counts()


# In[100]:


np.bincount(y_train_smt)


# In[101]:


catb_smote = CatBoostClassifier(learning_rate=0.1,iterations=400,reg_lambda=2,verbose=0,random_seed=10,eval_metric='Accuracy')


# In[102]:


catb_smote.fit(X_train_smt,y_train_smt,eval_set=[(X_train_smt,y_train_smt), (X_test,y_test)],early_stopping_rounds=40)


# In[103]:


print(metrics.classification_report(y_test, catb_smote.predict(X_test)))
print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, catb_smote.predict(X_test))*100))
print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train_smt, catb_smote.predict(X_train_smt))*100))
print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, catb_ros.predict(X_test))))


# #### SMOTE with XGBoost

# In[104]:


param = {}
param['learning_rate'] = 0.1
param['verbosity'] = 1
param['colsample_bylevel'] = 0.9
param['colsample_bytree'] = 0.9
param['subsample'] = 0.9
param['reg_lambda']= 1.5
param['max_depth'] = 5
param['n_estimators'] = 400
param['seed']=10
xgb_smote= XGBClassifier(**param)
xgb_smote.fit(X_train_smt, y_train_smt, eval_metric=['error'], eval_set=[(X_train_smt, y_train_smt),(X_test.values, y_test.values)],early_stopping_rounds=30)


# In[105]:


print(metrics.classification_report(y_test, xgb_smote.predict(X_test.values)))
print('Accuracy on test data: %f%%' % (metrics.accuracy_score(y_test, xgb_smote.predict(X_test.values))*100))
print('Accuracy on training data: %f%%' % (metrics.accuracy_score(y_train_smt, xgb_smote.predict(X_train_smt))*100))
print('Area under the ROC curve : %f' % (metrics.roc_auc_score(y_test, xgb_smote.predict(X_test.values))))


# In[106]:




