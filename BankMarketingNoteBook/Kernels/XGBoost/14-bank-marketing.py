#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing

# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The csv comes with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in (Moro et al., 2014). The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
# 
# Attribute Information:
# 
# Input variables:
# #### Bank client data
# 1 - age (numeric)
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# #### Related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone')
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 11 - duration: last contact duration, in seconds (numeric).
# #### Other attributes
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# #### Social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# Output variable:
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# ## 1) Load relevant packages

# In[ ]:


import math
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter as FF, StrMethodFormatter as SMF
import seaborn as sns
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, r2_score, classification_report
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ## 2) Load Data

# In[ ]:


df = pd.read_csv('../input/bank-marketing/bank-additional-full.csv',sep = ';')
pd.set_option("display.max_columns", None)
df.head()


# In[ ]:


df.describe


# In[ ]:


df.info()


# ## 3) Data Cleaning and Feature Engineering
# The data is only appears clean, but after examining the categorical variables I found that many of them had unknown values. It's hard to say what to do since many of the variables didn't have a clear value to replace the unknown value to. For now, I will examine the variables as they are while changing the target variable into a quantitative one. 

# In[ ]:


#Term Deposit Subscription (Target) 
# Yes = 1 and No = 0
df['y'] = 1 * (df['y']== 'yes')


# ## 4) Exploratory Data Analysis
# 

# From the initial heatmap, there doesn't appear to be a strong correlation between a term deposit subscription and our quantitative variables. Duration and previous have the strongest correlations, but we will not keep duration in mind because of the concern from the dataset provider.

# In[ ]:


sns.set()

#Graph 1
g1 = sns.heatmap(df.corr(), cmap='Blues')


# In[ ]:


#Graph 2
g2 = sns.histplot(data=df,x='age', hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscription vs Client Age', fontsize = 24)

#X-axis
ax.set_xlabel("Age", fontsize = 24)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
ax.set_ylim(0, 2000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Previous count
previous_count = df['age'].value_counts().sort_index().to_frame().rename(columns = {'age': "# of Clients"})
previous_count.index.name = "Age"
previous_count.T


# In[ ]:


#Graph 3
g3 = sns.countplot(data=df,x='previous', hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscription vs Number of Previous Contacts', fontsize = 24)

#X-axis
ax.set_xlabel("Number of Contacts", fontsize = 24)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
#ax.set_ylim(0, 5000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Previous count
previous_count = df['previous'].value_counts().sort_index().to_frame().rename(columns = {'previous': "# of Clients"})
previous_count.index.name = "Previous"
previous_count.T


# I'm using the same graph as above, but I wanted to zoom into the number of contacts 1-7. Clearly, the chances of the client subscribing to a deposit increases as you have more previous contacts.

# In[ ]:


#Graph 4
g4 = sns.countplot(data=df,x='previous', hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscription vs Number of Previous Contacts', fontsize = 24)

#X-axis
ax.set_xlabel("Number of Contacts", fontsize = 24)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
ax.set_ylim(0, 3700)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Age count
age_count = df['previous'].value_counts().sort_index().to_frame().rename(columns = {'previous': "# of Clients"})
age_count.index.name = "# of Previous Contacts"
age_count.T


# In[ ]:


#Graph 5
g5 = sns.countplot(data = df, x = 'job',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Job', fontsize = 24)

#X-axis
ax.set_xlabel("Job", fontsize = 24)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
ax.set_ylim(0, 10000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Job count
job_count = df['job'].value_counts().sort_index().to_frame().rename(columns = {'job': "# of Clients"})
job_count.index.name = "Job"
job_count.T


# In[ ]:


#Graph 6
g6 = sns.countplot(data = df, x = 'marital',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Marital Status', fontsize = 24)

#X-axis
ax.set_xlabel("Marital Status", fontsize = 24)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
ax.set_ylim(0, 25000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Marital count
marital_count = df['marital'].value_counts().sort_index().to_frame().rename(columns = {'marital': "# of Clients"})
marital_count.index.name = "Marital Status"
marital_count.T


# In[ ]:


#Graph 7
g7 = sns.countplot(data = df, x = 'education',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Education', fontsize = 24)

#X-axis
ax.set_xlabel("Education", fontsize = 24)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
#ax.set_ylim(0, 12000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Education count
education_count = df['education'].value_counts().sort_index().to_frame().rename(columns = {'education': "# of Clients"})
education_count.index.name = "Education"
education_count.T


# In[ ]:


#Graph 8
g8 = sns.countplot(data = df, x = 'default',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Credit Default', fontsize = 24)

#X-axis
ax.set_xlabel("Credit Default", fontsize = 24)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
#ax.set_ylim(0, 12000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Education count
marital_count = df['default'].value_counts().sort_index().to_frame().rename(columns = {'default': "# of Clients"})
marital_count.index.name = "Default"
marital_count.T


# In[ ]:


#Graph 9
g9 = sns.countplot(data = df, x = 'housing',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Housing Loan', fontsize = 24)

#X-axis
ax.set_xlabel("Housing Loan", fontsize = 24)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
#ax.set_ylim(0, 12000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Education count
marital_count = df['housing'].value_counts().sort_index().to_frame().rename(columns = {'housing': "# of Clients"})
marital_count.index.name = "Housing Loan"
marital_count.T


# In[ ]:


#Graph 10
g10 = sns.countplot(data = df, x = 'loan',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Personal Loans', fontsize = 24)

#X-axis
ax.set_xlabel("Personal Loan", fontsize = 24)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
#ax.set_ylim(0, 12000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Education count
marital_count = df['loan'].value_counts().sort_index().to_frame().rename(columns = {'loan': "# of Clients"})
marital_count.index.name = "Personal Loans"
marital_count.T


# In[ ]:


#Graph 11
g11 = sns.countplot(data = df, x = 'campaign',hue = 'y', palette = 'viridis_r')
plt.gcf().set_size_inches(16, 9)

#Axes
ax = plt.gca()

#Title
ax.set_title('Client Subscriptions vs Campaign', fontsize = 24)

#X-axis
ax.set_xlabel("Campaign", fontsize = 24)
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')

#Y-axis
ax.set_ylabel("Number of Clients", fontsize = 24)
#ax.set_ylim(0, 12000)
#ax.yaxis.set_major_formatter(SMF('${x:,.0f}'))
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

#Education count
marital_count = df['campaign'].value_counts().sort_index().to_frame().rename(columns = {'campaign': "# of Clients"})
marital_count.index.name = "Campaign"
marital_count.T


# ## 5) Data Modeling

# #### Classification models used:
# - Random Forest Classifier
# - Decision Tree Classifier
# - Support Vector Classifier
# - K-Nearest Neighbors
# - Logistic Regression Model
# - Gausian Naive Bayes
# - XGBoost
# - Gradient Boosting
# 

# In[ ]:


le = LabelEncoder()

cat_var =['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for i in cat_var:
    df[i]= le.fit_transform(df[i]) 

df


# In[ ]:


X = df.reset_index(drop=True).drop(['y'],axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[ ]:


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[ ]:


lr = LogisticRegression() 
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
k_fold = KFold(n_splits=10, shuffle=False, random_state=None)


# In[ ]:


print(confusion_matrix(y_test, lr_pred))
print(round(accuracy_score(y_test, lr_pred),2)*100)
lr_cvs = (cross_val_score(lr, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, lr_pred))


# In[ ]:


X_trainK, X_testK, y_trainK, y_testK = train_test_split(df, y, test_size = 0.2, random_state = 42)

neighbors = np.arange(0,50)
cv_scores = []

for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
    scores = model_selection.cross_val_score(knn, X_trainK, y_trainK, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=28)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print(confusion_matrix(y_test, knn_pred))
print(round(accuracy_score(y_test, knn_pred),2)*100)
knn_cvs = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, knn_pred))


# In[ ]:


svc= SVC(kernel = 'sigmoid')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print(confusion_matrix(y_test, svc_pred))
print(round(accuracy_score(y_test, svc_pred),2)*100)
svc_cvs = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, svc_pred))


# In[ ]:


dtree = DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

print(confusion_matrix(y_test, dtree_pred))
print(round(accuracy_score(y_test, dtree_pred),2)*100)
dtree_cvs = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, dtree_pred))


# In[ ]:


rf = RandomForestClassifier() 
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(confusion_matrix(y_test, rf_pred))
print(round(accuracy_score(y_test, rf_pred),2)*100)
rf_cvs = (cross_val_score(rf, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, rf_pred))


# In[ ]:


gnb= GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
probs = gnb.predict(X_test)

print(confusion_matrix(y_test,gnb_pred))
print(round(accuracy_score(y_test, gnb_pred),2)*100)
gnb_cvs = (cross_val_score(gnb, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, gnb_pred))


# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print(confusion_matrix(y_test, xgb_pred ))
print(round(accuracy_score(y_test, xgb_pred),2)*100)
xgb_cvs = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())
print(classification_report(y_test, xgb_pred))


# In[ ]:


gb= GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print(confusion_matrix(y_test, gb_pred ))
print(round(accuracy_score(y_test, gb_pred),2)*100)
gb_cvs = (cross_val_score(gb, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(classification_report(y_test, gb_pred))


# In[ ]:


models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'XGBoost', 'Gradient Boosting'],
                'Score':  [rf_cvs, dtree_cvs, svc_cvs, knn_cvs, lr_cvs, gnb_cvs, xgb_cvs, gb_cvs]})

models.sort_values(by='Score', ascending=False)


# In[ ]:


#ROC AUC graphs for the top 3 models with the best precision/recall scores

fig, (ax, ax1, ax2) = plt.subplots(nrows = 1, ncols = 3, figsize = (21,9))
#Gradient Boost
probs = gb.predict_proba(X_test)
preds = probs[:,1]
fpr_gb, tpr_gb, threshold_gb = metrics.roc_curve(y_test, preds)
roc_auc_gb = metrics.auc(fpr_gb, tpr_gb)

ax.plot(fpr_gb, tpr_gb, 'b', label = 'AUC = %0.2f' % roc_auc_gb)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('Gradient Boost ',fontsize=24)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=20)
ax.legend(loc = 'lower right', prop={'size': 16})

#Random Forest
probs = lr.predict_proba(X_test)
preds = probs[:,1]
fpr_rf, tpr_rf, threshold_rf = metrics.roc_curve(y_test, preds)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)

ax1.plot(fpr_rf, tpr_rf, 'b', label = 'AUC = %0.2f' % roc_auc_rf)
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_title('Random Forest ',fontsize=24)
ax1.set_ylabel('True Positive Rate',fontsize=20)
ax1.set_xlabel('False Positive Rate',fontsize=20)
ax1.legend(loc = 'lower right', prop={'size': 16})

#XG Boost
probs = xgb.predict_proba(X_test)
preds = probs[:,1]
fpr_xgb, tpr_xgb, threshold_xgb = metrics.roc_curve(y_test, preds)
roc_auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)

ax2.plot(fpr_xgb, tpr_xgb, 'b', label = 'AUC = %0.2f' % roc_auc_xgb)
ax2.plot([0, 1], [0, 1],'r--')
ax2.set_title('XGBoost',fontsize=24)
ax2.set_ylabel('True Positive Rate',fontsize=20)
ax2.set_xlabel('False Positive Rate',fontsize=20)
ax2.legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=1)


# 

# ### Hyperparameter Optimization (Too expensive)

# rf_parameters = {'n_estimators':range(10,300,10),'criterion':('gini', 'entropy'),'max_features':('auto','sqrt','log2')}
# gs_rf = GridSearchCV(rf,rf_parameters,scoring='roc_auc',cv=10)
# gs_rf.fit(X_train,y_train)
# 
# print(gs_rf.best_score_)
# print(gs_rf.best_estimator_)

# gb_parameters = {#'nthread':[3,4], #when use hyperthread, xgboost may become slower
#                "criterion": ["friedman_mse",  "mae"],
#               "loss":["deviance","exponential"],
#               "max_features":["log2","sqrt"],
#               'learning_rate': [0.01,0.05,0.1,1,0.5], #so called `eta` value
#               'max_depth': [3,4,5],
#               'min_samples_leaf': [4,5,6],
# 
#               'subsample': [0.6,0.7,0.8],
#               'n_estimators': [5,10,15,20],#number of trees, change it to 1000 for better results
#               'scoring':'roc_auc'
#               }
# 
# gs_gb = GridSearchCV(rf,rf_parameters,scoring='roc_auc',cv=10)
# gs_gb.fit(X_train,y_train)
# 
# print(gs_gb.best_score_)
# print(gs_gb.best_estimator_)
