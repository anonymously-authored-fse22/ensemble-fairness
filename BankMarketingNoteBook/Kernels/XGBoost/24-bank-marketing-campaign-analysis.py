#!/usr/bin/env python
# coding: utf-8

# Thanks for clicking!!
# 
# I will try to show my logic of each steps of the whole process.

# #**Dataset Introduction**
# 
# The data is related with direct marketing campaigns(phone calls) of a Portuguese banking institution. Our classification goal is to predict if the client will subscribe a product.
# 
# Therefore, first we can identity the Dependent variable(Y) is a dummy variable which is if the client will subscire(Yes/No) of a term deposit. That actually will affect our chose of models. 

# In[ ]:


#import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


#import data
d1 = pd.read_csv('../input/bank-additional-full.csv',sep = ';', header = 0)

#we can first find out the types of each variables that decide 
#what preprocessing skills that we will need for each variables
d1.info()

#There are 21 variables in total and no missing values,but later 
#we can know that theres unknown category in several variables


# In[ ]:


d1.head()


# 
# After have a basic understanding of the dataset's structure, next I will start data preprocessing part.
# 
# First, we can see from the dataframe, a lot variales are categorical variables. The following data description is from UCI
# source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# Bank client data:
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
# 

# In[ ]:


d1.job.value_counts()


# In[ ]:


d1.marital.value_counts()


# In[ ]:


d1.education.value_counts()


# In[ ]:



d2 = d1
unknown = {"job": {"unknown": "admin."},
          "marital": {"unknown": "married"},
          "education": {"unknown": "university.degree"},
          "default": {"unknown": "no"},
          "housing": {"unknown": "yes"},
          "loan": {"unknown": "no"}}
d2.replace(unknown,inplace = True)
#replace unknown in each column to the most frequent in that column


# In[ ]:


d2.age.describe()


# In[ ]:


# I divide age starting rom 25% quantile and then add 20 to each categories using the 

def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 52), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 52) & (dataframe['age'] <= 72), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 72) & (dataframe['age'] <= 98), 'age'] = 4
           
    return dataframe


# In[ ]:


age(d2).head()


# In[ ]:


#encode varables that already become dummy 
labelencoder_X = LabelEncoder()
d2.job = labelencoder_X.fit_transform(d2.job)
d2.marital = labelencoder_X.fit_transform(d2.marital)
d2.default = labelencoder_X.fit_transform(d2.default)
d2.housing = labelencoder_X.fit_transform(d2.housing)
d2.loan = labelencoder_X.fit_transform(d2.loan)


# In[ ]:


edu = {"illiterate" : 0,
       "basic.4y" : 1,
       "basic.6y" : 2,
       "basic.9y" : 3,
       "high.school" : 4,
       "professional.course" : 5,
       "university.degree" : 6}
d2['education'].replace(edu,inplace = True)

#Because I think education level has kind of ordinal, so i assign numbers to different level


# In[ ]:


d2.head()


# Related with the last contact of the current campaign:
# 
# 8 - contact: contact communication type (categorical: 'cellular','telephone') 
# 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

# In[ ]:


d1.contact.value_counts()


# In[ ]:


d1.month.value_counts()


# In[ ]:


d1.day_of_week.value_counts()


# In[ ]:


d2.contact = labelencoder_X.fit_transform(d2.contact)
d2.month = labelencoder_X.fit_transform(d2.month)
d2.day_of_week = labelencoder_X.fit_transform(d2.day_of_week)


# In[ ]:


d2.head()


# Other attributes:
# 
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
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

# In[ ]:


d1.poutcome.value_counts()


# In[ ]:


d2['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)


# In[ ]:


#transform Y variable to dummy
d2.y.value_counts()
d2.y= labelencoder_X.fit_transform(d2.y)


# In[ ]:



d2.describe()


# In[ ]:


corr = d2.corr()
corr.style.background_gradient(cmap = 'coolwarm')


# In[ ]:


sns.boxplot(x = 'duration', data = d2, orient = 'v')
#Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and 
#should be discarded if the intention is to have a realistic predictive model.
#Therefore, we drop this variable


# #**Model**
# 
# * Logistic Regression
# * Random Forest Classifier
# * XGB Classifier

# In[ ]:


Y = d2.y
X = d2.drop('y',axis = 1)
X = X.drop('duration',axis = 1)


# In[ ]:


#train for 75%, test dataset is 25%
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)


# In[ ]:



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


#############
#Logistic Regression

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
logpred = logmodel.predict(X_test)


# In[ ]:


import sklearn.metrics as metrics
conlg = print(metrics.confusion_matrix(Y_test, logpred))
acclg = print(round(metrics.accuracy_score(Y_test, logpred), 4)*100)
##89.7


# In[ ]:


#since the dataset is relatviely imbalaced, we should look at ROC_AUC
problg = logmodel.predict_proba(X_test)
predslg = problg[:,1]
fprlg, tprlg, threshold = metrics.roc_curve(Y_test, predslg)
roc_auclg = metrics.auc(fprlg, tprlg)


# In[ ]:


plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fprlg, tprlg, 'b', label = 'AUClg = %0.2f' % roc_auclg)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate lg')
plt.xlabel('False Positive Rate lg')
plt.show()
##0.79


# In[ ]:



#####
#Random Forest 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, Y_train)
rfpred = rf.predict(X_test)


# In[ ]:


conrf = print(metrics.confusion_matrix(Y_test, rfpred))
accrf = print(round(metrics.accuracy_score(Y_test, rfpred), 4)*100)


# In[ ]:


probrf = rf.predict_proba(X_test)
predsrf = probrf[:,1]
fprrf, tprrf, thresholdrf = metrics.roc_curve(Y_test, predsrf)
roc_aucrf = metrics.auc(fprrf, tprrf)


# In[ ]:


plt.title('Receiver Operating Characteristic for Random Forest')
plt.plot(fprrf, tprrf, 'b', label = 'AUCrf = %0.2f' % roc_aucrf)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate rf')
plt.xlabel('False Positive Rate rf')
plt.show()
##0.77


# In[ ]:


#########
##XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)
xgbpred = xgb.predict(X_test)

conxgb = print(metrics.confusion_matrix(Y_test, xgbpred))
accxgb = print(round(metrics.accuracy_score(Y_test, xgbpred), 4)*100)
##89.97


# In[ ]:


probxgb = xgb.predict_proba(X_test)
predsxgb = probxgb[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(Y_test, predsxgb)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)


plt.title('Receiver Operating Characteristic for XGBoost')
plt.plot(fprxgb, tprxgb, 'b', label = 'AUCxgb = %0.2f' % roc_aucxgb)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate xgb')
plt.xlabel('False Positive Rate xgb')
plt.show()
##0.8

