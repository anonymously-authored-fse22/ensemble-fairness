#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing

# In[ ]:


import pandas as pd 
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/bank-marketing/bank-additional-full.csv',sep = ';')
df.head(5)


#  ## Input variables:
#  
#    #### Bank client data:
#    * 1 - age (numeric)
#    * 2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
#    * 3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
#    * 4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
#    * 5 - default: has credit in default? (categorical: "no","yes","unknown")
#    * 6 - housing: has housing loan? (categorical: "no","yes","unknown")
#    * 7 - loan: has personal loan? (categorical: "no","yes","unknown")
#    #### Related with the last contact of the current campaign:
#    * 8 - contact: contact communication type (categorical: "cellular","telephone") 
#    * 9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#    * 10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
#    * 11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#    #### Other attributes:
#    * 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#    * 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#    * 14 - previous: number of contacts performed before this campaign and for this client (numeric)
#    * 15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
#    #### Social and economic context attributes
#    * 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#    * 17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
#    * 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
#    * 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
#    * 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
#    #### Output variable (desired target):
#    * 21 - y - has the client subscribed a term deposit? (binary: "yes","no")

# ## Dataset Analysis

# In[ ]:


# Information about attribute types

df.info()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


# Statistical description of numeric dataset attributes
df.describe()


# ## Duplicated Values

# In[ ]:


# Removing Duplicate Values

print(df.duplicated().sum()) # 12 duplicate values 
df = df.drop_duplicates() # Values have been removed


# ## Hidden Missing Values

# In[ ]:


# Checking for null values
df.isnull().values.any()


# > As stated in the description of the variables above, there are several missing values in some $categorical$ attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques.

# In[ ]:


print("# Missing job: {0}".format(len(df.loc[df['job'] == "unknown"])))
print("# Missing marital: {0}".format(len(df.loc[df['marital'] == "unknown"])))
print("# Missing education: {0}".format(len(df.loc[df['education'] == "unknown"])))
print("# Missing default: {0}".format(len(df.loc[df['default'] == "unknown"])))
print("# Missing housing:: {0}".format(len(df.loc[df['housing'] == "unknown"])))
print("# Missing loan: {0}".format(len(df.loc[df['loan'] == "unknown"])))
print("# Missing contact: {0}".format(len(df.loc[df['contact'] == "unknown"])))
print("# Missing month: {0}".format(len(df.loc[df['month'] == "unknown"])))
print("# Missing day_of_week: {0}".format(len(df.loc[df['day_of_week'] == "unknown"])))
print("# Missing poutcome: {0}".format(len(df.loc[df['poutcome'] == "unknown"])))


# ## Outliers Analysis

# In[ ]:


# Outliers are mainly found in duration. Changes will need to be made.

plt.figure(figsize=(14,6))
df.boxplot()
print()


# ### Removing outliers in 'duration' using IQR method.

# In[ ]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=df['duration'])
plt.show()


# In[ ]:


Q1 = df['duration'].quantile(.25)
Q3 = df['duration'].quantile(.75)

Q1,Q3


# In[ ]:


IQR = Q3 - Q1
IQR


# In[ ]:


lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

lower,upper


# #### Defining lower / upper

# In[ ]:


df = df[df['duration'] >= lower] 


# In[ ]:


df = df[df['duration'] <= upper]


# In[ ]:


# There were 41188 lines. Now, we have 38213 lines.
df.shape


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(14,6))
df.boxplot()
print()


# ## Correlations

# In[ ]:


# Term Deposit Subscription (Target). Tranforming Yes = 1 / No = 0
df['y'] = 1 * (df['y']== 'yes')


# In[ ]:


# Correlation between columns

plt.figure(figsize=(12,7))
correlacao = df.corr()
sns.heatmap(correlacao, annot = True);


# In[ ]:


# Analyzing the correlations between numeric columns with the target variable (y)

df.corr()['y'].drop('y').sort_values()


# ## Data Balancing

# In[ ]:


#Target distribution

sns.countplot(df['y']);


# In[ ]:


df_classe_majority = df[df.y==0]
df_classe_minority = df[df.y==1]


# In[ ]:


df_classe_majority.shape


# In[ ]:


df_classe_minority.shape


# In[ ]:


# Upsample of minority class
from sklearn.utils import resample
df_classe_minority_upsampled = resample(df_classe_minority, 
                                           replace = True,     
                                           n_samples = 35100,   
                                           random_state = 150) 


# In[ ]:


df_balanced_data = pd.concat([df_classe_majority, df_classe_minority_upsampled])


# In[ ]:


df_balanced_data.y.value_counts()


# In[ ]:


sns.countplot(df_balanced_data['y'])


# > Balanced data. Saving the dataset with the manipulated data.

# In[ ]:


df_balanced_data.to_csv('df_modified.csv', encoding = 'utf-8', index = False) #df2


# ## Business Questions

# In[ ]:


df2 = pd.read_csv('df_modified.csv')
df2.head()


# ### 1 - What is the average duration (in seconds) of the call for those who did not make a term deposit (0) ? And for those who made term deposits (1)?
# > For those who made term deposits (1), the average time was 331.72 seconds. For those who did not make a term deposit (0), the average time was 191.86 seconds. It means that, for a customer to make a term deposit, more time is needed to convince him/her.

# In[ ]:


time = df2.groupby('y').duration.mean()
time.plot.bar()
plt.title('Duration', fontsize = 15)
plt.xlabel('y', fontsize = 15)
plt.ylabel('Duration (seconds)')
plt.show()

print(time)


# ### 2 - In which month do customers usually make the most deposits?
# > May is the month when most customers make a term deposit

# In[ ]:


plt.title('Months/Deposits', fontsize = 15)
sns.countplot(df2['month'])
plt.show()


# ### 3 - Among those who made bank deposits, what was the main form of contact?
# > The main form of contact is the cellular. Few customers who made term bank deposits were contacted by telephone.

# In[ ]:


df2.groupby('y').contact.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()


# ### 4 - What type of job is most common among those who made bank deposits?
# > Admin.

# In[ ]:


jobs = df2[df2['y'] == 1].groupby('y').job.value_counts()
jobs.plot.bar()

plt.title('Jobs/Deposit', fontsize = 15)
plt.xlabel('Job', fontsize = 15)
plt.ylabel('Total')

plt.show()


# ### 5 - What is the Age Distribution of Customers?
# > Between 25 and 42 years old

# In[ ]:


sns.distplot(df2['age'], color = 'magenta')
plt.title('Customer Age Distribution', fontsize = 15)
plt.xlabel('Age', fontsize = 15)
plt.ylabel('Total')
plt.show()


# ##  Spliting

# > The variables (age job marital education default housing loan) will be discarded from the learning models as they have little relevance to the target variable

# In[ ]:


# Turning All Categorical Attributes to Numeric

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_var =['contact','month','day_of_week','poutcome']
for i in cat_var:
    df2[i]= le.fit_transform(df2[i]) 

df2.head()


# ### Checking again for missing values
# > We didn't generate missing values by accident, fortunately

# In[ ]:


df2.isnull().values.any()


# In[ ]:


df2.isnull().sum()


# ### Train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split

X = df2[['contact','month','day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']] # Only numeric values

y = df2['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[ ]:


X.shape, y.shape


# In[ ]:


# Printing the results

print("{0:0.2f}% training data".format((len(X_train)/len(df2.index)) * 100))
print("{0:0.2f}% test data".format((len(X_test)/len(df2.index)) * 100))


# ## Developing and training the model

# In[ ]:


# Model evaluation metrics

from sklearn import metrics


# ### Baseline - Basic cutoff point for predicting values

# In[ ]:


from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="stratified") #stratified: generates predictions by respecting the training sets class distribution.
baseline.fit(X_train, y_train.ravel())


# In[ ]:


accuracy_bl = baseline.score(X_train, y_train.ravel())
print("Accuracy: {0:.4f}".format(accuracy_bl))
print()


# ###  Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train.ravel())

nb_predict_test = nb.predict(X_test)

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test, labels = [1, 0])))
print("")

accuracy_nb = metrics.accuracy_score(y_test, nb_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_nb))
print()

print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test, labels = [1, 0]))


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 0.7, random_state = 42, max_iter = 1000)
lr.fit(X_train, y_train.ravel())

lr_predict_test = lr.predict(X_test)

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test, labels = [1, 0])))
print("")

accuracy_lr = metrics.accuracy_score(y_test, lr_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_lr))
print()

print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test, labels = [1, 0]))


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 42)
rfc.fit(X_train, y_train.ravel())

rfc_predict_test = rfc.predict(X_test)

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, rfc_predict_test, labels = [1, 0])))
print("")

accuracy_rfc = metrics.accuracy_score(y_test, rfc_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_rfc))
print()

print("Classification Report")
print(metrics.classification_report(y_test, rfc_predict_test, labels = [1, 0]))


# ### Decision Tree Classifier

# In[ ]:


from sklearn import tree

dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train, y_train.ravel())

dtc_predict_test = dtc.predict(X_test)

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, dtc_predict_test, labels = [1, 0])))
print("")

accuracy_dtc = metrics.accuracy_score(y_test, dtc_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_dtc))
print()

print("Classification Report")
print(metrics.classification_report(y_test, dtc_predict_test, labels = [1, 0]))


# ### Support Vector Machine Classifier

# In[ ]:


from sklearn import svm

svmc = svm.SVC()
svmc.fit(X_train, y_train.ravel())

svmc_predict_test = svmc.predict(X_test)

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, svmc_predict_test, labels = [1, 0])))
print("")

accuracy_svmc = metrics.accuracy_score(y_test, svmc_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_svmc))
print()

print("Classification Report")
print(metrics.classification_report(y_test, svmc_predict_test, labels = [1, 0]))


# ### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gb= GradientBoostingClassifier()
gb.fit(X_train, y_train.ravel())

gb_predict_test = gb.predict(X_test)

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, gb_predict_test, labels = [1, 0])))
print("")

accuracy_gb = metrics.accuracy_score(y_test, gb_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_gb))
print()

print("Classification Report")
print(metrics.classification_report(y_test, gb_predict_test, labels = [1, 0]))


# ## Comparing and evaluating models

# In[ ]:


# Table summary for better viewing

results = pd.DataFrame([
    {'Algorithm' : 'Baseline', 'Accuracy' : accuracy_bl*100},
    {'Algorithm' : 'Naive Bayes', 'Accuracy' : accuracy_nb*100},
    {'Algorithm' : 'Logistic Regression', 'Accuracy' : accuracy_lr*100},
    {'Algorithm' : 'Random Forest', 'Accuracy' : accuracy_rfc*100},
    {'Algorithm' : 'Decision Tree', 'Accuracy' : accuracy_dtc*100},
    {'Algorithm' : 'Support Vector Machine', 'Accuracy' : accuracy_svmc*100},
    {'Algorithm' : 'Gradient Boosting', 'Accuracy' : accuracy_gb*100}
])

results.sort_values(by=['Accuracy'], ascending=False)


# > Decision Tree was the model that achieved the best accuracy, with 97.08%. It will be the model used to predict whether or not the customer will be able to sign a term deposit. The model will be saved and ready to make predictions.

# ## Making Predictions with the Trained Model

# In[ ]:


import pickle

# Saving the model

filename = 'dtc.sav'
pickle.dump(dtc, open(filename, 'wb'))


# In[ ]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_test)
result[:50]


# In[ ]:


real_full=df2['y']
real=real_full[:1000]

pred = rfc.predict(X_test)

df3=pd.DataFrame({'real': real, 'prediction':pred[:1000]})


# In[ ]:


# Real x Prediction for the first 1000 lines

df3.head()


# In[ ]:


# How many of the predictions are the same or different from the real ones in the first 1000 lines

print(df3[df3['real'] == df3['prediction']].value_counts())
print(df3[df3['real'] != df3['prediction']].value_counts())


# ## Classification Test

# In[ ]:


df2.columns


# In[ ]:


# Putting all the necessary variables for the classification test, except the target variable (y).

test = np.array([[0,7,6,330,1,7,0,1,1.1,93.200,-22.6,4.961,5008.7]])


# In[ ]:


dtc.predict(test)


# ## Conclusion

# * Decision Tree was the model that achieved the best accuracy, with 97.08%.;
# 
# 
# * From the model, it will be possible to predict whether or not the customer will subscribe a term deposit, placing all the necessary variables for the classification test;
# 
# 
# * The 'duration' outliers were removed. In this case, there would need to be an agreement with the bank's business area, which would (or not) guide the removal of outliers, in case they were unnecessary;
# 
# 
# * The same case would apply to the 'unknown' values, found in several columns of the dataset. Who should decide the exclusion or imputation of these data would be the business area;
# 
# 
# * Other models can and should be tested;

# In[ ]:




