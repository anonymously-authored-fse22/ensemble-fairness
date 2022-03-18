#!/usr/bin/env python
# coding: utf-8

#  # MACHINE_LEARNING_PROJECT_2_BANK_MARKETING_DATA

# <h3> Dataset Description </h3>
# The data is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. 
#  
# * __bank client attributes__:
#     * age: age of client (numeric)   
#     * job : type of job (categorical: "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services")   
#     * marital : marital status (categorical: "married", "divorced", "single")  
#     * education: client highest education (categorical: "unknown", "secondary", "primary", "tertiary")
#     * default: has credit in default? (binary/2-categories: "yes", "no")
#     * balance: average yearly balance, in euros (numeric)  
#     * housing: has housing loan? (binary/2-categories: "yes", "no")  
#     * loan: has personal loan? (binary/2-categories: "yes", "no")  
# * __related with the last contact of the current campaign__:
#     * contact: contact communication type (categorical: "unknown", "telephone", "cellular") 
#     * day: last contact day of the month (numeric)
#     * month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#     * duration: last contact duration, in seconds (numeric)
# * __other attributes__:
#     * campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#     * pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#     * previous: number of contacts performed before this campaign and for this client (numeric)
#     * poutcome: outcome of the previous marketing campaign ( categorical: 'unknown","other", "failure", "success")
# * __Output variable (desired target)__:
#     * y: has the client subscribed a term deposit? (binary: "yes", "no")

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/bank-pred/bank.csv')
df


# In[ ]:


df.info()


# ### Describe the pdays column, make note of the mean, median and minimum values. Anything fishy in the values? 

# In[ ]:


df.pdays.describe()


# In[ ]:


df.pdays.median()


# - As we can observe above,there are values as -1 which doesnot conclude anything,because pdays basically means "is the number of days that passed by after the client was last contacted from a previous campaign",so in this -1 is bit out of fit.
# - here mean = -1,minimum value=-1 and median =-1,which does not infer anything

# In[ ]:


df1 = df[df['pdays']>0]
df1


# In[ ]:


df1.pdays.describe()


# In[ ]:


df1.pdays.median()


# - After selecting relevant values we can observe ,mean=224,minimum value =1 and median= 194

# In[ ]:





# ### Plot a horizontal bar graph with the median values of balance for each education level value. Which group has the highest median? 

# In[ ]:


df.groupby('education').median()['balance']


# In[ ]:


sns.barplot(x=df['balance'],y=df['education'],estimator=np.median)


# - As we can observe.Tertiary education has higher median value under balances and next we have few values which are 
# - unknown,which second highest median under balance

# ### Make a box plot for pdays. Do you see any outliers?

# In[ ]:


df1.pdays.value_counts()


# In[ ]:


print(sns.boxplot(y=df1['pdays']))


# - Above plot is by using pday with know values and ignoring missing values(-1)

# In[ ]:


print(sns.boxplot(y=df['pdays']))


# In[ ]:


df.pdays.value_counts()


# - this plot is including missing values(-1),and we have see outliers in both kind of data,as we see above there are 36954 values missing (-1) for pdays in the data

# ### FINAL OBJECTIVE= To make a predictive model to predict if the customer will respond positively to the campaign or not. The target variable is “response”. 

#  - Performing bi-variate analysis to identify the features that are directly associated with the target variable before predicting 
#  
# - Converting the response variable to a dummy variable

# In[ ]:


df=pd.get_dummies(df,columns=['response'],drop_first=True)
df


# ### Plotting numerical features against categorical features’ 

# In[ ]:


sns.scatterplot(x='poutcome',y='pdays',data=df)


# In[ ]:





# In[ ]:


df.head(2)


# In[ ]:


sns.pairplot(df[['age','salary','balance','previous','duration','campaign','response_yes']])


# __Observation:__  
# * Pair plots of age-campaign and day-campaign are much efficient in distinguishing between different classes with very few overlapes.

# In[ ]:


sns.distplot(df['age'],bins=50)


# In[ ]:





# In[ ]:


lst = [df]
for column in lst:
    column.loc[column["age"] < 30,  'age group'] = 30
    column.loc[(column["age"] >= 30) & (column["age"] <= 44), 'age group'] = 40
    column.loc[(column["age"] >= 45) & (column["age"] <= 59), 'age group'] = 50
    column.loc[column["age"] >= 60, 'age group'] = 60


# In[ ]:


agewise_response = pd.crosstab(df['response_yes'],df['age group']).apply(lambda x: x/x.sum() * 100)
agewise_response = agewise_response.transpose()


# In[ ]:


sns.countplot(x='age group', data=df, hue='response_yes')


# In[ ]:


print('Success rate and total people with different age groups contacted:')
print('People with age < 30 contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 30]), df[df['age group'] == 30].response_yes.value_counts()[1]/len(df[df['age group'] == 30])))
print('People between 30 & 45 contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 40]), df[df['age group'] == 40].response_yes.value_counts()[1]/len(df[df['age group'] == 40])))
print('People between 40 & 60 contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 50]), df[df['age group'] == 50].response_yes.value_counts()[1]/len(df[df['age group'] == 50])))
print('People with 60+ age contacted: {}, Success rate: {}'.format(len(df[df['age group'] == 60]), df[df['age group'] == 60].response_yes.value_counts()[1]/len(df[df['age group'] == 60])))


# __Observation:__  
# * People with age < 30 or 60+ have higher success rate.  
# * Only 3% of people with age of 60+ 

# ### With respect to JOB field

# In[ ]:


sns.set(rc={'figure.figsize':(20,5)})
sns.countplot(x=df['job'], data=df, hue=df['response_yes'])
plt.title('Response recieved with respect to JOB')


# - Now lets create a table to count number of people in different job role and response of succes rate

# In[ ]:


from prettytable import PrettyTable


# In[ ]:


counts = PrettyTable(['Job', 'Total Clients', 'Success rate'])
counts.add_row(['Blue-collar', len(df[df['job'] == 'blue-collar']), df[df['job'] == 'blue-collar'].response_yes.value_counts()[1]/len(df[df['job'] == 'blue-collar'])])
counts.add_row(['Management', len(df[df['job'] == 'management']), df[df['job'] == 'management'].response_yes.value_counts()[1]/len(df[df['job'] == 'management'])])
counts.add_row(['Technician', len(df[df['job'] == 'technician']), df[df['job'] == 'technician'].response_yes.value_counts()[1]/len(df[df['job'] == 'technician'])])
counts.add_row(['Admin', len(df[df['job'] == 'admin.']), df[df['job'] == 'admin.'].response_yes.value_counts()[1]/len(df[df['job'] == 'admin.'])])
counts.add_row(['Services', len(df[df['job'] == 'services']), df[df['job'] == 'services'].response_yes.value_counts()[1]/len(df[df['job'] == 'services'])])
counts.add_row(['Retired', len(df[df['job'] == 'retired']), df[df['job'] == 'retired'].response_yes.value_counts()[1]/len(df[df['job'] == 'retired'])])
counts.add_row(['Self-employed', len(df[df['job'] == 'self-employed']), df[df['job'] == 'self-employed'].response_yes.value_counts()[1]/len(df[df['job'] == 'self-employed'])])
counts.add_row(['Entrepreneur', len(df[df['job'] == 'entrepreneur']), df[df['job'] == 'entrepreneur'].response_yes.value_counts()[1]/len(df[df['job'] == 'entrepreneur'])])
counts.add_row(['Unemployed', len(df[df['job'] == 'unemployed']), df[df['job'] == 'unemployed'].response_yes.value_counts()[1]/len(df[df['job'] == 'unemployed'])])
counts.add_row(['Housemaid', len(df[df['job'] == 'housemaid']), df[df['job'] == 'housemaid'].response_yes.value_counts()[1]/len(df[df['job'] == 'housemaid'])])
counts.add_row(['Student', len(df[df['job'] == 'student']), df[df['job'] == 'student'].response_yes.value_counts()[1]/len(df[df['job'] == 'student'])])
counts.add_row(['Unknown', len(df[df['job'] == 'unknown']), df[df['job'] == 'unknown'].response_yes.value_counts()[1]/len(df[df['job'] == 'unknown'])])
print(counts)


# __Observation:__  
# * Top contacted people are from job type: 'blue-collar', 'management' & 'technician'
# * Success rate is highest for student

# ### Poutcome against Response

# In[ ]:


sns.countplot(x=df['poutcome'], data=df, hue=df['response_yes'])
plt.title('Count Plot of poutcome for target variable')


# __Observation:__
# * Most of the clients contacted have previous outcome as 'unknown'.

# ### Salary

# In[ ]:


sns.countplot(x=df['salary'], data=df, hue=df['response_yes'])
plt.title('Salary wise and response')


# __Observation:__
# * Contacted people were with salary rane betwen 20000 to 100000
# * Successn was the people having salary of 100000

# ###  Education Analysis

# In[ ]:


sns.countplot(x=df['education'], data=df, hue=df['response_yes'])
plt.title('Respinse received based on education')


# In[ ]:





# In[ ]:


df.education.value_counts()


# __Observation:__
# * Most of the people who are contacted have tertiary or secondary education.

# In[ ]:


sns.countplot(x=df['default'], data=df, hue=df['response_yes'])
plt.title('Response received against defaulters')


# In[ ]:


df.default.value_counts()


# In[ ]:


df[df['default']=='yes'].response_yes.count()


# #### As we can observe above,Very few people are contacted who are defaulter,

# ### Loan

# In[ ]:


sns.countplot(x=df['loan'], data=df, hue=df['response_yes'])
plt.title('Count plot of loan for target variable y')


# __Observation:__  
# * As we observe aboe, less people are contacted who have loan.

# ### Contacts-Mode of communications

# In[ ]:


sns.countplot(x=df['contact'], data=df, hue=df['response_yes'])
plt.title('Modes of Communication:')


# In[ ]:


df.contact.value_counts()


# __Observation:__  
# * As we observe above, most of the communication was made through 'cellular' network

# ### Month 

# In[ ]:


sns.countplot(x=df['month'], data=df, hue=df['response_yes'])
plt.title('MOnth wise communication and response')


# __Observation:__
# * Most of the people are contacted in the month of May
# * Decemeber is the month less people are contacted

# In[ ]:


df.head(1)


# In[ ]:


categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical = [x for x in df.columns.to_list() if x not in categorical]
numerical.remove('response_yes')
numerical.remove('age group')


# In[ ]:





# In[ ]:


corr_data = df[numerical + ['response_yes']]
corr = corr_data.corr()
plt.close()
cor_plot = sns.heatmap(corr,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':10})
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.xticks(fontsize=10,rotation=-30)
plt.yticks(fontsize=10)
plt.title('Correlation Matrix')
plt.show()


# ### Are the features about the previous campaign data useful? 

# - As we can observe in above correlatiion matrix,previous cmapaign data is not much correlated,and have only 0.093 almost close 0 .so previous data cannot be used to prediict much

# __Other important Observations:__  
# * pdays and previous have higher correlation
# * duration have a higher correlation with our target variable
# * Numerical features have very less correlation between them.

# ### Are pdays and poutcome associated with the target? 

# In[ ]:


pd.crosstab(df['pdays'],df['poutcome'])


# In[ ]:


pd.crosstab(df['pdays'],df['poutcome'],values=df['response_yes'],aggfunc='count',margins=True,normalize=True)


# In[ ]:


pd.crosstab(df['pdays'],df['previous'],values=df['response_yes'],aggfunc='count',margins=True,normalize=True)


# __Observations:__  
# * pdays and poutcome are associated with each other
# * From previous marketing campaign ,we can see that there was a outcome ,when there are more contacts made to a particular client
# * pdays=-1 has overall count of 36954 same as previous=0 ,,which implies that many of people were not contacted.
# 
# 
# ### Inference:-- So to proceed further, we can just retain pdays value,as we can say,in previous campaign the clients were not contact for so many people ,so the previous value is 0 for all those 36954 contacts

# ### - Necessary transformations of categorical variables and the numeric variables
# ### -Handling variables corresponding to the previous campaign

# - __(We will be converting previous marketing campaigns variables into dummy variables)__

# In[ ]:


df=pd.get_dummies(df,drop_first=True)
df


# In[ ]:


df.columns


# ## Train test split

# In[ ]:


X=df.drop('response_yes', axis=1)
Y=df['response_yes']


# ## Predictive model 1: Logistic regression 

# __1.Before we build a model,lets select top features using RFE(recursive feature elimination),so that model predicts better__

# - __ FInding RFE to select top n features in an automated fashion (choose n as you see fit__

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# In[ ]:


logm=LogisticRegression()


# In[ ]:


from sklearn.feature_selection import RFE
import statsmodels.api as sm 


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# 30% of the data will be used for testing
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=0)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


logm.fit(X_train,Y_train)


# In[ ]:


rfe = RFE(logm, 10)
rfe = rfe.fit(X_train, Y_train)
rfe_ = X_train.columns[rfe.support_]
rfe_


# __We will check VIF(Variance Inflation factor to delete the redundant factor___

# In[ ]:


def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# In[ ]:


X_new_train=X_train[rfe_]


# In[ ]:


checkVIF(X_new_train)


# ** As we can observe above,Month of dec has highest VIF value, we will rmeove that feature

# -- So we will consdier the rest of the features for building our model

# In[ ]:


X_new=df[['housing_yes', 'contact_unknown','month_aug','month_jan',
       'month_jul', 'month_mar', 'month_oct', 'month_sep', 'poutcome_success']]
Y=df['response_yes']


# In[ ]:


X_new_train, X_new_test, Y_train, Y_test= train_test_split(X_new, Y, test_size=0.3, random_state=0)


# In[ ]:


z=logm.fit(X_new_train,Y_train)
z


# ### Estimate the model performance using k fold cross validation

# In[ ]:


auc=[X_train,X_new_train]
models = []
models.append(('LogisticRegression', LogisticRegression()))
for i in auc:
        kfold = KFold(n_splits=10, random_state=0)    
        # train the model
        cv_results = cross_val_score(LogisticRegression(), i, Y_train, cv=kfold, scoring='accuracy')    
        msg = "%s: %f (%f)" % (LogisticRegression, cv_results.mean(), cv_results.std())
        print(msg)


# ### By using the features we got from VIF and Kfold ,we have got an accuracy of 89%.

# In[ ]:


Y_pred=z.predict(X_new_test)
Y_pred


# In[ ]:


Y_pred.shape


# In[ ]:


# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print(confusion_matrix(Y_test,Y_pred))


# ### Precision, recall, accuracy of your model

# In[ ]:


print(classification_report(Y_test,Y_pred))


# ### Which features are the most important from your model? 

# __The most important features are as follows__
#  * Month,housing,poutcome,contact

# ## Predictive model 2: Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rfc = RandomForestClassifier(n_estimators=30,max_depth=30)
rfc.fit(X_train, Y_train)


# In[ ]:


y_pred=rfc.predict(X_test)
y_pred


# In[ ]:


print(confusion_matrix(Y_test,y_pred))


# In[ ]:


print(classification_report(Y_test, y_pred))


# ### Estimating the model performance using k fold cross validation

# In[ ]:


p=[X_train,X_new_train]


# In[ ]:


for i in p:
    kf = KFold(n_splits=10)    
    cross_v = cross_val_score(RandomForestClassifier(), i, Y_train, cv=kfold, scoring='accuracy')  
    print('Cross validation score:',cross_v.mean())


# __As we can observe above__
# - CV score(for all features): 0.9043196330140578
# - CV score(for selected features):  0.8933231392793823

# In[ ]:


model_new = RandomForestClassifier(n_estimators=45,max_depth=10)
model_new.fit(X_new_train, Y_train)


# In[ ]:


y1_pred=model_new.predict(X_new_test)
y1_pred


# In[ ]:


print('For all features')
print(accuracy_score(Y_test, y1_pred))
print('For selected features')
print(accuracy_score(Y_test, y1_pred))


# In[ ]:


print(classification_report(Y_test, y1_pred))


# In[ ]:


print(confusion_matrix(Y_test,y1_pred))


# ### Which metric did you choose and why? 

# We used classification performance metrics such as, Precision score,accuracy score , recall score and Cross val score etc.
# In order to estimate the error, the model is required to test a dataset which it hasn’t seen yet.
# 
# Therefore for the purpose of testing the model, we would require a labelled dataset. This can be achieved by splitting the training dataset into training dataset and testing dataset. This can be achieved by various techniques such as, k-fold cross validation,

# ### Which model has better performance on the test set? 

# Logistic has got better accuracy score compared to random forset, hence we can say that it has better performance,is an important model as it results in high AUC score.

# 

# In[ ]:




