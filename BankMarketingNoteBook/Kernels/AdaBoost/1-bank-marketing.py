#!/usr/bin/env python
# coding: utf-8

# <h3>Given:</h3>
# The data is related with direct marketing campaigns of a Portuguese banking institution.
# The marketing campaigns were based on phone calls. Often, more than one contact to
# the same client was required, in order to access if the product (bank term deposit) would
# be ('yes') or not ('no') subscribed.
# <h3>Objective:</h3>
# The classification goal is to predict the likelihood of a customer subscribing term deposit
# loans.

# <h2>1. Reading the data</h2>

# In[3]:


# Importing the libraries
import pandas as pd        # for data manipulation
import seaborn as sns      # for statistical data visualisation
import numpy as np         # for linear algebra
import matplotlib.pyplot as plt      # for data visualization
from scipy import stats        # for calculating statistics

# Importing various machine learning algorithm from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,roc_curve,auc,accuracy_score
from  sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier


# In[ ]:


dataframe= pd.read_csv("bank-full.csv")  # Reading the data
dataframe.head()   # showing first 5 datas


# In[ ]:


dataframe.shape


# The data given has  17 columns and consist of 45211 data. And all the data is read correctly.

# In[ ]:


dataframe.info()


# <body>
#     The above information shows the following:<br>
#     a. There are no null or missing values present <br>
#     b. The attributes are either int or object <br>
#     c. Further there maybe necessary to label encode this objects<br>
#     d. Many attribute has "unknown" present in the data that may mean different for each of the attribute accordingly
#        <br><br>
# 

# In[ ]:


dataframe.isnull()


# There are no null values in any row or column

# In[ ]:


dataframe.apply(lambda x: len(x.unique()))


# From  the data:<br>
# 
# 8 variable have interval data:
# <ul><li>Age: Age of the customer</li>
# <li>Balance: average yearly balance, in euros</li>
# <li>Day: last contact day of the month</li>
# <li>Duration: last contact duration, in seconds (numeric). </li>
# <li>Month: last contact month of the year</li>
# <li>Campaign: number of contacts performed during this campaign and for this
# client</li>
# <li>Pdays: number of days that passed by after the client was last contacted from a
# previous campaign</li>
# <li>Previous: number of contacts performed before this campaign and for this client</li>
# 
# </ul>
# 
# 4 variables have categorical data:
# <ul><li>Housing: has housing loan?</li>
# <li>Loan: has personal loan?</li>
# <li>Target: customer have a certificate of deposit or not </li>
# <li>Defualt: has credit in default? </li>
# 
# </ul>
# 5 variables contains Ordinal categorical data:
# <ul><li>Marital: marital status</li>
# <li>Job: Type of job</li>
# <li>Education: Education level of the customer</li>
# <li>Contact: contact communication type</li>
# <li>Poutcome: outcome of the previous marketing campaign</li>
# </ul>
# 

# In[ ]:


dataframe.describe()


# The balance and pdays column data contain negative values as experience. This can be seen in the value of min of Experience.The data in balance my go negative as the customer may have loan but in pdays column days cannot go negative, therefore may need cleansing.<br><br>
# Mean Age is aproximately 41 years old. (Minimum: 18 years old and Maximum: 95 years old.)
# 
# The mean balance is 1,362. However, the Standard Deviation (std) is a high number so we can understand through this that the balance is heavily distributed across the dataset.
# 
# As the data information said it will be better to drop the duration column since duration is highly correlated in whether a potential client will buy a term deposit. 
# 
# 

# In[ ]:


sns.pairplot(dataframe)


# In[ ]:


dataframe.Target.value_counts()


# Target will be our target column

# In[ ]:


f = plt.subplots(1, figsize=(12,4))

colors = ["#FA5858", "#64FE2E"]
labels ="Did not Open Term Deposit", "Opened Term Deposit"

plt.suptitle('Information on Term Deposits', fontsize=20)

plt.pie(dataframe.Target.value_counts(),explode=[0,0.25], shadow=True,colors=colors,labels=labels, startangle=25,autopct='%1.1f%%')

plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(5,1,1)
sns.boxplot(dataframe.age)
plt.subplot(5,1,2)
sns.boxplot(dataframe.balance)
plt.subplot(5,1,3)
sns.boxplot(dataframe.campaign)
plt.subplot(5,1,4)
sns.boxplot(dataframe.duration)


# In[ ]:


dataframe.skew()


# <h3>Obv</h3>
# The above boxplot and the above information shows:
# Day feature is normally distributed, as the mean is nearly equal to median<br>
# <br>
# And all other attributed are hignly positively skewed as we can see the mean is greater than the medianand  has lot of outliers.<br>
# <br>
# There are some negative values contained in experience that actually dont make any sense. Its better to clean them by applying the median of experience of the group having same age and education but positive experience.

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
sns.distplot(dataframe.age)
plt.subplot(3,1,2)
sns.countplot(dataframe.job)
plt.subplot(3,1,3)
sns.countplot(dataframe.marital)
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = dataframe, ax = ax1)
sns.countplot(x = 'housing', data = dataframe, ax = ax2)
sns.countplot(x = 'loan', data = dataframe, ax = ax3)


# <h3>Obv</h3>
# <li>The ages dont mean to much, has a medium dispersion</li>
# 
# <li>Looks like Jobs, Marital and Education will have more effect on whether the customer will subscribe term deposit of not</li><li>Management is the occupation that is more prevalent in this dataset.</li>
# <li>Can see that only few has credit as default</li>
# <li>Nearly 40% of the customers has housing loan</li>
# <li> And many of the customer do not have personal loan</li>
# 

# In[ ]:


fig,(a1,a2)=plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'duration', data = dataframe, orient = 'v', ax = a1)
sns.distplot(dataframe.duration, ax = a2)


# In[ ]:


dataframe[dataframe.duration==0]


# <h3>Obv</h3>The customer who has the duration zero indicates that they surely have not talked with bank officers which in turn means they surely have not taken the term deposit

# 
# 
# <h2>Choosing the target column</h2>
# As the objective is to redict the likelihood of a liability customer subscribing , the Target column will be target column.<br>
# And the distribution is as shown

# In[ ]:


dataframe["Target"].hist(bins=2)


# In[ ]:


dataframe["Target"].value_counts()


# As in the data, the count of customer how takes the term deposit is very less compared to who didn't. Due to which there maybe chances that the model perdiction will be effected due to this.<br>
# <h2>Checking the influence of various attributes on customer subscribing term deposit </h2>
# <h3>Influence of Default & Balance on Target</h3>
# 
# 

# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='default',y='balance',data=dataframe,hue='Target',palette='muted')


# <h4>Obv:</h4>  The graph shows the customers those who dont have the credit card in defualt will go for the subcribing the term deposit compared to those who have credit card.<br>
# <h3>Influence of Job & Banlance on target
# </h3>

# In[ ]:


fig,(a1,a2)=plt.subplots(nrows = 2, ncols = 1, figsize = (13, 15))
sns.boxplot(x='job',y='age',data=dataframe,ax=a1)
sns.boxplot(x='job',y='balance',hue='Target',data=dataframe,ax=a2)


# <h4>Obv</h4>a. Management is the occupation that is more prevalent in this dataset.<br>
# b. As expected, the retired are the ones who have the highest median age while student are the lowest.<br>
# c. Management and Retirees are the ones who have the highest balance in their accounts.

#  The graph shows that the customer who has retired like to subscribe the term deposit<br>
# <h3>Influence of Education on target
# </h3>

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(x="education", data=dataframe,hue="Target")


# <h3>Obv</h3>
# The customer with secondary and tertiary education tends to subscribe term deposit as they will have goos income.

# <h3>Influence of Customers' Mortgage on taking personal Loan
# </h3>

# In[ ]:


sns.catplot(x='duration',y='balance',data=dataframe,hue='marital')


# In[ ]:


sns.countplot(x='marital',hue='Target',data=dataframe)


# <h4>Obv</h4>The customers who are divorced looks like having low balance and also customers who Married and Single tends to have surscribed term deposit than divorced customer<br>
# <h3>Influence of contact duration on taking term deposit
# </h3>

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('whitegrid')
avg_duration = dataframe['duration'].mean()

lst = [dataframe]
dataframe["duration_status"] = np.nan

for col in lst:
    col.loc[col["duration"] < avg_duration, "duration_status"] = "below_average"
    col.loc[col["duration"] > avg_duration, "duration_status"] = "above_average"
    
pct_term = pd.crosstab(dataframe['duration_status'], dataframe['Target']).apply(lambda r: round(r/r.sum(), 2) * 100, axis=1)


ax = pct_term.plot(kind='bar', stacked=False, cmap='RdBu')
plt.xlabel("Duration Status", fontsize=18);
plt.ylabel("Percentage (%)", fontsize=18)

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))
    

plt.show()


# <h3>Obv</h3>The customers who have talked with bank more than average duration are likely to subscribe the term deposit than customers who have talked below average duration<br>
# <h2>Corelation of Attributes</h2>

# In[ ]:


labelencoder_X=LabelEncoder()
dataframe.Target=labelencoder_X.fit_transform(dataframe.Target)
corelation=dataframe.corr()


# In[ ]:


plt.figure(figsize=(10,10))
a=sns.heatmap(corelation,annot=True)


# <h3>Obv</h3>
# <li>We can see Target and duration are corelated</li>
# <li> Looks like previous number of contacts performed and pdays are corelated</li>
# <li>And also the last contacted day and campaign are also corelated</li>
# 
# <h2>Preparing the data for modeling</h2>
# <h3>Label encoding the attributes having object datatype</h3>

# In[ ]:


labelencoder_X = LabelEncoder()
dataframe['job']      = labelencoder_X.fit_transform(dataframe['job']) 
dataframe['marital']  = labelencoder_X.fit_transform(dataframe['marital']) 
dataframe['education']= labelencoder_X.fit_transform(dataframe['education']) 
dataframe['default']  = labelencoder_X.fit_transform(dataframe['default']) 
dataframe['housing']  = labelencoder_X.fit_transform(dataframe['housing']) 
dataframe['loan']     = labelencoder_X.fit_transform(dataframe['loan'])
dataframe['contact']     = labelencoder_X.fit_transform(dataframe['contact']) 
dataframe['day'] = labelencoder_X.fit_transform(dataframe['day']) 
dataframe['month'] = labelencoder_X.fit_transform(dataframe['month']) 


# Its better to remove duration column from dataset but checking how it may work if we group the duration upon the quartile range on the target attribute.<br>
# Also grouping the data of age attribute upon quartile range.

# In[ ]:


print('1º Quartile: ', dataframe['duration'].quantile(q = 0.25))
print('2º Quartile: ', dataframe['duration'].quantile(q = 0.50))
print('3º Quartile: ', dataframe['duration'].quantile(q = 0.75))
print('4º Quartile: ', dataframe['duration'].quantile(q = 1.00))

print('1º Quartile: ', dataframe['age'].quantile(q = 0.25))
print('2º Quartile: ', dataframe['age'].quantile(q = 0.50))
print('3º Quartile: ', dataframe['age'].quantile(q = 0.75))
print('4º Quartile: ', dataframe['age'].quantile(q = 1.00))


# In[ ]:


dataframe.loc[dataframe['age'] <= 33, 'age'] = 1
dataframe.loc[(dataframe['age'] > 33) & (dataframe['age'] <= 39), 'age'] = 2
dataframe.loc[(dataframe['age'] > 39) & (dataframe['age'] <= 48), 'age'] = 3
dataframe.loc[(dataframe['age'] > 48) & (dataframe['age'] <= 98), 'age'] = 4

dataframe.loc[dataframe['duration'] <= 103, 'duration'] = 1
dataframe.loc[(dataframe['duration'] > 103) & (dataframe['duration'] <= 180)  , 'duration']    = 2
dataframe.loc[(dataframe['duration'] > 180) & (dataframe['duration'] <= 319)  , 'duration']   = 3
dataframe.loc[(dataframe['duration'] > 319) & (dataframe['duration'] <= 644.5), 'duration'] = 4
dataframe.loc[dataframe['duration']  > 644.5, 'duration'] = 5


# In[ ]:


dataframe['poutcome'].replace(['unknown', 'failure','other','success'], [1,2,3,4], inplace  = True)
#dataframe['Target'].replace(['no', 'yes'], [0,1], inplace  = True)


# In[ ]:


dataframe.head()


# 
# <h2>Classification Models</h2>
# <h3> Splitting the Data</h3>

# In[ ]:


dataframe.columns


# In[ ]:


features=['age', 'job', 'marital', 'education', 'default','balance','duration',
       'housing', 'loan', 'contact','month', 'day','campaign','pdays','previous','poutcome']
X=dataframe[features]
Y=dataframe['Target']         


# As for the side note as the range of various attribute vary  a lot (like range of age is 23 to 67 where are the income is 8 to 224 having different units),there may come a need to normalize the data.<br>
# But for Logestic Regression and Naive Bayes classification Normalization is not required as it doesnot effect it.<br>
# For KNN algorithm normalization is required as it depends on distance of data points.

# Splitting the model in 7:3 ratio

# In[ ]:


train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 


# In[ ]:


train_X.head()


# In[ ]:


test_X.count()


# In[ ]:


test_X.head()


# In[ ]:



scaling = StandardScaler()
train_X = scaling.fit_transform(train_X)
test_X = scaling.fit_transform(test_X)
print(train_X)


# <h2>Using Logestic Regression for prediction</h2><br>
# Trianing the model

# In[ ]:


LR_Model=LogisticRegression(random_state=1)
Logestic_Model=LR_Model.fit(train_X,train_y)
Logestic_Model


# Predicting from the trained model and showing the confusion matrix

# In[ ]:


predict=LR_Model.predict(test_X)
print(predict[0:1000])
metrics=confusion_matrix(test_y,predict)
metrics


# In[ ]:


sns.heatmap(metrics,annot=True,fmt='g',cmap='Blues')


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


probability=Logestic_Model.predict_proba(test_X)
pred=probability[:,1]
fpr,tpr,thresh=roc_curve(test_y,pred)
roc_auc=auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label='AUC =%0.2f'%roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


LR_accuracy=accuracy_score(test_y,predict)
LR_accuracy


# In[ ]:


LR_AUC=roc_auc
LR_AUC


# In[ ]:


LR_Gini = 2*roc_auc - 1
LR_Gini


# <h3>Obv:</h3>
# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is not so good (417 out of 1134).<br>
# The confusion matrix shows that the model prediction of customer subscribe term deposit not that satisfactory. This maybe due to lack of available data of customers who goes for subscribing term deposit for the model to learn.<br>As we can see the accuracy is 89.6% along with Area Under the Curve is87.3% which pretty good.<br>
# The Gini value is 0.736.<br><br>
# <h2> Using KNN Classification Model</h2>
# <br>
# 
# 

# Predicting from the trained model

# Checking for suitable number for number of nearest neighbor the model should look into for prediction

# In[ ]:


n=[1,3,5,7,11,13,15,17,19,21,23,25,27,29,31,33,35]
accuracy_scores=[]
for i in n:
    KNN_Model=KNeighborsClassifier(n_neighbors=i)
    KNN_Model.fit(train_X,train_y)
    predict=KNN_Model.predict(test_X)
    accuracy_scores.append(accuracy_score(test_y,predict))
accuracy_scores
    


# Looks like for N_neigbors = 11 the accuracy score is highest for this model.<br>
# Checking fro whether we should use  manhattan_distance (p=1) or euclidean_distance (p=2)

# In[ ]:


p=[1,2]
accuracy_scores=[]
for i in p:
    KNN_Model=KNeighborsClassifier(n_neighbors=11,p=i)
    KNN_Model.fit(train_X,train_y)
    predict=KNN_Model.predict(test_X)
    accuracy_scores.append(accuracy_score(test_y,predict))
accuracy_scores

    


# The p value doesnot make a difference in this model.

# In[ ]:


KNN_Model=KNeighborsClassifier(n_neighbors=13,p=1)  
KNN_Model.fit(train_X,train_y)
predict=KNN_Model.predict(test_X)
print(predict[0:200,])
Knn_matrics=confusion_matrix(test_y,predict)
Knn_matrics


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


sns.heatmap(Knn_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs = KNN_Model.predict_proba(test_X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


KNN_accuracy=accuracy_score(test_y,predict)
KNN_accuracy


# In[ ]:


KNN_Gini=2*roc_auc-1
KNN_Gini


# In[ ]:


KNN_AUC=roc_auc
KNN_AUC


# <h3>Obv:</h3>
# The heat map shows that the model perdicts customer who dont take term deposit pretty well but prediction on wheather customer taking term deposit is not good compared to Logestic regression (372 out of 1551).<br>
# The confusion matrix shows that the model prediction of customer taking loan is comparitively ok.<br>As we can see the accuracy has increased to 89.7% along with Area Under the Curve is 86.5% which pretty good.<br>
# The Gini value is 0.73.<br>The AUC and Gini value decreased but not by huge difference.<br>
# <h2> Using Naive Bayes Classification Model</h2>
# <br>
# <br>
# Splitting the data again and Training the model.
# 

# In[ ]:



NB_Model=GaussianNB()
naiveB_Model=NB_Model.fit(train_X,train_y)
naiveB_Model


# Predicting with the above trained model.

# In[ ]:


predict=NB_Model.predict(test_X)
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


NB_matrics=confusion_matrix(test_y,predict)
NB_matrics


# In[ ]:


sns.heatmap(NB_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=NB_Model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


NB_accuracy=accuracy_score(test_y,predict)
NB_accuracy


# In[ ]:


NB_Gini=2*roc_auc-1
NB_Gini


# In[ ]:


NB_AUC=roc_auc
NB_AUC


# <h3>Obv:</h3>
# The heat map shows that the model perdicts customer how dont take personal loan pretty well as well as  prediction on wheather customer taking loan is also good compared to Logestic regression (683 out of 1551).<br>
# The confusion matrix shows that the model prediction of customer taking loan is comparitively satisfactory.<br>As we can see the accuracy has decreased  to 84.1% compared to along with Area Under the Curve is 80.5% which pretty good.<br>
# The Gini value is 0.60.9.<br>
# <h2>SVC model</h2>
# 
# <br>
# Splitting the data again and Training the model.
# 

# In[ ]:



svc=SVC(kernel='sigmoid',random_state=1,probability=True)
svc_Model=svc.fit(train_X,train_y)
svc_Model


# Predicting with the above trained model.

# In[ ]:


predict=svc_Model.predict(test_X)
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


svc_matrics=confusion_matrix(test_y,predict)
svc_matrics


# In[ ]:


sns.heatmap(svc_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=svc_Model.predict_proba(test_X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


svc_accuracy=accuracy_score(test_y,predict)
svc_accuracy


# In[ ]:


svc_Gini=2*roc_auc-1
svc_Gini


# In[ ]:


svc_AUC=roc_auc
svc_AUC


# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is not so good (512 out of 1551).<br>
# The confusion matrix shows that the model prediction of customer subscribe term deposit not that satisfactory. This maybe due to lack of available data of customers who goes for subscribing term deposit for the model to learn.<br>As we can see the accuracy is 83.4% along with Area Under the Curve is 69.6% which pretty bad.<br>
# The Gini value is 0.392.<br><br>
# <h2> Comparing Standard Models</h2>

# In[ ]:


data=[[LR_accuracy,LR_Gini,LR_AUC],[KNN_accuracy,KNN_Gini,KNN_AUC],[NB_accuracy,NB_Gini,NB_AUC],[svc_accuracy,svc_Gini,svc_AUC]]


# In[ ]:


comparison=pd.DataFrame(data,index=['Logestic','KNN','Naive Bayes','SVC'],columns=['Accuracy','Gini','AUC'])
comparison


# Till now the KNN model holds best as the accuracy is better than others.
# <h2>Using Essemble techniques</h2>
# <h2>Decision Tree</h2>
# Training the model and selecting the best tree.

# In[ ]:


dtree_accuracy_score=[]
for m in range(1,18):
    dtree=DecisionTreeClassifier(criterion='gini',max_depth=m,random_state=1)
    dtree_model=dtree.fit(train_X,train_y)
    predict=dtree.predict(test_X)
    dtree_accuracy_score.append(accuracy_score(test_y,predict))
dtree_accuracy_score


# In[ ]:


dtree=DecisionTreeClassifier(criterion='gini',max_depth=8,random_state=1)
dtree_model=dtree.fit(train_X,train_y)
predict=dtree.predict(test_X)


# Predicitng from the model

# In[ ]:


predict=dtree_model.predict(test_X)
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


dtree_matrics=confusion_matrix(test_y,predict)
dtree_matrics


# In[ ]:


sns.heatmap(dtree_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=dtree_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


dtree_accuracy=accuracy_score(test_y,predict)
dtree_accuracy


# In[ ]:


dtree_Gini=2*roc_auc-1
dtree_Gini


# In[ ]:


dtree_AUC=roc_auc
dtree_AUC


# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is comparitively good (597 out of 1551).<br>
# The confusion matrix shows that the model prediction of customer subscribe term deposit not that satisfactory. This maybe due to lack of available data of customers who goes for subscribing term deposit for the model to learn.<br>As we can see the accuracy is 90.23% along with Area Under the Curve is 86.6% which pretty gppd.<br>
# The Gini value is 0.866.<br><br>
# <h2> Bagging Model</h2>
# Training the model.

# In[ ]:


#bagging=BaggingClassifier(n_estimators=50,random_state=1)
bagging_accuracy_score=[]
for m in range(50,90):
    bagging=BaggingClassifier(n_estimators=m,random_state=1)    
    bagging_model=bagging.fit(train_X,train_y)
    predict=dtree.predict(test_X)
    bagging_accuracy_score.append(accuracy_score(test_y,predict))
bagging_accuracy_score


# Predicting from the model

# In[ ]:


bagging=BaggingClassifier(n_estimators=50,random_state=1)    
bagging_model=bagging.fit(train_X,train_y)
predict=bagging.predict(test_X)


# In[ ]:


predict=bagging_model.predict(test_X)
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


bagging_matrics=confusion_matrix(test_y,predict)
bagging_matrics


# In[ ]:


sns.heatmap(bagging_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=bagging_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


bagging_accuracy=accuracy_score(test_y,predict)
bagging_accuracy


# In[ ]:


bagging_Gini=2*roc_auc-1
bagging_Gini


# In[ ]:


bagging_AUC=roc_auc
bagging_AUC


# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is not so good (721 out of 1551).<br>
#  we can see the accuracy is 90.32% along with Area Under the Curve is 90.7% which good.<br>
# The Gini value is 0.8157.<br><br>
# <h2> AdaBoosting Model</h2>
# Traing the data

# In[ ]:


aboost_accuracy_score=[]
for m in range(50,90):
    aboost=AdaBoostClassifier(n_estimators=m,random_state=1)    
    aboost_model=aboost.fit(train_X,train_y)
    predict=aboost.predict(test_X)
    aboost_accuracy_score.append(accuracy_score(test_y,predict))
aboost_accuracy_score


# In[ ]:


aboost=AdaBoostClassifier(n_estimators=66,random_state=1)    
aboost_model=aboost.fit(train_X,train_y)
predict=aboost.predict(test_X)


# Predicting from the model.

# In[ ]:


predict=aboost_model.predict(test_X)
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


aboost_matrics=confusion_matrix(test_y,predict)
aboost_matrics


# In[ ]:


sns.heatmap(aboost_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=aboost_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


aboost_accuracy=accuracy_score(test_y,predict)
aboost_accuracy


# In[ ]:


aboost_Gini=2*roc_auc-1
aboost_Gini


# In[ ]:


aboost_AUC=roc_auc
aboost_AUC


# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is not so good (555 out of 1551).<br>
#  we can see the accuracy is 90.1% along with Area Under the Curve is 90.1% which good.<br>
# The Gini value is 0.801.<br><br>
# <h2> Gradient Boosting Model</h2>
# Training the model

# In[ ]:


gboost_accuracy_score=[]
for m in range(50,90):
    gboost=GradientBoostingClassifier(n_estimators=m,random_state=1)    
    gboost_model=gboost.fit(train_X,train_y)
    predict=gboost.predict(test_X)
    gboost_accuracy_score.append(accuracy_score(test_y,predict))
aboost_accuracy_score


# In[ ]:


gboost=GradientBoostingClassifier(n_estimators=66,random_state=1)    
gboost_model=gboost.fit(train_X,train_y)


# Predicting from the model

# In[ ]:


predict=gboost_model.predict(test_X)`
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


gboost_matrics=confusion_matrix(test_y,predict)
gboost_matrics


# In[ ]:


sns.heatmap(gboost_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=gboost_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


gboost_accuracy=accuracy_score(test_y,predict)
gboost_accuracy


# In[ ]:


gboost_Gini=2*roc_auc-1
gboost_Gini


# In[ ]:


gboost_AUC=roc_auc
gboost_AUC


# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is better (569 out of 1551).<br>
#  we can see the accuracy is 90.37% along with Area Under the Curve is 91.1% which good.<br>
# The Gini value is 0.8222.<br><br>
# <h2> Random Forest Model</h2>
# Training the model

# In[ ]:


R_forest=RandomForestClassifier(n_estimators=50,random_state=1,max_features=15)    
R_forest_model=R_forest.fit(train_X,train_y)
predict=R_forest.predict(test_X)


# Predicting from the model

# In[ ]:


predict=R_forest_model.predict(test_X)
predict[0:200,]


# In[ ]:


ac_score=accuracy_score(test_y,predict)
ac_score


# In[ ]:


print(classification_report(test_y,predict))


# In[ ]:


R_forest_matrics=confusion_matrix(test_y,predict)
R_forest_matrics


# In[ ]:


sns.heatmap(R_forest_matrics,annot=True,cmap='Blues',fmt='g')


# In[ ]:


probs=R_forest_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


R_forest_accuracy=accuracy_score(test_y,predict)
R_forest_accuracy


# In[ ]:


R_forest_Gini=2*roc_auc-1
R_forest_Gini


# In[ ]:


R_forest_AUC=roc_auc
R_forest_AUC


# The heat map shows that the model perdicts customer how dont subscribe term deposit pretty well whereas prediction on customer subscribing is good (695 out of 1551).<br>
#  we can see the accuracy is 89.89% along with Area Under the Curve is 90.7% which good.<br>
# The Gini value is 0.8145.<br><br>
# <h2> Comparing all the Model</h2>

# In[ ]:


data=[[LR_accuracy,LR_Gini,LR_AUC],[KNN_accuracy,KNN_Gini,KNN_AUC],[NB_accuracy,NB_Gini,NB_AUC],[svc_accuracy,svc_Gini,svc_AUC],
     [dtree_accuracy,dtree_Gini,dtree_AUC],[bagging_accuracy,bagging_Gini,bagging_AUC],[aboost_accuracy,aboost_Gini,aboost_Gini],
     [gboost_accuracy,gboost_Gini,gboost_AUC],[R_forest_accuracy,R_forest_Gini,R_forest_AUC]]


# In[1]:


comparison=pd.DataFrame(data,index=['Logestic','KNN','Naive Bayes','SVC','Decision Tree','Bagging','AdaBoosting','GradientBoosting','Random Forest'],columns=['Accuracy','Gini','AUC'])
comparison


# As for the above matrix, the accuracy of Gradient Boosting model is highest among others along with the Gini value and AUC is also better than other model <br> 
# <h3>So, in this case Gradient Boosting would be the best model to use for predicting the likelihood of a liability customer purchasing Term Deposit.<h3>

# <h5>Thank you</h5>
