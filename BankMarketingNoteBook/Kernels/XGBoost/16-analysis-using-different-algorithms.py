#!/usr/bin/env python
# coding: utf-8

# ##  Bank Marketing
# 
# 
# **Abstract:** 
# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# **Data Set Information:**
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# ###  Attribute Information:
# 
# ####  Bank client data:
# 
#  - Age (numeric)
#  - Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
#  - Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)
#  - Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school',  'illiterate', 'professional.course', 'university.degree', 'unknown')
#  - Default: has credit in default? (categorical: 'no', 'yes', 'unknown')
#  - Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
#  - Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
# 
#     
# ####  Related with the last contact of the current campaign:
# 
#  - Contact: contact communication type (categorical:
#    'cellular','telephone')
#  - Month: last contact month of year (categorical: 'jan', 'feb', 'mar',
#    ..., 'nov', 'dec')
#  - Day_of_week: last contact day of the week (categorical:
#    'mon','tue','wed','thu','fri')
#  - Duration: last contact duration, in seconds (numeric). Important
#    note: this attribute highly affects the output target (e.g., if
#    duration=0 then y='no'). Yet, the duration is not known before a call
#    is performed. Also, after the end of the call y is obviously known.
#    Thus, this input should only be included for benchmark purposes and
#    should be discarded if the intention is to have a realistic
#    predictive model.
# 
#     
# ####  Other attributes:
# 
#  - Campaign: number of contacts performed during this campaign and for
#    this client (numeric, includes last contact)
#  - Pdays: number of days that passed by after the client was last
#    contacted from a previous campaign (numeric; 999 means client was not
#    previously contacted)
#  - Previous: number of contacts performed before this campaign and for
#    this client (numeric)
#  - Poutcome: outcome of the previous marketing campaign (categorical:
#    'failure','nonexistent','success')
# 
#     
# ####  Social and economic context attributes
#  - Emp.var.rate: employment variation rate - quarterly indicator
#    (numeric)
#  - Cons.price.idx: consumer price index - monthly indicator (numeric)
#  - Cons.conf.idx: consumer confidence index - monthly indicator
#    (numeric)
#  - Euribor3m: euribor 3 month rate - daily indicator (numeric)
#  - Nr.employed: number of employees - quarterly indicator (numeric)
# 
# ####  Output variable (desired target):
# 
#  - y - has the client subscribed a term deposit? (binary: 'yes', 'no')
# 

# ### Import important libraries and Dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("../input/bank-marketing/bank-additional-full.csv",delimiter=';')
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
data.head()


# ***Data Reading***

# In[ ]:


data.isnull().sum().sum()


# In[ ]:


data['job'].unique()


# In[ ]:


data['marital'].unique()


# In[ ]:


data['education'].unique()


# In[ ]:


print("Default \n",data['default'].unique())
print("Loan \n",data['loan'].unique())
print("Housing \n",data['housing'].unique())
print("contact \n",data['contact'].unique())
print("poutcome \n",data['poutcome'].unique())
print("day_of_week \n",data['day_of_week'].unique())
print("month \n",data['month'].unique())


# ### Data Visualization

# In[ ]:


# What kind of 'marital clients' this bank have, if you cross marital with default, loan or housing, there is no relation
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(60,15))

##Marital Count Status
sns.set(font_scale=5)
sns.countplot(x = 'marital', data=data,ax=ax1, order=['married', 'single', 'divorced', 'unknown'])
ax1.set_title('Marital count status', fontsize=35)
ax1.set_xlabel('Marital', fontsize=35)
ax1.set_ylabel('Count', fontsize=35)
ax1.tick_params(labelsize=35)

###Education Count Distribution
sns.set(font_scale=5)
sns.countplot(x = 'education', data = data, ax=ax2,order=['basic.4y', 'high.school', 'basic.6y', 'basic.9y',
       'professional.course', 'unknown', 'university.degree',
       'illiterate'])
ax2.set_title('Education Count Distribution', fontsize=35)
ax2.set_xlabel('Education', fontsize=35)
ax2.set_ylabel('Count', fontsize=35)

ax2.tick_params(labelsize=35)
sns.despine()


# In[ ]:


# Default, has credit in default ?
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = data, ax = ax1, order = ['no', 'unknown', 'yes'])
ax1.set_title('Default', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)

# Housing, has housing loan ?
sns.countplot(x = 'housing', data = data, ax = ax2, order = ['no', 'unknown', 'yes'])
ax2.set_title('Housing', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)

# Loan, has personal loan ?           
sns.countplot(x = 'loan', data = data, ax = ax3, order = ['no', 'unknown', 'yes'])
ax3.set_title('Loan', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)


plt.subplots_adjust(wspace=0.25)


# In[ ]:



##
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
##Count plot for contact variable
sns.countplot(x = 'contact', data = data, ax = ax1, order = ['telephone','cellular'])
ax1.set_title('contact', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)

##Count plot for Marital variable
sns.countplot(x = 'marital', data = data, ax = ax2, order = ['married', 'single', 'divorced', 'unknown'])
ax2.set_title('marital', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)


# In[ ]:


sns.set(font_scale=1.5)
sns.lmplot( x="age", y="cons.conf.idx", data=data, fit_reg=False, hue='emp.var.rate', legend=False)


# In[ ]:


sns.set(font_scale=1.5)
sns.jointplot(x='campaign',y='age',data=data)


# In[ ]:


##Correlation plot
plt.subplots(figsize=(25,25))
sns.set(font_scale=2)
sns.heatmap(data.corr(), annot=True)
plt.show()


# ### Data preprocessing

# In[ ]:


data.info()


# In[ ]:


##Converting categorical data into numerical data by using Label Encoder.
stringcols = ('job','marital','education', 'default','housing','loan','month', 'day_of_week', 'contact','poutcome','emp.var.rate','cons.conf.idx','euribor3m','nr.employed','y')
from sklearn.preprocessing import LabelEncoder


# In[ ]:


for c in stringcols:
    lbl = LabelEncoder() 
    lbl.fit(list(data[c].values)) 
    data[c] = lbl.transform(list(data[c].values))


# In[ ]:


data.isnull().sum()


# ### Analysis
# 
# 
# Here in Analysis, i have taken all independent variable for better accuracy
# 
# and my dependent value would be ***'y':{'yes':"1",'No':"0"}*** (Is client has taken term deposite or not)

# In[ ]:


x=data[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 
        'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,data['y'] ,test_size = 0.2, random_state = 100)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ***SVM***
# > making model using different kernel's in svm (SVC)

# **Sigmoid SVC**

# In[ ]:


from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC


# In[ ]:


svcS= SVC(kernel = 'sigmoid')
svcS.fit(X_train, y_train)
svcSpred = svcS.predict(X_test)


# In[ ]:


print("Confusion Matrix using sigmoid kernel \n",confusion_matrix(y_test, svcSpred))
print("Accuracy Score using sigmoid kernel \n",round(accuracy_score(y_test, svcSpred),2)*100)


# In[ ]:


svcS_fpr,svcS_tpr,threshold=roc_curve(y_test,svcSpred)


# In[ ]:


auc_svcS=auc(svcS_fpr,svcS_tpr)


# In[ ]:


SVCCV = (cross_val_score(svcS, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# In[ ]:


print(SVCCV)


# In[ ]:


from sklearn.svm import SVC
svcR= SVC(kernel = 'rbf')
svcR.fit(X_train, y_train)
svcRpred = svcR.predict(X_test)


# In[ ]:


print("Confusion Matrix using rbf kernel \n",confusion_matrix(y_test, svcRpred))
print("Accuracy Score using rbf kernel \n",round(accuracy_score(y_test, svcRpred),2)*100)


# In[ ]:


svcR_fpr,svcR_tpr,threshold=roc_curve(y_test,svcRpred)
auc_svcR=auc(svcR_fpr,svcR_tpr)


# In[ ]:


SVCCV1 = (cross_val_score(svcR, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# In[ ]:


SVCCV1


# In[ ]:


from sklearn.svm import SVC
svcP= SVC(kernel = 'poly',random_state=42)
svcP.fit(X_train, y_train)
svcPpred = svcP.predict(X_test)
print("Confusion matrix using polynomial kernel \n",confusion_matrix(y_test, svcPpred))
print("Accuracy Score using polynomial kernel \n",round(accuracy_score(y_test, svcPpred),2)*100)


# In[ ]:


svcP_fpr,svcP_tpr,threshold=roc_curve(y_test,svcPpred)
auc_svcP=auc(svcP_fpr,svcP_tpr)


# In[ ]:


SVCCV2 = (cross_val_score(svcP, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
SVCCV2


# ***Random Forest***

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=1,random_state=42)
forest.fit(X_train,y_train)


# In[ ]:


forestpred = forest.predict(X_test)


# In[ ]:


print("Accuracy on the training subset:(:.3f)",format(forest.score(X_train,y_train)))
print("Accuracy on the testing subset:(:.3f)",format(forest.score(X_test,y_test)))


# In[ ]:


forest_fpr,forest_tpr,threshold=roc_curve(y_test,forestpred)
auc_forest=auc(forest_fpr,forest_tpr)


# In[ ]:


forestCV = (cross_val_score(forest, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
forestCV


# ***Decision Tree***

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini',random_state=42) #criterion = entopy, gini
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)

print(confusion_matrix(y_test, dtreepred))
print(round(accuracy_score(y_test, dtreepred),2)*100)


# In[ ]:


dtree_fpr,dtree_tpr,threshold=roc_curve(y_test,dtreepred)
auc_dtree=auc(dtree_fpr,dtree_tpr)


# In[ ]:


print("Accuracy on the training subset:(:.3f)",format(dtree.score(X_train,y_train)))
print("Accuracy on the testing subset:(:.3f)",format(dtree.score(X_test,y_test)))


# In[ ]:


dtreecv = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
dtreecv


# ***XgBoost Classifier***

# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=20,random_state=42)
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)

print(confusion_matrix(y_test, xgbprd ))
print(round(accuracy_score(y_test, xgbprd),2)*100)


# In[ ]:


xgb_fpr,xgb_tpr,threshold=roc_curve(y_test,xgbprd)
auc_xgb=auc(xgb_fpr,xgb_tpr)


# In[ ]:


xgbcv = (cross_val_score(xgb, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
xgbcv


# ***KNN Classifier***

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#Neighbors
neighbors = np.arange(0,25)

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = KFold(n_splits=10, random_state=123)
    scores =cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)

print(confusion_matrix(y_test, knnpred))
print(round(accuracy_score(y_test, knnpred),2)*100)


# In[ ]:


knn_fpr,knn_tpr,threshold=roc_curve(y_test,knnpred)
auc_knn=auc(knn_fpr,knn_tpr)


# In[ ]:


KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# In[ ]:


KNNCV


# ## Evaluation

# In[ ]:


plt.figure(figsize=(10,8),dpi=100)
plt.plot(svcS_fpr,svcS_tpr,linestyle="-",label='svcS (auc=%0.3f)' %auc_svcS)
plt.plot(svcP_fpr,svcP_tpr,marker='.',label='svcP (auc=%0.3f)' %auc_svcP)
plt.plot(svcR_fpr,svcR_tpr,marker='o',label='svcR (auc=%0.3f)' %auc_svcR)
plt.xlabel('FPR--->')
plt.ylabel('TPR--->')
plt.title("Comparison of every kernel's in ROC Curve of SVC")
plt.legend()
plt.show()
plt.savefig('ROC Curve.png')


# In[ ]:


plt.figure(figsize=(10,8),dpi=100)
plt.plot(forest_fpr,forest_tpr,linestyle="-",label='forest (auc=%0.3f)' %auc_forest)
plt.plot(dtree_fpr,dtree_tpr,marker='.',label='dtree (auc=%0.3f)' %auc_dtree)
plt.plot(knn_fpr,knn_tpr,marker='o',label='knn (auc=%0.3f)' %auc_knn)
plt.xlabel('FPR--->')
plt.ylabel('TPR--->')
plt.title("ROC Curve of dtree,Random_forest,Knn algorithms ")
plt.legend()
plt.show()
plt.savefig('ROC Curve(different algorithms).png')


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print("Report of KNN algo: \n", classification_report(y_test,knnpred))
print(" Predicted KNN",knnpred)
print("\n")
print("Report of xgboost algo: \n", classification_report(y_test,xgbprd))
print("Predicted Xgboost",xgbprd )
print("\n")
print("Report of dtree algo: \n", classification_report(y_test,dtreepred))
print("Predicted Decision Tree", dtreepred)
print("\n")
print("Report of random forest algo: \n", classification_report(y_test,forestpred))
print("Predicted Random Forest",forestpred)


# ***Here***
# > ***'0' means "client hasn't subscribed a term deposite"***
# 
# > ***'1' means "Client has subscribed a term deposite"***

# ***KNN Algorithm***
# > Accuracy in KNN algorithm is 0.91 means our model is 91.00% accurate.
# In precision,High Precision means that false positive rate is low we have got 0.93 for 0 (not subscriber of term deposite)
# and 0.64 precision for 1 (Subscriber of term deposite)
# we got recall as 0.97 for 0 (not subscriber of term deposite) and 0.43 for 1 (Subscriber of term deposite).
# f1-score gives better result than accuracy especially when we have uneven class distribution,Accuracy works better if false positive and false negative have similar cost if it is different then we have to look into recall and precision.
# In our case accuracy score and f1 score differ much,f1 score is 0.95 for 0(not subscriber of term deposite) and 0.52 for 1(Subscriber of term deposite)
# 
# ***xgboost Algorithm***
# > Accuracy in xgboost algorithm is 0.92 means our model is 92.00% accurate.
# In precision,High Precision means that false positive rate is low we have got 0.94 for 0 (not subscriber of term deposite)
# and 0.65 precision for 1 (Subscriber of term deposite)
# we got recall as 0.97 for 0 (not subscriber of term deposite) and 0.52 for 1 (Subscriber of term deposite).
# f1-score gives better result than accuracy especially when we have uneven class distribution,Accuracy works better if false positive and false negative have similar cost if it is different then we have to look into recall and precision.
# In our case accuracy score and f1 score differ much,f1 score is 0.95 for 0(not subscriber of term deposite) and 0.58 for1(Subscriber of term deposite)
# 
# ***decision tree Algorithm***
# 
# > Accuracy in decision tree algorithm is 0.89 means our model is 89.00% accurate.
# In precision,High Precision means that false positive rate is low we have got 0.94 for 0 (not subscriber of term deposite)
# and 0.49 precision for 1 (Subscriber of term deposite)
# we got recall as 0.93 for 0 (not subscriber of term deposite) and 0.52 for 1 (Subscriber of term deposite).
# f1-score gives better result than accuracy especially when we have uneven class distribution,Accuracy works better if false positive and false negative have similar cost if it is different then we have to look into recall and precision.
# In our case accuracy score and f1 score differ much,f1 score is 0.94 for 0(not subscriber of term deposite) and 0.50 for 1 (Subscriber of term deposite)
# 
# ***Random Forest Algorithm***
# 
# > Accuracy in decision tree algorithm is 0.89 means our model is 89.00% accurate.
# In precision,High Precision means that false positive rate is low we have got 0.94 for 0 (not subscriber of term deposite)
# and 0.50 precision for 1 (Subscriber of term deposite)
# we got recall as 0.94 for 0 (not subscriber of term deposite) and 0.49 for 1 (Subscriber of term deposite).
# f1-score gives better result than accuracy especially when we have uneven class distribution,Accuracy works better if false positive and false negative have similar cost if it is different then we have to look into recall and precision.
# In our case accuracy score and f1 score differ much,f1 score is 0.94 for 0(not subscriber of term deposite) and 0.49 for 1 (Subscriber of term deposite)
# 

# In[ ]:


print("Confusion metrics of SVC using 'rbf' kernel : \n", classification_report(y_test,svcRpred))
print("\n")
print("Confusion metrics of SVC using 'sigmoid' kernel: \n", classification_report(y_test,svcSpred))
print("\n")
print("Confusion metrics of SVC using 'Polynomial' kernel : \n", classification_report(y_test,svcPpred))


# > ***In SVM we have used different kernel's and we got best accuracy using rbf kernel***

# ### From the above results we can conclude that xgboost is giving the best model for this classification problem.
# 

# In[ ]:




