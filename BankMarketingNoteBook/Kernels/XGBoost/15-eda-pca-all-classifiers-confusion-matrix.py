#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[ ]:


df = pd.read_csv('../input/bankmarketingdatasetbank/bank.csv')


# In[ ]:


import pandas_profiling as pp
pp.ProfileReport(df)


# ## Dataframe Check

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include='object')


# # **Data Distribution**

# ## Categorical Data

# In[ ]:


categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['deposit']))]
categorical_features


# In[ ]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))


# In[ ]:


#check count based on categorical features
plt.figure(figsize=(15,80), facecolor='white')
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(y=categorical_feature,data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()


# ## Relationship between Categorical Features and Label

# In[ ]:


#check target label split over categorical features
#Find out the relationship between categorical variable and dependent variable
for categorical_feature in categorical_features:
    sns.catplot(x='deposit', col=categorical_feature, kind='count', data= df)
plt.show()


# In[ ]:


for categorical_feature in categorical_features:
    print(df.groupby(['deposit',categorical_feature]).size())


# ##  Numerical Features

# In[ ]:


numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['deposit']))]
print('Number of numerical variables: ', len(numerical_features))


df[numerical_features].head()


# ##  Discrete Numerical Features

# In[ ]:


discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# ## Continous Numerical Features

# In[ ]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['deposit']]
print("Continuous feature Count {}".format(len(continuous_features)))


# ## Distribution of Continous Numerical Features

# In[ ]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for continuous_feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.distplot(df[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()


# ## Relation between Continous numerical Features and Labels

# In[ ]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(x="deposit", y= df[feature], data=df)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()


# ## Feature Analysis

# In[ ]:


print(df.isnull().sum())


# In[ ]:


plt.figure(figsize=(15, 5))
sns.countplot(x = "job", data = df, label = "Count")
plt.show()


# In[ ]:


sns.countplot(x = "marital", data=df, hue = "deposit")
plt.show()


# In[ ]:


plt.figure(figsize=(15, 5))
sns.countplot(x = "education", data = df, hue = "deposit")
plt.show()


# If you want, we could integrate "primary" and "secondary" as they have similar factorial level. But I think the categories are reasonable. We could make the class by number such as 1: primary, 2: secondary, 3: tertiary, however, it is difficult how to classify the "unknown". Then it might be treated as the categories separately.

# In[ ]:


sns.countplot(x ="default", data = df, hue = "deposit")
plt.show()


# In[ ]:


sns.countplot(x = "contact", data = df, hue = "deposit")
plt.show()


# In[ ]:


sns.countplot(x = "loan", data = df,label = "Count")
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.heatmap(df.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")


# In[ ]:


sns.countplot(x = "month", data = df, hue= "deposit")


# In[ ]:


sns.countplot(x = "poutcome", data = df, hue= "deposit")
plt.show()


# In[ ]:


sns.boxplot(x = "deposit", y = "duration", data = df)
plt.show()


# In[ ]:


sns.violinplot(x = "deposit", y = "duration", data = df)
plt.show()


# ## Find Outliers

# In[ ]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for numerical_feature in numerical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(df[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# ## Correlation between numerical features

# In[ ]:


cor_mat=df.corr()
fig = plt.figure(figsize=(15,7))
sns.heatmap(cor_mat,annot=True)


# # **Feature Engineering**

# ##  Convert Categorical into Numeric Form

# In[ ]:


df.dropna(inplace=True)


# ## Checking Outliers

# In[ ]:


df.plot.box()
plt.xticks(list(range(len(df.columns))),df.columns, rotation="vertical")


# ## Checking outlier of age attribute
# 

# In[ ]:


ax = sns.boxplot(x="age" , data=df)


# ## Handleing Outliers

# In[ ]:


df2=df.copy()


# In[ ]:


low = 0.01
high = 0.99
qdf=df2.quantile([low,high])


# In[ ]:


df2.age= df2.age.apply(lambda v:v if qdf.age[low]< v < qdf.age[high] else np.nan)


# In[ ]:


ax = sns.boxplot(x="age" , data=df2)


# In[ ]:


#Checki Oultier
df2.groupby(['deposit','default']).size()


# In[ ]:


#Removing Outlier
df2.drop(['default'],axis=1, inplace=True)


# In[ ]:


#Checki Oultier
df2.groupby(['deposit','pdays']).size()


# In[ ]:


# drop pdays as it has -1 value for around 40%+ 
df2.drop(['pdays'],axis=1, inplace=True)


# In[ ]:


# remove outliers in feature balance...
df2.groupby(['deposit','balance'],sort=True)['balance'].count()
# these outlier should not be remove as balance goes high, client show interest on deposit


# In[ ]:


# remove outliers in feature duration...
df2.groupby(['deposit','duration'],sort=True)['duration'].count()
# these outlier should not be remove as duration goes high, client show interest on deposit


# In[ ]:


# remove outliers in feature campaign...
df2.groupby(['deposit','campaign'],sort=True)['campaign'].count()


# In[ ]:


df3 = df2[df2['campaign'] < 33]


# In[ ]:


# remove outliers in feature previous...
df3.groupby(['deposit','previous'],sort=True)['previous'].count()


# In[ ]:


df4 = df3[df3['previous'] < 31]


# In[ ]:


cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for col in  cat_columns:
    df4 = pd.concat([df4.drop(col, axis=1),pd.get_dummies(df4[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)


# In[ ]:


bool_columns = ['housing', 'loan','deposit']
for col in  bool_columns:
    df4[col+'_new']=df4[col].apply(lambda x : 1 if x == 'yes' else 0)
    df4.drop(col, axis=1, inplace=True)


# In[ ]:


df4.head()


# In[ ]:


print(df4.isnull().sum())


# In[ ]:


df4.dropna(inplace=True)


# In[ ]:


import pandas_profiling as pp
pp.ProfileReport(df)


# # **Split Dataset into Training set and Test set**

# In[ ]:


X = df4.drop(['deposit_new'],axis=1)
y = df4['deposit_new']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler

features = ['campaign', 'previous']

x_pca = df4.loc[:, features].values
x_pca = StandardScaler().fit_transform(x_pca)


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x_pca)
principalDF = pd.DataFrame(data = principalComponents, columns=['PC1', 'PC2'])


# In[ ]:


pca.components_


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


# Concatenation of dataframes
df_all = pd.concat([principalDF, df4], axis=1)
plt.figure(figsize=(15,5))
sns.heatmap(df_all.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")


# ##  Different Classifiers libraries

# In[ ]:


import xgboost as xgb

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import r2_score,confusion_matrix, mean_squared_error,accuracy_score, f1_score,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore", category=FutureWarning)


# ## **Random Forest, Cross Validation and Confusion Matrix**

# In[ ]:


rf = RandomForestClassifier() 
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(round(accuracy_score(y_test, rf_pred),2)*100)


# In[ ]:


RForest=RandomForestClassifier(n_estimators=20)
RForest.fit(X_train,np.ravel(y_train,order='C'))
RFPred=RForest.predict(X_test)
#rint("Random Forest Accuracy : " ,accuracy_score(y_test,RFPred))

print("Cross Valudate Score : ", cross_val_score(RForest,X_test,y_test.values.ravel()))
print("Confisuon matrix :", confusion_matrix(y_test,RFPred.ravel()))
print(classification_report(y_test,RFPred))


# ## Bayes Model

# In[ ]:


BayesModel=GaussianNB()
BayesModel.fit(X_train,y_train.values.ravel())
bayesPred=BayesModel.predict(X_test)
print("Bayes Model Accuracy :", accuracy_score(y_test,bayesPred.ravel()))
print("Cross val score :", cross_val_score(BayesModel,X_train,y_train.values.ravel(),cv=10,n_jobs=2,scoring='accuracy').mean())


# ## K-Neighbors Classifier

# In[ ]:


#grid={'n_neighbors': np.arange(1,20,1)}
KnnClass=KNeighborsClassifier(n_neighbors=9)
#KnnCV=GridSearchCV(KnnClas, grid, cv=10)
KnnClass.fit(X_train,np.ravel(y_train,order='C'))
#print("Best Parameters : {}\nBest Score {} ".format(KnnCV.best_params_, KnnCV.best_score_) )


# ## Xg Boost Classifier

# In[ ]:



Xgboost=XGBClassifier(learning_rate=0.1,max_depth=4, n_estimators=100,verbosity=1)
Xgboost.fit(X_train,np.ravel(y_train,order='C'))
y_pred=Xgboost.predict(X_test)
print("Test accuracy with XGBoos: ",accuracy_score(y_test,y_pred))


# Logistic Regression

# ## SVM

# In[ ]:


grid={'C':[0.0001,0.001,0.01,1] ,'gamma':['auto','scale'],'kernel':['rbf','linear','sigmoid'] , 'max_iter':[10,100]}
SVCModel=SVC(probability=True)
SVCGCV=GridSearchCV(SVCModel,grid,cv=10)
SVCGCV.fit(X_train,np.ravel(y_train, order='C'))
print("Best Params {} and best score {}".format(SVCGCV.best_params_, SVCGCV.best_score_))
#print(SVCModel.score(x_test,np.ravel(y_test, order='C')))


# In[ ]:


fig, ax_Array = plt.subplots(nrows = 1,  figsize = (8,6))

# bayes roc
probs = BayesModel.predict_proba(X_test)
preds = probs[:,1]
fprbayes, tprxbayes, thresholdbayes = metrics.roc_curve(y_test, preds)
roc_aucbayes = metrics.auc(fprbayes, tprxbayes)


# KNN roc
probs=KnnClass.predict_proba(X_test)
predKnn=probs[:,1]
fprKnn, tprKnn, thresholdKnn=metrics.roc_curve(y_test,predKnn)
roc_aucknn=metrics.auc(fprKnn,tprKnn)


# Random Forest 
probs=RForest.predict_proba(X_test)
pred_RForest=probs[:,1]
fprRF,tprRF,thresholfRF= metrics.roc_curve(y_test,pred_RForest)
roc_aucRF=metrics.auc(fprRF,tprRF)

 # SVM model roc
 
prob=SVCGCV.predict_proba(X_test)
pred_Svm=prob[:,1]
fprSvm,tprsvm,tresholdSvm=metrics.roc_curve(y_test,pred_Svm)
roc_aucSvm=metrics.auc(fprSvm,tprsvm) 




ax_Array.plot(fprSvm,tprsvm, 'b', label='SMV Auc %0.2f' %roc_aucSvm, color="green")
ax_Array.plot(fprRF,tprRF,'b', label="RF Auc %0.2f"%roc_aucRF, color="blue")
ax_Array.plot(fprKnn,tprKnn,'b', label="Knn Auc %0.2f" %roc_aucknn, color="red")
ax_Array.plot(fprbayes,tprxbayes,'b', label='Bayes Auc %0.2f' % roc_aucbayes, color="black")
ax_Array.set_title('Receiver Operating Characteristic LR ',fontsize=10)
ax_Array.set_ylabel('True Positive Rate',fontsize=20)
ax_Array.set_xlabel('False Positive Rate',fontsize=15)
ax_Array.legend(loc = 'lower right', prop={'size': 10})



plt.subplots_adjust(wspace=1)


# ## Mean Squared Error

# In[ ]:



from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1)
trees, train_loss, test_loss = [], [], []
for iter in range(20):
    rf.fit(X_train, y_train)
    y_train_predicted = rf.predict(X_train)
    y_test_predicted = rf.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_predicted)
    mse_test = mean_squared_error(y_test, y_test_predicted)
    #print("Iteration: {} Train mse: {} Test mse: {}".format(iter, mse_train, mse_test))
    trees += [rf.n_estimators]
    train_loss += [mse_train]
    test_loss += [mse_test]
    rf.n_estimators += 1
plt.figure(figsize=(8,6))  
plt.plot(trees, train_loss, color="blue", label="MSE on Train data")
plt.plot(trees, test_loss, color="red", label="MSE on Test data")
plt.xlabel("# of trees")
plt.ylabel("Mean Squared Error");
plt.legend()


# ## **Predictions**

# In[ ]:


#rf.fit(X_train, y_train)
#rf_pred = rf.predict(X_test)
predict = rf.predict(X_test)
predict


# In[ ]:


real_full=df4['deposit_new']
real=real_full[:1000]

pred = rf.predict(X_test)

df4_1=pd.DataFrame({'real': real, 'prediction':pred[:1000]})


# In[ ]:


df4.head()

