#!/usr/bin/env python
# coding: utf-8

# Attribute Information:
# 
# The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds. The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.
# 
# We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time. So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}. 
# 
# The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178 
# 
# y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}: 
# 
# 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open 
# 
# 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed 
# 
# 3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area 
# 
# 2 - They recorder the EEG from the area where the tumor was located 
# 
# 1 - Recording of seizure activity 
# 
# All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure. Our motivation for creating this version of the data was to simplify access to the data via the creation of a .csv version of it. Although there are 5 classes most authors have done binary classification, namely class 1 (Epileptic seizure) against the rest.
# 
# 

# ##http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition 
# For dataset : UCI link

# 

# In[ ]:


#importing numpy and pandas
import numpy as np
import pandas as pd

#importing for visualization
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

r_s=123456


# In[ ]:


#importing dataset
df = pd.read_csv('../input/data.csv')


# # Analysis

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


#Structural analysis
df.info()


# In[ ]:


#statistical analysis
df.describe()


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.corr()


# From the analysis we know that the data
# RangeIndex: 11500 entries
# dtypes: int64(179), object(1)
# NA values : zero
# Column 'Unnamed: 0' is object

# # Preprocessing 

# In[ ]:


#Dropping the 'Unnamed: 0' column
df.drop('Unnamed: 0',inplace=True,axis=1)


# In[ ]:


df.head()


# In[ ]:


#target variable
#df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)
y=df['y']


# In[ ]:


y = y.apply(lambda x: 1 if x == 1 else 0)
y.unique()


# In[ ]:


#dropping the y column
X=df.drop('y',axis=1)
X.head()


# In[ ]:


#SCALING
from sklearn.preprocessing import StandardScaler
scale=StandardScaler().fit(X)
x_scaled=scale.transform(X)
x_scaled


# # Dimensionality Reduction

# As the data has the High number of variables 
# 
# Using principal component analysis we are reducing the dimension of the data

# In[ ]:


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  #two component
principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[ ]:


principalDf.head()


# In[ ]:


#Concating Principal components and y varaible
PCAdf= pd.concat([principalDf, y], axis = 1)
PCAdf.head()


# In[ ]:


PCAdf.shape


# # T-SNE

# T-sne is an another dimensionality reduction techinique 
# 
# This algorithm reduces the variables into two components

# for more information on this algorithm
# https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

# In[ ]:


from sklearn.manifold import TSNE

TNSEdf = TSNE(random_state=r_s).fit_transform(x_scaled)
TNSEdf.shape


# In[ ]:


tnsedf = pd.DataFrame(TNSEdf)


# In[ ]:


tnsedf['y']=y


# In[ ]:


tnsedf.head()


# # K-MEANS

# We use k-Means clustering algorithm for clustering  pca and T-sne components

# In[ ]:


#K-Means
from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=5)

# Centroid values
#centroids = kmeans.cluster_centers_


# # Implementing kmeans in pca

# In[ ]:


#implementing kmeans in pcadf
kmeansPCA = kmeans.fit(principalDf)
#Getting the cluster labels
labels = kmeans.predict(principalDf)


# In[ ]:


PK_df= PCAdf


# In[ ]:


PK_df['kclust']=labels


# In[ ]:


PK_df['kclust'].unique()


# # Implementing kmeans in T-SNE
# 

# In[ ]:


#K-Means
from sklearn.cluster import KMeans
# Number of clusters
kmeans = KMeans(n_clusters=5)
#implementing kmeans in tnsedf
kmeanstnse = kmeans.fit(TNSEdf)
#Getting the cluster labels
k_labels = kmeans.predict(TNSEdf)


# In[ ]:


TK_df = tnsedf


# In[ ]:


TK_df['cluster']=k_labels


# In[ ]:


TK_df.info()


# In[ ]:


TK_df['cluster'].unique()


# So, now we have four dataframes,
# 
# 1.PCAdf    => PCA only.
# 
# 2.tnsedf   => T-SNE only.
# 
# 3.PK_df    => PCA + KMeans.
# 
# 4.TK_df    => T-SNE + Kmeans
# 
# but for simplicity only two dataframes are used to compare the accuracies

# # Splitting the data into train and test

# In[ ]:


#splitting into x and y varialbles
'''
p_x=PCAdf.loc[:,['principal component 1','principal component 2']]
p_y=PCAdf ['y']

t_x=tnsedf.loc[:,[0,1]]
t_y=tnsedf['y']
'''

pk_x=PK_df.loc[:,['principal component 1','principal component 2','kclust']]
pk_y=PK_df['y']

tk_x=TK_df.loc[:,[0,1,'cluster']]
tk_y=TK_df['y']


# In[ ]:


from sklearn.model_selection import train_test_split
'''
#only PCA
px_train, px_test, py_train, py_test = train_test_split(p_x,p_y,
                                                    test_size = 0.2, 
                                                    random_state = 101)'''

#PCA +k-means
pkx_train, pkx_test, pky_train, pky_test = train_test_split(pk_x,pk_y,
                                                            test_size = 0.2, 
                                                            random_state = 101)

'''
#T-sne only
tx_train, tx_test, ty_train, ty_test = train_test_split(t_x,t_y,
                                                        test_size = 0.2, 
                                                        random_state = 101)'''

#T-SNE + k-means
tkx_train, tkx_test, tky_train, tky_test = train_test_split(tk_x,tk_y,
                                                            test_size = 0.2, 
                                                            random_state = 101)


# # Metrics for Classification

# In[ ]:


#Confusion matrix and accuraccy score
from sklearn.metrics import confusion_matrix, accuracy_score
#Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# There are several metrics can be used to tell how good the model performance is.(EX: ROC,AUC,SENSIVITY,MISCLASSIFICATION ERROR,R^2,ADJ R^2,ETC.,.)
# 
# But here we take only confusion matrix and Accuracy score.
# 

# # Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 

logmodel.fit(pkx_train,pky_train)
pklogpred = logmodel.predict(pkx_test)


logmodel.fit(tkx_train,tky_train)
tklogpred = logmodel.predict(tkx_test)


# In[ ]:


print(confusion_matrix(pky_test, pklogpred))
print(round(accuracy_score(pky_test, pklogpred),2)*100)


# In[ ]:


print(confusion_matrix(tky_test, tklogpred))
print(round(accuracy_score(tky_test, tklogpred),2)*100)


# log(PCA + K-Means) gives the higher accuracy of 90.0

# In[ ]:


#cross validation
LOGCV = (cross_val_score(logmodel, tkx_train, tky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(pkx_train, pky_train)
pknnpred = knn.predict(pkx_test)

knn.fit(tkx_train,tky_train)
tknnpred = knn.predict(tkx_test)


# In[ ]:


print(confusion_matrix(pky_test, pknnpred))
print(round(accuracy_score(pky_test, pknnpred),2)*100)


# In[ ]:


print(confusion_matrix(tky_test, tknnpred))
print(round(accuracy_score(tky_test, tknnpred),2)*100)


# KNN(T-sne + k-means) gives the higher accuracy of 94.0

# In[ ]:


KNNCV = (cross_val_score(knn, tkx_train, tky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# # SVM

# In[ ]:


from sklearn.svm import SVC
svc= SVC(kernel = 'sigmoid')
#There are various kernels linear,rbf
svc.fit(pkx_train, pky_train)
pspred = svc.predict(pkx_test)

svc.fit(tkx_train, tky_train)
tspred = svc.predict(tkx_test)


# In[ ]:



print(confusion_matrix(pky_test, pspred))
print(round(accuracy_score(pky_test, pspred),2)*100)


# In[ ]:


print(confusion_matrix(tky_test, tspred))
print(round(accuracy_score(tky_test, tspred),2)*100)


# SVM(PCA + k-means) gives the accuracy of 72.0

# In[ ]:


SVCCV = (cross_val_score(svc, pkx_train, pky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# # Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini

rfc.fit(pkx_train, pky_train)
prpred = rfc.predict(pkx_test)

rfc.fit(tkx_train, tky_train)
trpred = rfc.predict(tkx_test)


# In[ ]:


print(confusion_matrix(pky_test, prpred))
print(round(accuracy_score(pky_test, prpred),2)*100)


# In[ ]:


print(confusion_matrix(tky_test, trpred))
print(round(accuracy_score(tky_test, trpred),2)*100)


# RF(T-sne + k-Means) gives the higher accuracy of  96.0

# In[ ]:


#Cross validation
RFCCV = (cross_val_score(rfc, tkx_train, tky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()

gaussiannb.fit(pkx_train, pky_train)
pgpred = gaussiannb.predict(pkx_test)

gaussiannb.fit(tkx_train, tky_train)
tgpred = gaussiannb.predict(tkx_test)


# In[ ]:


print(confusion_matrix(pky_test, pgpred))
print(round(accuracy_score(pky_test, pgpred),2)*100)


# In[ ]:


print(confusion_matrix(tky_test, tgpred))
print(round(accuracy_score(tky_test, tgpred),2)*100)


# NB(PCA + k-means) gives as higher accuracy of 94.0

# In[ ]:


NBCV = (cross_val_score(gaussiannb, pkx_train, pky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# # XGboost

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier()

xgb.fit(pkx_train, pky_train)
pxpred = xgb.predict(pkx_test)

xgb.fit(tkx_train, tky_train)
txpred = xgb.predict(tkx_test)


# In[ ]:


print(confusion_matrix(pky_test, pxpred))
print(round(accuracy_score(pky_test, pxpred),2)*100)


# In[ ]:


print(confusion_matrix(tky_test, txpred))
print(round(accuracy_score(tky_test, txpred),2)*100)


# XG(T-sne + k-means) gives as higher accuracy of 95.0

# In[ ]:


XGCV = (cross_val_score(xgb, pkx_train, pky_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())


# # Performance comparison

# In[ ]:


models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'XGBoost'],
                'Score':  [RFCCV, SVCCV, KNNCV, LOGCV, NBCV, XGCV]})

models.sort_values(by='Score', ascending=False)


# From the above cross validation scores we get that random forest gives the high performance for this dataset.

# This is only the conlusion for the models built here , still the performance can be increased by adjusting the parameters.
