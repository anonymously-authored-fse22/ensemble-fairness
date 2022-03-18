#!/usr/bin/env python
# coding: utf-8

# # For learning Data Visualizaiton and NLP do check following notebooks
# # [Data Visualization](https://www.kaggle.com/vanshjatana/data-visualization)
#  # [NLP](https://www.kaggle.com/vanshjatana/text-classification)

# # Table of Content
# 1. Machine Learning and Types
# 2. Application of Machine Learning
# 3. Steps of Machine Learning
# 4. Factors help to choose algorithm
# 5. Algorithm
#          Linear Regression
#          TheilSenRegressor
#          RANSAC Regressor
#          HuberRegressor
#          Logistic Regression
#          GaussianProcessClassifier
#          Support Vector Machine
#          Nu-Support Vector Classification
#          Naive Bayes Algorithm
#          KNN
#          Perceptron
#          Random Forest
#          Decision Tree
#          Extra Tree
#          AdaBoost Classifier
#          PassiveAggressiveClassifier
#          Bagging Classifier
#          Gradient Boosting
#                  Light GBM
#                  XGBoost
#                  Catboost
#                  Stochastic Gradient Descent
#          Lasso
#          RidgeC lassifier CV
#          Kernel Ridge Regression
#          Bayesian Ridge
#          Elastic Net Regression
#          LDA
#          K-Means Algorithm
#          CNN
#          LSTM
#          PCA
#          Apriori
#          Prophet
#          ARIMA
# 6. Evaluate Algorithms
#                 
#    
# 

# # Machine Learning

# **Machine Learning is the science of getting computers to learn and act like humans do, and improve their learning over time in autonomous fashion, by feeding them data and information in the form of observations and real-world interactions.
# There are many algorithm for getting machines to learn, from using basic decision trees to clustering to layers of artificial neural networks depending on what task you’re trying to accomplish and the type and amount of data that you have available.  
# **

# **There are three types of machine learning** 
# 1. Supervised Machine Learning 
# 2. Unsupervised Machine Learning 
# 3. Reinforcement Machine Learning 

# #  Supervised Machine Learning 
# 
# **It is a type of learning in which both input and desired output data are provided. Input and output data are labeled for classification to provide a learning basis for future data processing.This algorithm consist of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data.   
# **

# # Unsupervised Machine Learning
# 
# **Unsupervised learning is the training of an algorithm using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance.The main idea behind unsupervised learning is to expose the machines to large volumes of varied data and allow it to learn and infer from the data. However, the machines must first be programmed to learn from data. **
# 
# ** Unsupervised learning problems can be further grouped into clustering and association problems.  
# **
# 1. Clustering: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behaviour. 
# 2. Association: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y. 
# 
# 
# 

# # Reinforcement Machine Learning 
# **Reinforcement Learning is a type of Machine Learning which allows machines to automatically determine the ideal behaviour within a specific context, in order to maximize its performance. Simple reward feedback is required for the agent to learn its behaviour; this is known as the reinforcement signal.It differs from standard supervised learning, in that correct input/output pairs need not be presented, and sub-optimal actions need not be explicitly corrected. Instead the focus is on performance, which involves finding a balance between exploration of uncharted territory and exploitation of current knowledge  
# **
# 

# # Application of Supervised Machine Learning 
# 1. Bioinformatics 
# 2. Quantitative structure 
# 3. Database marketing 
# 4. Handwriting recognition 
# 5. Information retrieval 
# 6. Learning to rank 
# 7. Information extraction 
# 8. Object recognition in computer vision 
# 9. Optical character recognition 
# 10. Spam detection 
# 11. Pattern recognition 
# 
# 

# # Application of Unsupervised Machine Learning 
# 1. Human Behaviour Analysis 
# 2. Social Network Analysis to define groups of friends. 
# 3. Market Segmentation of companies by location, industry, vertical. 
# 4. Organizing computing clusters based on similar event patterns and processes. 
# 

# # Application of Reinforcement Machine Learning 
# 1. Resources management in computer clusters 
# 2. Traffic Light Control 
# 3. Robotics 
# 4. Web System Configuration 
# 5. Personalized Recommendations 
# 6. Deep Learning 
# 

# # We can apply machine learning model by following six steps:-
# 1. Problem Definition 
# 2. Analyse Data 
# 3. Prepare Data 
# 4. Evaluate Algorithm 
# 5. Improve Results 
# 6. Present Results 
# 

# # Factors help to choose algorithm 
# 1. Type of algorithm 
# 2. Parametrization 
# 3. Memory size 
# 4. Overfitting tendency 
# 5. Time of learning 
# 6. Time of predicting

# # Linear Regression 
# **It is a basic and commonly used type of predictive analysis. These regression estimates are used to explain the relationship between one dependent variable and one or more independent variables. 
# Y = a + bX where **
# * Y – Dependent Variable 
# * a – intercept 
# * X – Independent variable 
# * b – Slope 
# 
# **Example: University GPA' = (0.675)(High School GPA) + 1.097**

# **Library and Data **

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
train = pd.read_csv("../input/random-linear-regression/train.csv") 
test = pd.read_csv("../input/random-linear-regression/test.csv") 
train = train.dropna()
test = test.dropna()
train.head()


# **Model with plots and accuracy**

# In[ ]:


X_train = np.array(train.iloc[:, :-1].values)
y_train = np.array(train.iloc[:, 1].values)
X_test = np.array(test.iloc[:, :-1].values)
y_test = np.array(test.iloc[:, 1].values)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

plt.plot(X_train, model.predict(X_train), color='green')
plt.show()
print(accuracy)


# # TheilSen Regressor

# In[ ]:


from sklearn.linear_model import  TheilSenRegressor
model = TheilSenRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)


# # RANSAC Regressor

# In[ ]:


from sklearn.linear_model import  RANSACRegressor
model = RANSACRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)


# # Huber Regressor

# In[ ]:


from sklearn.linear_model import  HuberRegressor
model = HuberRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)


# In[ ]:





# # Logistic Regression 
# **It’s a classification algorithm, that is used where the response variable is categorical. The idea of Logistic Regression is to find a relationship between features and probability of particular outcome.**   
# * odds= p(x)/(1-p(x)) = probability of event occurrence / probability of not event occurrence 
# 
# **Example- When we have to predict if a student passes or fails in an exam when the number of hours spent studying is given as a feature, the response variable has two values, pass and fail. 
# **

# **Libraries and data**

# In[ ]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from statistics import mode


train = pd.read_csv("../input/titanic/train.csv")
test  = pd.read_csv('../input/titanic/test.csv')
train.head()


# In[ ]:


ports = pd.get_dummies(train.Embarked , prefix='Embarked')
train = train.join(ports)
train.drop(['Embarked'], axis=1, inplace=True)
train.Sex = train.Sex.map({'male':0, 'female':1})
y = train.Survived.copy()
X = train.drop(['Survived'], axis=1) 
X.drop(['Cabin'], axis=1, inplace=True) 
X.drop(['Ticket'], axis=1, inplace=True) 
X.drop(['Name'], axis=1, inplace=True) 
X.drop(['PassengerId'], axis=1, inplace=True)
X.Age.fillna(X.Age.median(), inplace=True) 


# **Model and Accuracy**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)


# **Confusion Matrix**

# In[ ]:


print(confusion_matrix(y_test,y_pred))


# **Report**

# In[ ]:


print(classification_report(y_test,y_pred))


# # Gaussian Process Classifier

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
from sklearn.gaussian_process import GaussianProcessClassifier
model = GaussianProcessClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)


# In[ ]:


print(classification_report(y_test,y_pred))


# # Support Vector Machine 
# **Support Vector Machines are perhaps one of the most popular and talked about machine learning algorithms.It is primarily a classier method that performs classification tasks by constructing hyperplanes in a multidimensional space that separates cases of different class labels. SVM supports both regression and classification tasks and can handle multiple continuous and categorical variables 
# **
# 
# **Example: One class is linearly separable from the others like if we only had two features like Height and Hair length of an individual, we’d first plot these two variables in two dimensional space where each point has two co-ordinates **

# **Libraries and Data**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
data_svm = pd.read_csv("../input/svm-classification/UniversalBank.csv")
data_svm.head()


# **Model and Accuracy**

# In[ ]:


X = data_svm.iloc[:,1:13].values
y = data_svm.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


# In[ ]:


print(classification_report(y_test,y_pred))


# # Nu Support Vector Classification

# **Library and Data**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import NuSVC
nu_svm = pd.read_csv("../input/svm-classification/UniversalBank.csv")
nu_svm.head()


# **Model and Accuracy**

# In[ ]:


X = nu_svm.iloc[:,1:13].values
y = nu_svm.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
classifier = NuSVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


# In[ ]:


print(classification_report(y_test,y_pred))


# # Naive Bayes Algorithm 
# **A naive Bayes classifier is not a single algorithm, but a family of machine learning algorithms which use probability theory to classify data with an assumption of independence between predictors It is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods    
# **
# 
# **Example: Emails are given and we have to find the spam emails from that.A spam filter looks at email messages for certain key words and puts them in a spam folder if they match.**

# **Libraries and Data**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/classification-suv-dataset/Social_Network_Ads.csv')
data_nb = data
data_nb.head()


# **Model and Accuracy**

# **Gaussian NB**

# In[ ]:


X = data_nb.iloc[:, [2,3]].values
y = data_nb.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# In[ ]:


print(classification_report(y_test,y_pred))


# **BernoulliNB**

# In[ ]:


X = data_nb.iloc[:, [2,3]].values
y = data_nb.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=BernoulliNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# In[ ]:


print(classification_report(y_test,y_pred))


# # KNN 
# **KNN does not learn any model. and stores the entire training data set which it uses as its representation.The output can be calculated as the class with the highest frequency from the K-most similar instances. Each instance in essence votes for their class and the class with the most votes is taken as the prediction 
# **
# 
# **Example: Should the bank give a loan to an individual? Would an individual default on his or her loan? Is that person closer in characteristics to people who defaulted or did not default on their loans? **
# 

# **Libraries and Data**

# **As Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
knn = pd.read_csv("../input/iris/Iris.csv")
knn.head()


# **Model and Accuracy**

# In[ ]:


X = knn.iloc[:, [1,2,3,4]].values
y = knn.iloc[:, 5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# In[ ]:


print(classification_report(y_test,y_pred))


# **As Regression**

# **Library and Data**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
train = pd.read_csv("../input/random-linear-regression/train.csv") 
test = pd.read_csv("../input/random-linear-regression/test.csv") 
train = train.dropna()
test = test.dropna()
X_train = np.array(train.iloc[:, :-1].values)
y_train = np.array(train.iloc[:, 1].values)
X_test = np.array(test.iloc[:, :-1].values)
y_test = np.array(test.iloc[:, 1].values)


# **Model and Accuracy**

# In[ ]:


model = KNeighborsRegressor(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)


# # Perceptron 

# ** It is single layer neural network and used for classification **

# In[ ]:


from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
p = pd.read_csv("../input/iris/Iris.csv")
p.head()


# In[ ]:


X = p.iloc[:, [1,2,3,4]].values
y = p.iloc[:, 5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=Perceptron()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# In[ ]:


print(classification_report(y_test,y_pred))


# # Random Forest 
# **Random forest is collection of tress(forest) and it builds multiple decision trees and merges them together to get a more accurate and stable prediction.It can be used for both classification and regression problems.**
# 
# **Example: Suppose we have a bowl of 100 unique numbers from 0 to 99. We want to select a random sample of numbers from the bowl. If we put the number back in the bowl, it may be selected more than once. 
# **

# **Libraries and Data**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
rf.head()


# **Model and Accuracy**

# In[ ]:


X = rf.drop('class', axis=1)
y = rf['class']
X = pd.get_dummies(X)
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:





# # Decision Tree
# **Decision tree algorithm is classification algorithm under supervised machine learning and it is simple to understand and use in data.The idea of Decision tree is to split the big data(root) into smaller(leaves)**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = data
dt.head()


# In[ ]:


X = dt.iloc[:, [2,3]].values
y = dt.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# # Extra Tree

# **Library and  Data**

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
et = data
et.head()


# **Model and Accuracy**

# In[ ]:


X = et.iloc[:, [2,3]].values
y = et.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=ExtraTreesClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# # AdaBoost Classifier

# **Library and Data**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ac = data
ac.head()


# **Model and Accutacy**

# In[ ]:


X = ac.iloc[:, [2,3]].values
y = ac.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=AdaBoostClassifier(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# # Passive Aggressive Classifier

# **Library and Data**

# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier
pac = data
pac.head()


# **Model and Accuracy**

# In[ ]:


X = pac.iloc[:, [2,3]].values
y = pac.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=PassiveAggressiveClassifier(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# # Bagging Classifier

# **Library and Data**

# In[ ]:


from sklearn.ensemble import BaggingClassifier
bc = data
bc.head()


# **Model and Accuracy**

# In[ ]:


X = bc.iloc[:, [2,3]].values
y = bc.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=BaggingClassifier(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# # Gradient Boosting
# **Gradient boosting is an alogithm under supervised machine learning, boosting means converting weak into strong. In this new tree is boosted over the previous tree**

# **Libraries and Data**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb = data
gb.head()


# **Model and Accuracy**

# In[ ]:


X = gb.iloc[:, [2,3]].values
y = gb.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
pred = gbk.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)


# # Light GBM

# **LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:**
# 
# 1. Faster training speed and higher efficiency.
# 2. Lower memory usage.
# 3. Better accuracy.
# 4. Support of parallel and GPU learning.
# 5. Capable of handling large-scale data.

# **Library and Data**

# In[ ]:


import lightgbm as lgbm
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import preprocessing


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
data = pd.concat([train, test], sort=False)
data = data.reset_index(drop=True)
data.head()


# **Preprocessing**

# In[ ]:


nans=pd.isnull(data).sum()

data['MSZoning']  = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data["BsmtFinSF1"]  = data["BsmtFinSF1"].fillna(0)
data["BsmtFinSF2"]  = data["BsmtFinSF2"].fillna(0)
data["BsmtUnfSF"]   = data["BsmtUnfSF"].fillna(0)
data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)
data["BsmtFullBath"] = data["BsmtFullBath"].fillna(0)
data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(0)
data["BsmtQual"] = data["BsmtQual"].fillna("None")
data["BsmtCond"] = data["BsmtCond"].fillna("None")
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")

data['KitchenQual']  = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data["Functional"]   = data["Functional"].fillna("Typ")
data["FireplaceQu"]  = data["FireplaceQu"].fillna("None")

data["GarageType"]   = data["GarageType"].fillna("None")
data["GarageYrBlt"]  = data["GarageYrBlt"].fillna(0)
data["GarageFinish"] = data["GarageFinish"].fillna("None")
data["GarageCars"] = data["GarageCars"].fillna(0)
data["GarageArea"] = data["GarageArea"].fillna(0)
data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")

data["PoolQC"] = data["PoolQC"].fillna("None")
data["Fence"]  = data["Fence"].fillna("None")
data["MiscFeature"] = data["MiscFeature"].fillna("None")
data['SaleType']    = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['LotFrontage'].interpolate(method='linear',inplace=True)
data["Electrical"]  = data.groupby("YearBuilt")['Electrical'].transform(lambda x: x.fillna(x.mode()[0]))
data["Alley"] = data["Alley"].fillna("None")

data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
nans=pd.isnull(data).sum()
nans[nans>0]


# In[ ]:


_list = []
for col in data.columns:
    if type(data[col][0]) == type('str'): 
        _list.append(col)

le = preprocessing.LabelEncoder()
for li in _list:
    le.fit(list(set(data[li])))
    data[li] = le.transform(data[li])

train, test = data[:len(train)], data[len(train):]

X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']

test = test.drop(columns=['SalePrice', 'Id'])


# **Model and Accuracy**

# In[ ]:


kfold = KFold(n_splits=5, random_state = 2020, shuffle = True)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X, y)
r2_score(model_lgb.predict(X), y)


# # **XGBoost**

# **XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks.It is a perfect combination of software and hardware optimization techniques to yield superior results using less computing resources in the shortest amount of time.**

# **Library and Data**

# In[ ]:


import xgboost as xgb
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X, y)
r2_score(model_xgb.predict(X), y)


# In[ ]:





# # Catboost

# **Catboost is a type of gradient boosting algorithms which can  automatically deal with categorical variables without showing the type conversion error, which helps you to focus on tuning your model better rather than sorting out trivial errors.Make sure you handle missing data well before you proceed with the implementation.
# **

# **Library and Data**

# In[ ]:


from catboost import CatBoostRegressor
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
cb_model.fit(X, y)
r2_score(cb_model.predict(X), y)


# # Stochastic Gradient Descent

# **Stochastic means random , so in Stochastic Gradient Descent dataset sample is choosedn random instead of the whole dataset.hough, using the whole dataset is really useful for getting to the minima in a less noisy or less random manner, but the problem arises when our datasets get really huge and for that SGD come in action**

# **Library and Data**

# In[ ]:


from sklearn.linear_model import SGDRegressor
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


SGD = SGDRegressor(max_iter = 100)
SGD.fit(X, y)
r2_score(SGD.predict(X), y)


# # Lasso

# **In statistics and machine learning, lasso (least absolute shrinkage and selection operator; also Lasso or LASSO) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces. Though originally defined for least squares, lasso regularization is easily extended to a wide variety of statistical models including generalized linear models, generalized estimating equations, proportional hazards models, and M-estimators, in a straightforward fashion**

# **Library and Data**

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(X, y)
r2_score(lasso.predict(X), y)


# # Ridge Classifier CV

# **Library and Data**

# In[ ]:


from sklearn.linear_model import RidgeClassifierCV
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and  Accuracy**

# In[ ]:


rcc = RidgeClassifierCV()
rcc.fit(X, y)
r2_score(rcc.predict(X), y)


# # Kernel Ridge Regression

# **KRR combine Ridge regression and classification with the kernel trick.It is similar to Support vector Regression but relatively very fast.This is suitable for smaller dataset (less than 100 samples)**

# **Library and Data**

# In[ ]:


from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and  Accuracy**

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR.fit(X, y)
r2_score(KRR.predict(X), y)


# # BayesianRidge

# ** Bayesian regression, is a regression model defined in probabilistic terms, with explicit priors on the parameters. The choice of priors can have the regularizing effect.Bayesian approach is a general way of defining and estimating statistical models that can be applied to different models.**

# **Library and Data**

# In[ ]:


from sklearn.linear_model  import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


BR = BayesianRidge()
BR.fit(X, y)
r2_score(BR.predict(X), y)


# # Elastic Net Regression 
# 

# **Elastic net is a hybrid of ridge regression and lasso regularization.It combines feature elimination from Lasso and feature coefficient reduction from the Ridge model to improve your model's predictions.**

# **Library and Data**

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(X, y)
r2_score(ENet.predict(X), y)


# # **LDA**

# **A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.Itis  used in statistics, pattern recognition, and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.**

# **Library and Data**

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = data
lda.head()


# **Model and Accuracy**

# In[ ]:


X = lda.iloc[:, [2,3]].values
y = lda.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Model=LinearDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print('accuracy is ',accuracy_score(y_pred,y_test))


# # K-Means Algorithm 
# K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data and the goal of this algorithm is to find groups in the data 
# 
# **Steps to use this algorithm:-**
# * 1-Clusters the data into k groups where k is predefined. 
# * 2-Select k points at random as cluster centers. 
# * 3-Assign objects to their closest cluster center according to the Euclidean distance function. 
# * 4-Calculate the centroid or mean of all objects in each cluster. 
# 
# **Examples: Behavioral segmentation like segment by purchase history or by activities on application, website, or platform Separate valid activity groups from bots  **
# 

# **Libraries and Data**

# In[ ]:


from sklearn.cluster import KMeans
km = pd.read_csv("../input/k-mean/km.csv")
km.head()


# **Checking for number of clusters**

# In[ ]:


K_clusters = range(1,8)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = km[['latitude']]
X_axis = km[['longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.show()


# **Fitting Model**

# In[ ]:


kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(km[km.columns[1:3]])
km['cluster_label'] = kmeans.fit_predict(km[km.columns[1:3]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(km[km.columns[1:3]])
km.cluster_label.unique()


# **Plotting Clusters**

# In[ ]:


km.plot.scatter(x = 'latitude', y = 'longitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)


# # CNN

# **Library and Data**

# In[ ]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
train_data.head()


# **Preprocessing and Data Split**

# In[ ]:


X = np.array(train_data.drop("label", axis=1)).astype('float32')
y = np.array(train_data['label']).astype('float32')
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.show()

X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_test = np.array(test_data).astype('float32')
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
plt.figure(figsize=(10,10))


# **Model**

# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model1.png')


# **Compiling model**

# In[ ]:


#increse to epochs to 30 for better accuracy
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, batch_size=85, validation_data=(X_val, y_val))


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.show()

print(model.evaluate(X_val, y_val))


# In[ ]:


prediction = model.predict_classes(X_test)
submit = pd.DataFrame(prediction,columns=["Label"])
submit["ImageId"] = pd.Series(range(1,(len(prediction)+1)))
submission = submit[["ImageId","Label"]]
submission.to_csv("submission.csv",index=False)


# # LSTM 

# **LSTM  blocks are part of a recurrent neural network structure. Recurrent neural networks are made to utilize certain types of artificial memory processes that can help these artificial intelligence programs to more effectively imitate human thought.It is  capable of learning order dependence 
# LSTM can be used for machine translation, speech recognition, and more.**

# **Library and Data**

# In[ ]:


import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
lstm = pd.read_csv("../input/nyse/prices.csv")
lstm = lstm[lstm['symbol']=="NFLX"]
lstm['date'] = pd.to_datetime(lstm['date'])
lstm.set_index('date',inplace=True)
lstm = lstm.reset_index()
lstm.head()


# **Preprocessing**

# In[ ]:


data = lstm.filter(['close'])
dataset = data.values 
training_data_len = math.ceil(len(dataset)*.75)  
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
x_train,y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# **Model**

# In[ ]:


model =Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64, return_sequences= False))
model.add(Dense(32))
model.add(Dense(1))
model.summary()
from tensorflow.keras.utils import plot_model 
plot_model(model, to_file='model1.png')


# **Compiling Model**

# In[ ]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train, batch_size=85, epochs=20)


# **Prediction and Accuracy**

# In[ ]:


test_data= scaled_data[training_data_len-60:, :]
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# # Principle Component Analysis

# **It's an important method for dimension reduction.It extracts low dimensional set of features from a high dimensional data set with a motive to capture as much information as possible and to visualise high-dimensional data, it also reduces noise and finally makes other algorithms to work better because we are injecting fewer inputs.**
# * Example: When we have to bring out strong patterns in a data set or to make data easy to explore and visualize

# In[ ]:


from sklearn.datasets import make_blobs
from sklearn import datasets
class PCA:
  def __init__(self, n_components):
    self.n_components = n_components
    self.components = None
    self.mean = None

  def fit(self, X):
    self.mean = np.mean(X, axis=0)
    X = X - self.mean
    cov = np.cov(X.T)

    evalue, evector = np.linalg.eig(cov)

    eigenvectors = evector.T
    idxs = np.argsort(evalue)[::-1]
    
    evalue = evalue[idxs]
    evector = evector[idxs]
    self.components = evector[0:self.n_components]

  def transform(self, X):
    #project data
    X = X - self.mean
    return(np.dot(X, self.components.T))

data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)



x1 = X_projected[:,0]
x2 = X_projected[:,1]

plt.scatter(x1,x2,c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()


# # Apriori

# **It is a categorisation algorithm attempts to operate on database records, particularly transactional records, or records including certain numbers of fields or items.It is mainly used for sorting large amounts of data. Sorting data often occurs because of association rules. **
# * Example: To analyse data for frequent if/then patterns and using the criteria support and confidence to identify the most important relationships. 

# In[ ]:


df = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv',names=['products'],header=None)
data = list(df["products"].apply(lambda x:x.split(',')))
data


# In[ ]:


from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df1 = apriori(df,min_support=0.01,use_colnames=True)
df1.head()


# # Prophet

# 
# Prophet is an extremely easy tool for analysts to produce reliable forecasts

# 1. Prophet only takes data as a dataframe with a ds (datestamp) and y (value we want to forecast) column. So first, let’s convert the dataframe to the appropriate format.
# 1. Create an instance of the Prophet class and then fit our dataframe to it.
# 2. Create a dataframe with the dates for which we want a prediction to be made with make_future_dataframe(). Then specify the number of days to forecast using the periods parameter.
# 3. Call predict to make a prediction and store it in the forecast dataframe. What’s neat here is that you can inspect the dataframe and see the predictions as well as the lower and upper boundaries of the uncertainty interval.
# 

# **Library and Data**

# In[ ]:


import plotly.offline as py
import plotly.express as px
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

pred = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
pred = pred.fillna(0)
predgrp = pred.groupby("Date")[["Confirmed","Recovered","Deaths"]].sum().reset_index()
pred_cnfrm = predgrp.loc[:,["Date","Confirmed"]]
pr_data = pred_cnfrm
pr_data.columns = ['ds','y']
pr_data.head()


# **Model and Forecast**

# In[ ]:


m=Prophet()
m.fit(pr_data)
future=m.make_future_dataframe(periods=15)
forecast=m.predict(future)
forecast


# In[ ]:


fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')


# # Arima

# **Library and Data**

# In[ ]:


import datetime
from statsmodels.tsa.arima_model import ARIMA
ar = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
ar.date=ar.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
ar=ar.groupby(["date_block_num"])["item_cnt_day"].sum()
ar.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ar=ar.reset_index()
ar=ar.loc[:,["index","item_cnt_day"]]
ar.columns = ['confirmed_date','count']
ar.head()


# **Model**

# In[ ]:


model = ARIMA(ar['count'].values, order=(1, 2, 1))
fit_model = model.fit(trend='c', full_output=True, disp=True)
fit_model.summary()


# **Prediction**

# In[ ]:


fit_model.plot_predict()
plt.title('Forecast vs Actual')
pd.DataFrame(fit_model.resid).plot()
forcast = fit_model.forecast(steps=6)
pred_y = forcast[0].tolist()
pred = pd.DataFrame(pred_y)


# # **Evaluate Algorithms** 
# **The evaluation of algorithm consist three following steps:- **
# 1. Test Harness  
# 2. Explore and select algorithms 
# 3. Interpret and report results 
# 
# 

# # If you like this notebook, do hit upvote
# # Thanks
# 
# 
