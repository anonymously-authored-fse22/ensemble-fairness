#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Classification A Machine Learning Perspective: Survey 
# 
# 

# **Abstract:** In this work  the Adults Income Census dataset in Kaggle website is selected as subject for applying diverse classification techniques. This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over 50K dollars a year.
# 247 coding literary works were written on this dataset in Kaggle in different computing languages, in this work we consider only the top notebooks written in python .The task   is to predict whether the particular adult earns more or less than $50000, by finding patterns in the independent variables.
# 
# **Keywords**—Kaggle, Data Scientists, Machine Learning Engineers,Python
# 
# # I. INTRODUCTION
# 
# 
# Human dependence on data insights in society has increased over the past two decades. With the emerging technologies there is a huge demand for machine learning which has its applications in all types of the industries. By Machine learning, automated decisions will be predicted based on sample data inputs.. Problems relating to significant domains of social life, retail sector and public safety have been addressed by using machine learning techniques. These three domains play a very important role in daily human life, bringing machine learning techniques to these domains can bring significant change in the life style of humans. The economic status play an important role in determining the social life of an individual, there is a significant interest in these days from government to standardize these social survey platforms in their country and there is a tremendous scope for machine learning techniques to be implement in these survey to obtain interesting insights on social and economic life of citizens.
# Raw data is like crude oil, by processing we can get desired products. Similarly by preprocessing the data we can draw insights on what factors the target variable depends on. Our dataset contains 14 independent variables and one target variable with 32561 samples. 
# 
# # II. DATA DESCRIPTION
# 
# This dataset is extracted from 1994 Census in such way to to focus on adults, to study their income.This dataset contains 32516 samples and 15 variables in which 14 are independent but Income variable is a target or dependent variable as it depends upon these independent variables.
# 
# income: >50K, <=50K 
# 
# age: age of a person
# 
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
# 
# fnlwgt: The weight given by the census board
# 
# education_num: Categorical variable of the education
# 
# 
# 
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool education-num: continuous
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
# 
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspect, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
# 
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# 
# sex: Female, Male
# 
# capital-gain: gain in capital
# 
# capital-loss: loss in capital
# 
# hours-per-week: chorus the person worked for a week
# 
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
# 
# # III. DATA PREPROCESSING
# 
# Before going into data preprocessing, I would like to mention about the libraries that were imported and reading the dataset. Different notebooks did it in different style, one can import all the libraries in single step like in [2][3][6], or multiple steps whenever they required like in [1][4][5]. All the notebooks read_csv function in pandas to read the data.
# 
# Most popular preprocessing libraries written in python computing language are Pandas, Numpy, Sklearn. 
# 
# Using .info or .describe can help us understand more about the data [1][5].
# 
# ## A. Handling Missing values
# 
# Most of the datasets will have null values, we have to process them before building a model. The ways of treating these  missing values is to drop the samples that have missing values, replace them with mean or median or mode , treat all the missing values as a new class(possible only in few cases).In this dataset null values are in the in the form of ‘?’ so replace it with ‘np.nan’.
# 
# 
# 
# We can also try by merging different categories in a particular categorical column or columns  like in [2][3][6].
# 
# ## B. Label Encoding
# Generally independent variables are of two kinds of datatypes which is Numerical and String, most of the algorithms works better with numerical values so in order to draw insights from  the string variables, we need to encode them .This can be done by the label encoder or onehot encoder from sklearn library like in [4][5][6].
# 
# ## C. Scaling
# The variables might be in different ranges, the huge difference in magnitude of variables might effect the prediction of  the target variable. .MinMaxScaler, StandardScaler,  RobustScaler and Normalizer can help us to bring entire data onto a uniform scaled range of values. 
# 
# ## D. Correlation
# Correlation, is a statistical technique to determines how one variables varies with the other variable, this gives the idea on degree of the relationship of the two variables. It’s a bi-variate analysis measure which describes the association between different variables. In most of the analysis works it is useful to express one variable in terms of  others.
# 
# 
# As we observe there are various data types across the independent variables of dataset. To maintain uniformity the columns [ workclass, education, marital_status, occupation, relationship, race, sex , native_country] were encoded into categorical variables. The target variable ‘income’ is also categorized to form ‘Two’ categories for more than 50k dollars/year or less than 50k dollars year. The dataset is made split into train and test datasets with 70:30 ratio and scaling of independent variables is made using standard scaling techniques to obtain data of zero mean and unit variance.
# 
# # IV.  MODEL BUILDING
# 
# Different Classifaction algorithms are applied on the dataset and different results yie'lded.
# 
# ## A. Logistic Regression:
# Logistic Regression is a Machine Learning algorithm primarily used for the classification; this is a predictive analysis algorithm which is based on probability concept.
# 
# Logistic Regression uses a Sigmoid or logistic function as cost function.
# The hypothesis of logistic regression tends to limit the cost function between 0 and 1. Therefore linear functions fail to represent it as it can have a value greater than 1 or less than 0 which is not possible as per the hypothesis of logistic regression.
# 
# **Logistic Regression hyper parameters:**
# 
# *Solver* in [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
# Regularization (penalty) can sometimes be helpful.
# 
# *Penalty* in [‘none’, ‘l1’, ‘l2’, ‘elasticnet’]
# 
# Note: not all solvers support all regularization terms.
# 
# The *C parameter* controls the penalty strength, which can also be effective.
# C in [100, 10, 1.0, 0.1, 0.01]
# 
# ## B. KNN:
# In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# 
# **KNN hyperparameters:-**
# 
# *n_neighbors* in [1 to 21]
# 
# *metric* in [‘euclidean’, ‘manhattan’, ‘minkowski’]
# 
# *weights* in [‘uniform’, ‘distance’]
# 
# ## C. SVM:
# 
# “Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However,  it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.
# 
# In the SVM classifier, it is easy to have a linear hyper-plane between these two classes. But, another burning question which arises is, should we need to add this feature manually to have a hyper-plane. No, the SVM  algorithm has a technique called the kernel trick. The SVM kernel is a function that takes low dimensional input space and transforms it to a higher dimensional space i.e. it converts not separable problem to separable problem. It is mostly useful in non-linear separation problem. Simply put, it does some extremely complex data transformations, then finds out the process to separate the data based on the labels or outputs you’ve defined.
# 
# **SVM Hyperparameters:-**
# 
# *C parameter*: It handles the tradeoff between the two goals below.
# 
# Increase the distance of decision boundary to classes (or support vectors)
# 
# Maximize the number of points that are correctly classified in the training set
# 
# *Kernel*: Linear, RBF, Poly
# 
# *Gamma*: One of the commonly used kernel functions is radial basis function (RBF). Gamma parameter of RBF controls the distance of influence of a single training point. Low values of gamma indicates a large similarity radius which results in more points being grouped together. For high values of gamma, the points need to be very close to each other in order to be considered in the same group (or class). Therefore, models with very large gamma values tend to overfit.
# 
# As the gamma decreases, the regions separating different classes get more generalized. Very large gamma values result in too specific class regions (overfitting).
# 
# ## D. Decision Tree:
# Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works for both categorical and continuous input and output variables. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables.
# 
# **Decision Tree Hyperparameters**
# 
# *ROOT Node*: It represents entire population or sample and this further gets divided into two or more homogeneous sets.
# 
# *SPLITTING*: It is a process of dividing a node into two or more sub-nodes.
# Decision Node: When a sub-node splits into further sub-nodes, then it is called decision node.
# 
# *Leaf/ Terminal Node*: Nodes do not split is called Leaf or Terminal node.
# Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.
# 
# *Branch / Sub-Tree*: A sub section of entire tree is called branch or sub-tree
# Parent and Child Node: A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.
# 
# ## E. Random Forest:
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# 
# Random forest is like bootstrapping algorithm with Decision tree (CART) model. Say, we have 1000 observation in the complete population with 10 variables. Random forest tries to build multiple CART models with different samples and different initial variables. For instance, it will take a random sample of 100 observation and 5 randomly chosen initial variables to build a CART model. It will repeat the process (say) 10 times and then make a final prediction on each observation. Final prediction is a function of each prediction. This final prediction can simply be the mean of each prediction.
# 
# For a Random Forest Classifier, there are several different hyperparameters that can be adjusted.But the following four parameters are most important
# 
# **Random Forest Hyperparameters:-**
# 
# *n_estimators*: The n_estimators parameter specifies the number of trees in the forest of the model. The default value for this parameter is 10, which means that 10 different decision trees will be constructed in the random forest.
# 
# *max_depth*: The max_depth parameter specifies the maximum depth of each tree. The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.
# 
# *min_samples_split*: The min_samples_split parameter specifies the minimum number of samples required to split an internal leaf node. The default value for this parameter is 2, which means that an internal node must have at least two samples before it can be split to have a more specific classification.
# 
# *min_samples_leaf*: The min_samples_leaf parameter specifies the minimum number of samples required to be at a leaf node. The default value for this parameter is 1, which means that every leaf must have at least 1 sample that it classifies.
# 
# ## F. XGBOOST: 
# Boosting is used to create a collection of predictors. In this technique, learners are learned sequentially with early learners fitting simple models to the data and then analysing data for errors. Consecutive trees (random sample) are fit and at every step, the goal is to improve the accuracy from the prior tree. When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. This process converts weak learners into better performing model.
# 
# The XGBoost library implements the gradient boosting decision tree algorithm.
# 
# Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
# 
# This approach supports both regression and classification predictive modelling problems.
# 
# **XGBoost Hyperparameters:-**
# 
# *max_depth*:The maximum depth of a tree
# min_child_weight:Defines the minimum sum of weights of all observations required in a child.
# 
# *gamma*:A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
# 
# *subsample*:   Denotes the fraction of observations to be randomly chosen samples for each tree.
# 
# *colsample_bytree*:  Denotes the fraction of columns to be randomly chosen  samples for each tree.
# 
# *reg_alpha*:     Regularization parameter
# 
# *reg_lamda*:  Regularization parameter
# 
# *learning rate*:  Parameter in Gradient Descent
# 
# ## G. CATBOOST:
# It yields state-of-the-art results without extensive data training typically required by other machine learning methods, and provides powerful out-of-the-box support for the more descriptive data formats that accompany many business problems.
# 
# “CatBoost” name comes from two words “Category” and “Boosting”.
# 
# It can work with multiple Categories of data, such as audio, text, image including historical data.
# 
# “Boost” comes from gradient boosting machine learning algorithm as this library is based on gradient boosting library. Gradient boosting is a powerful machine learning algorithm that is widely applied to multiple types of business challenges like fraud detection, recommendation items, forecasting and it performs well also. It can also return very good result with relatively less data, unlike DL models that need to learn from a massive amount of data.
# 
# **Advantages of CatBoost Library**
# 
# *Performance*: CatBoost provides state of the art results and it is competitive with any leading machine learning algorithm on the performance front.
# 
# *Handling Categorical features automatically*: We can use CatBoost without any explicit pre-processing to convert categories into numbers. CatBoost converts categorical values into numbers using various statistics on combinations of categorical features and combinations of categorical and numerical features. You can read more about it here.
# 
# *Robust*: It reduces the need for extensive hyper-parameter tuning and lower the chances of overfitting also which leads to more generalized models. Although, CatBoost has multiple parameters to tune and it contains parameters like the number of trees, learning rate, regularization, tree depth, fold size, bagging temperature and others. You can read about all these parameters here.
# 
# *Easy-to-use*: You can use CatBoost from the command line, using an user-friendly API for both Python and R.
# 
# 
# # V TUNING HYPERPARAMETERS
# 
# 
# ## A. Grid Search:- 
#  Grid Search is particularly used for tuning hyperparameters. Now before going into deep lets know what is a hyperparameter.
# 
# A model hyperparameter is a characteristic of a model that is external to the model and whose value cannot be estimated from data. The value of the hyperparameter has to be set before the learning process begins. For example k in k-Nearest Neighbours, the number of hidden layers in Neural Networks.
# 
# In contrast, a parameter is an internal characteristic of the model and its value can be estimated from data. Example, beta coefficients of linear/logistic regression or support vectors in Support Vector Machines.
# 
# Grid-search is used to find the optimal hyperparameters of a model which results in the most ‘accurate’ predictions.
# 
# Trying with different values of hyperparameters and checking the metrics is a long process. Grid Search does the work for us by iterating with different values that we specifically give and results the optimal one from the given.
# 
# *estimator*: estimator object you created
# 
# *params_grid*: the dictionary object that holds the hyperparameters you want to try
# 
# *scoring*: evaluation metric that you want to use, you can simply pass a valid string/ object of evaluation metric
# 
# *cv*: number of cross-validation you have to try for each selected set of hyperparameters
# 
# *verbose*: you can set it to 1 to get the detailed print out while you fit the data to GridSearchCV
# 
# *n_jobs*: number of processes you wish to run in parallel for this task if it -1 it will use all available processors.
# 
# ## B. Random Search:
# Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. It tries random combinations of a range of values. To optimise with random search, the function is evaluated at some number of random configurations in the parameter space.
# 
# The chances of finding the optimal parameter are comparatively higher in random search because of the random search pattern where the model might end up being trained on the optimised parameters without any aliasing. Random search works best for lower dimensional data since the time taken to find the right set is less with less number of iterations. Random search is the best parameter search technique when there are less number of dimensions. In the paper Random Search for Hyper-Parameter Optimization by Bergstra and Bengio, the authors show empirically and theoretically that random search is more efficient for parameter optimization than grid search.
# 
# ## C. Bayesian Optimization
# Bayesian approaches, in contrast to random or grid search, keep track of past evaluation results which they use to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function.
# 
# In the literature, this model is called a “surrogate” for the objective function and is represented as p(y | x). The surrogate is much easier to optimize than the objective function and Bayesian methods work by finding the next set of hyperparameters to evaluate on the actual objective function by selecting hyperparameters that perform best on the surrogate function. In other words:
# 
# 1.Build a surrogate probability model of the objective function
# 
# 2.Find the hyperparameters that perform best on the surrogate
# 
# 3.Apply these hyperparameters to the true objective function
# 
# 4.Update the surrogate model incorporating the new results
# 
# 5.Repeat steps 2–4 until max iterations or time is reached
# 
# The aim of Bayesian reasoning is to become “less wrong” with more data which these approaches do by continually updating the surrogate probability model after each evaluation of the objective function.
# 
# At a high-level, Bayesian optimization methods are efficient because they choose the next hyperparameters in an informed manner. The basic idea is: spend a little more time selecting the next hyperparameters in order to make fewer calls to the objective function. In practice, the time spent selecting the next hyperparameters is inconsequential compared to the time spent in the objective function. By evaluating hyperparameters that appear more promising from past results, Bayesian methods can find better model settings than random search in fewer iterations.
# 
# # VI CURSE OF DIMENSIONALITY
# 
# ## Principal Component Analysis
# 
# When we have too many features to train in the model then there might be a chance of over fitting, so two options
# 
# **Feature Elimination**: We shall drop few features and train the model with features that contribute most towards target variable. This can be done by RFE(Recursive Feature Elimination) or RFECV(RFE with Cross Validation).They can give how much each feature is important , like a list, so we can select the important features. But the only problem is we lose the information whatever the dropped features were contributing to the target variable.  So
# 
# **Feature Extraction**:  PCA is a feature extraction method. In feature extraction, we create “new” independent variables, where each “new” independent variable is a combination of each of the all “old” independent variables. However, we create these new independent variables in a specific way and order these new variables by how well they predict our dependent variable.
# 
# We keep as many of the new independent variables as we want, but we drop the “least important ones.” Because we ordered the new variables by how well they predict our dependent variable, we know which variable is the most important and least important. Because these new independent variables are combinations of our old ones, we’re still keeping the most valuable parts of our old variables, even when we drop one or more of these “new” variables
# 
# **Pros:**
# When we want to reduce the number of features but don’t know what to drop, PCA can help us
# New variables are independent off each other
# 
# **Cons:**
# New features are less interpretable
# 
# # VII   EVALUATION METRICS
# 
# ## A. ROC
# Many machine learning models were performed on this dataset that includes Logistic and Decision Tree algorithms. The metrics which were used to determine the efficiency includes sensitivity or Recall, specificity and accuracy and also using Receiver operating charecteristics curve. The area under curve value reveals that Logistic Regression performed better than Decision Tree. 
# 
# 
# ## B. Confusion Matrix
# Performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.
# 
# 
# It is extremely useful for measuring Recall, Precision, Specificity, Accuracy and most importantly AUC-ROC Curve.
# 
# ## C. Classification Report
# 
# A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report
# 
# The report shows the main classification metrics precision, recall and f1-score on a per-class basis. The metrics are calculated by using true and false positives, true and false negatives. Positive and negative in this case are generic names for the predicted classes. There are four ways to check if the predictions are right or wrong:
# 
# TN / True Negative: when a case was negative and predicted negative
# 
# TP / True Positive: when a case was positive and predicted positive
# 
# FN / False Negative: when a case was positive but predicted negative
# 
# FP / False Positive: when a case was negative but predicted positive
# 
# ## D. Precision
# An ability of a classifier not to label positive to the negatives
# When you say a male is pregnant, which is not possible yet, then this would be detected under this precision score.
# (Number of true positive cases) / (Number of all the positive cases)
# *all the positive classes = true positive + false positive
# 
# ## E. Recall
# 
# An ability of a classifier to find all positive instances. So, only corrected measured instances, which are true-positive and false-negatives, are concerned.
# (Number of true positives) / (# of true positives + # of false negatives)
# *# signifies ‘number’
# By far, we can tell both Precision and Recall focus on true-positive cases in different perspectives.
# 
# ## F. F1-score
# 
# This is a weighted harmonic mean value using both Precision and Recall. This measure is pretty useful when the dataset has an imbalanced distribution of different labels.
# {(Precision * Recall) * 2} / (Precision + Recall)
# 
# 
# 
# 
# 
# # VIII CONCLUSION
# Primarily in this work classification is performed using machine learning techniques. Various techniques deployed were compared and contrasted in terms of evaluation metrics. Most of the classification tasks will suit to be processed in the pipline discussed in above sections. In future this work can extend to explore the  potential of Neural Networks for classification tasks. 
# 
# # IX REFERENCES
# 
# 1.   [EDA + Logistic Regression + PCA by *Prashant Banerjee*](https://www.kaggle.com/prashant111/eda-logistic-regression-pca)
# 
# 1.   [Income Prediction (84.369% Accuracy) by *IPByrne*](https://www.kaggle.com/ipbyrne/income-prediction-84-369-accuracy)
# 2.   [Multiple ML Techniques and Analysis of Dataset by *Matt Green* ](https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset)
# 
# 2.   [Catboost and other class.algos with 88% accuracy by *Kanav Anand*](https://www.kaggle.com/kanav0183/catboost-and-other-class-algos-with-88-accuracy)
# 
# 2.   [EDA and Income predictions (86.75 % accuracy) by *Sumit Mishra*](https://www.kaggle.com/sumitm004/eda-and-income-predictions-86-75-accuracy)
# 
# 1.   [Income prediction using Random Forest and XGBoost by *Nitineshwar*](https://www.kaggle.com/grayphantom/income-prediction-using-random-forest-and-xgboost)
# 
# 

# Now lets dive into the code and see how to solve the dataset.

# ##  Importing necessary Libraries
# 

# Catboost algorithm might need installation, the below code is in the comment form. Run it if installation is required 

# In[ ]:


#pip install catboost


# In[ ]:


#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score,f1_score,log_loss,confusion_matrix,classification_report,precision_score,recall_score


# In[ ]:


#Reading Dataset
dataset = pd.read_csv('../input/adult-census-income/adult.csv')
dataset.head()


# We have two variables regarding Education, "education" and "education.num"."education.num" is the categorical version of "education" so we can drop the "education" variable

# In[ ]:


#Dropping education since we have its categorical coloumn education.num
dataset.drop(columns=['education'],inplace=True)


# In[ ]:


#data types of columns
dataset.dtypes


# In[ ]:


#List of columns present in Dataset
dataset.columns


# In[ ]:


#Shape of Dataset
dataset.shape


# All these columns names consits of ".", which cause problem while calling the columns, so replacing all the "." with "_" in the column names

# In[ ]:


#Renaming few columns
dataset.rename(columns = {'education.num':'education_num', 'marital.status':'marital_status', 'capital.gain':'capital_gain',
                          'capital.loss':'capital_loss','hours.per.week':'hours_per_week','native.country':'native_country'}, inplace = True) 


# In[ ]:


#Checking the change
dataset.columns


# "?" represents null values , lets replace them with np.nan

# In[ ]:


#Replacing "?" with NAN
dataset['workclass'].replace('?', np.nan, inplace= True)
dataset['occupation'].replace('?', np.nan, inplace= True)
dataset['native_country'].replace('?', np.nan, inplace= True)


# In[ ]:


#A detailed description of the datset
dataset.describe(include='all')


# ## Handling the Missing values
# 
# Every datset have some missing values, lets find out in which cloumns they are?

# In[ ]:


#Number of null values in the dataset column wise
dataset.isnull().sum()


# In[ ]:


#Grouping Workclass
dataset.groupby(['workclass']).size().plot(kind="bar",fontsize=14)
plt.xlabel('Work Class Categories')
plt.ylabel('Count of People')
plt.title('Barplot of Workclass Variable')


# All the null values in the "workclass" can be replaced by "private"

# In[ ]:


#Grouping Occupation
dataset.groupby(['occupation']).size().plot(kind="bar",fontsize=14)
plt.xlabel('Occupation Categories')
plt.ylabel('Count of People')
plt.title('Barplot of Occupation Variable')


# "occupation" null values cant be replaced by mode , its more or like equally distributed. So drop the null values in this column

# In[ ]:


#Grouping Native Country
dataset.groupby(['native_country']).size().plot(kind="bar",fontsize=10)
plt.xlabel('Native Country Categories')
plt.ylabel('Count of People')
plt.title('Barplot of Native Country Variable')


# Its clear the null values of "native_country" could be easily replaced by mode.

# In[ ]:


#Droping null values in occupation column
dataset.dropna(subset=['occupation'],inplace=True)
dataset.isnull().sum()


# By dropping the null values in "occupation" we lose the null values in "workclass" also.

# In[ ]:


#Imputing null values with Mode

dataset['native_country'].fillna(dataset['native_country'].mode()[0], inplace=True)


# In[ ]:


#Checking for null values
dataset.isnull().sum()


# In[ ]:


#Confirming the Categorical Features
categorical_feature_mask = dataset.dtypes==object
categorical_feature_mask


# ## Label Encoding
# 
# All the categorical columns and the columns with text data are being Label Encodeded in this step.

# In[ ]:


##Label encoding the all the categorical features


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_list=['income','workclass','marital_status','occupation','relationship','race','sex','native_country']
dataset[cat_list]=dataset[cat_list].apply(lambda x:le.fit_transform(x))


# In[ ]:


#Number of categories in dataset
dataset.nunique()


# ## Correlation
# 
# To find out whether there is any relation between variables, in other terms multicollineariaty.
# 
# 

# In[ ]:


#Finding Correlation between variables
corr = dataset.corr()
mask = np.zeros(corr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
plt.subplots(figsize=(10,7))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=True,cmap='RdYlGn',mask = mask)


# In[ ]:


#Dropping "sex" variable since it is highly correlated with "relationship" variable 
dataset.drop(columns=['sex'],inplace=True)


# In[ ]:


#Slicing dataset into Independent(X) and Target(y) varibles
y = dataset.pop('income')
X = dataset


# In[ ]:


#Scaling the dependent variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


#Dividing dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print(X_train.shape)
print(y_train.shape)


# In[ ]:


#Performing Recursive Feauture Elimation with Cross Validation
#Using Random forest for RFE-CV and logloss as scoring
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
clf_rf=RandomForestClassifier(random_state=0)
rfecv=RFECV(estimator=clf_rf, step=1,cv=5,scoring='neg_log_loss')
rfecv=rfecv.fit(X_train,y_train)


# In[ ]:


#Optimal number of features
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])


# In[ ]:


#Feauture Ranking
clf_rf = clf_rf.fit(X_train,y_train)
importances = clf_rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# In[ ]:


#Selecting the Important Features
X_train = X_train.iloc[:,X_train.columns[rfecv.support_]]
X_test = X_test.iloc[:,X_test.columns[rfecv.support_]]


# In[ ]:


#Creating anew dataframe with column names and feature importance
dset = pd.DataFrame()
data1 = dataset

dset['attr'] = data1.columns


dset['importance'] = clf_rf.feature_importances_
#Sorting with importance column
dset = dset.sort_values(by='importance', ascending=True)

#Barplot indicating Feature Importance
plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()


# # 3 CLASSIFICATION MODEL BUILDING

# In[ ]:


classifier_lg = LogisticRegression(random_state=0)
classifier_dt = DecisionTreeClassifier(random_state=0)
classifier_nb = GaussianNB()
classifier_knn = KNeighborsClassifier()
classifier_rf = RandomForestClassifier(random_state=0)
classifier_xgb = XGBClassifier(random_state=0)
classifier_cgb = CatBoostClassifier(random_state=0)


# ## Training with didfferent Algorithms

# In[ ]:




# Instantiate the classfiers and make a list
classifiers = [classifier_lg,
               classifier_dt,
               classifier_nb,
               classifier_knn,
               classifier_rf,
               classifier_xgb,
               classifier_cgb]
# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','Roc Auc','Accuracy','f1 Score','logloss','Confusion Matrix','Precision','Recall'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[::,1]
    y_pred = model.predict(X_test)
    print(cls, '\n','Confusion Matrix','\n',confusion_matrix(y_test,  y_pred))
    print('\n','Classification Report','\n',classification_report(y_test,  y_pred))
    print('='*170)
    fpr, tpr, _ = roc_curve(y_test,  y_proba)
    auc = roc_auc_score(y_test, y_proba)
    Accuracy = accuracy_score(y_test,y_pred)
    f1score = f1_score(y_test,y_pred)
    logloss = log_loss(y_test,y_proba)
    cm = confusion_matrix(y_test,  y_pred)
    precision = precision_score(y_test,  y_pred)
    recall = recall_score(y_test,  y_pred)
  
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'Roc Auc':auc,
                                        'Accuracy':Accuracy,
                                        'f1 Score':f1score,
                                        'logloss':logloss,
                                        'Confusion Matrix': cm,
                                        'Precision':precision,
                                        'Recall':recall}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


# ## Roc Plot

# In[ ]:


fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['Roc Auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# In[ ]:


result_table[['Roc Auc','Accuracy','f1 Score','logloss','Confusion Matrix','Precision','Recall']]


# CatBoost, XGBoost both performed well, both gave good accuracy, f1score, log_loss.
