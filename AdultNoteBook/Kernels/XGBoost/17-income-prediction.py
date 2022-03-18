#!/usr/bin/env python
# coding: utf-8

# # **Predicting An Individual Incomes via Census Data**
# 
# This is my fist Kaggle Notebook, it is a further exploration on top of introductory studies to the fundamentals of analytics and machine learning models. The initial section will explain the data mining problem at hand and the process taken to provide a solution to this problem. The second section will provide an exploration of the attributes with in the dataset, identifying key attributes that show promise in being able to assist with the classification task. The final section will involve the creation, development & results of the classification methodologies used to solve the task.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import cufflinks as cf
import plotly.offline
import pandas_profiling
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import cufflinks as cf
import warnings


cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
warnings.filterwarnings('ignore')


# **Importing file and brief analysis of the datasets content**

# In[ ]:


AdultDf = pd.read_csv('../input/adult-census-income/adult.csv')
AdultDf.head()


# # Exploritory Analysis
# 
# **Pandas Profiling** was implemented to provide quick and efficient initial exploratory analysis of the dataset. The Profiling tool shows initial attribute distributions, identifies outliers and highlights correlations between data that can be further explored in the EDA process. 
# 
# Pandas Profiling is a quite new and exciting package for EDA utilising Pandas, with consistent continued version releases it is a tool that is only going to improve. To find out more visit the [Pandas Profiling GitHub:](https://github.com/pandas-profiling/pandas-profiling)

# In[ ]:


profile = pandas_profiling.ProfileReport(AdultDf, title='Pandas Profiling Report for training dataset', html={'style':{'full_width':True}})


# In[ ]:


profile.to_notebook_iframe()


# the profiling report identified a few issues that needed to be addressed before continued exploration, these included removal of outliers and null values which used the sentinel value '?'.

# In[ ]:


AdultDf = AdultDf.replace('?', np.nan)
AdultDf = AdultDf.dropna()

AdultDf = AdultDf[AdultDf.age != 90]
AdultDf = AdultDf[AdultDf['capital.gain'] != 99999]

AdultDf.head()


# # Continued Exploration & Visulisations
# 
# This notebook uses cufflinks which integrates with Plotly, feel free to explore the visualisations with the inbuilt, interactive tools provided by the package.

# # Age:
# The distribution of incomes between ages shows a clear positive skew with regard to age, and an increase in the proportion of individuals earning over $50k per year between the ages of 30 and 55. This conclusion is not unsurprising as it follows the expected pattern of adult career progression with younger individuals likely to either studying or starting their careers. Those above 55 are likely to be starting to progress towards retirement, which is associated with lower expected levels of income due to reduced working hours.

# In[ ]:


IncomeDist = pd.crosstab(AdultDf.age, AdultDf.income)
IncomeDist.iplot(kind ='bar', barmode = 'stack', xTitle = 'Age', yTitle = 'Num of Individuals', title = 'Distribution of Income between Ages', theme = 'white')


# # Education:
# The removal of one of these attributed would be beneficial to analysis as it acts to reduce the overall dimensionality of the dataset. The initial exploration through Pandas Profiling utilising 'Cramérs V' showed that the two attributes were highly correlated. Further analysis of these attributes and their relationship to income was conducted to confirm this conclusion.

# In[ ]:


AdultDf.iplot(kind ='histogram', column ='education.num', color ='orange', xTitle = 'Years Spent in Education', yTitle = 'Num of Individuals', title = 'Distribution of Income Levels', theme = 'white', bargap = 0.05)


# In[ ]:


EduIncome = pd.crosstab(AdultDf.education, AdultDf.income)
EducationLevels = {"Preschool":0, "1st-4th":1, "5th-6th":2, "7th-8th":3, "9th":4, "10th":5, "11th":6, "12th":7, "HS-grad":8, "Some-college":9, "Assoc-voc":10, "Assoc-acdm":11, "Bachelors":12, "Masters":13, "Prof-school":14, "Doctorate":15}
EduIncome = EduIncome.sort_values(by=['education'], key=lambda x: x.map(EducationLevels))
EduIncome.iplot(kind = 'bar', barmode = 'stack', xTitle = 'Education Levels', yTitle = 'Num of Individuals', title = 'Distibution of Education Levels', theme = 'white')


# # Relationship Status:
# 
# The relationship attribute was identified in the exploration as providing insight into the solution to the classification problem, further exploration of the attribute and its relationship between income confirmed this analysis. There is a clear increased proportion of higher income earners amongst those who are 'Married' either Husband or Wife, when compared to those who are unmarried, Own-Child, or Other-Relative. This may potentially be due to the increased age of those married when compared to those unmarried or classified as an 'Own-Child'.

# In[ ]:


IncomeDist = pd.crosstab(AdultDf['relationship'], AdultDf.income)
IncomeDist.iplot(kind ='bar', barmode = 'stack', xTitle = 'Relationship Status', yTitle = 'Num of Individuals', title = 'Distribution of Income between Relationship Status', theme = 'white')


# In[ ]:


MarAge = pd.crosstab(AdultDf.age, AdultDf.relationship)
MarAge.iplot(kind = 'bar', barmode = 'stack', xTitle = 'Age', yTitle = 'Num of Individuals', title = 'Distribution of Relationships between Ages', theme = 'white')


# # Gender:
# 
# A quick visualisation of the Sex attribute with respect to the proportion of income distributions between each shows the disparity between incomes between the genders. this clear distinction provides a useful attribute for assistance in the classification task.

# In[ ]:


SexIncome = pd.crosstab(AdultDf.sex, AdultDf.income)
SexIncome.iplot(kind = 'bar', barmode = 'stack', xTitle = 'Sex', yTitle = 'Num of Individuals', title = 'Distribution of Income between Sex', theme = 'white')


# # Occupation:
# 
# Occupation another attribute that was identified as having potential to be a useful attribute for use within the classification task due to its correlation with the income attribute. There is a increase in the proportion of those earning over $50K with respect to the 'Prof-Speciality' and 'Exec-Managerial' categories when compared to the other categories within the attribute.

# In[ ]:


OccuInc = pd.crosstab(AdultDf.occupation , AdultDf.income)
OccuInc = OccuInc.sort_values('<=50K', ascending = False)
OccuInc.iplot(kind = 'bar', barmode = 'stack', theme ='white', xTitle = 'Occupation', yTitle = 'Num of Individuals', title = 'Distribution of Income between Occupations')


# # Feature Engineering and Model Development

# The first step in this process is to convert the Income attribute the target attribute into 0 and 1 for implementation in model training.

# In[ ]:


AdultDfTarget = AdultDf.copy()
target = AdultDf['income'].unique()
Ints = {name: n for n, name in enumerate(target)}
AdultDf['target'] = AdultDf['income'].replace(Ints)
AdultDf.head()


# A number of attributes were identified as redundant. This was either due to the attribute providing no meaningful insight into the classification task, not being understandable from the analyst’s perspective, or already describing a feature that was also described by another attribute. Overall this should reduce the dimensionality of the dataset.

# In[ ]:


HeadDrop = ['capital.gain','capital.loss','education','fnlwgt','income']
AdultDf.drop(HeadDrop, inplace = True, axis = 1)
AdultDf.head()


# Due to the highly skewed nature of the 'native.country' with the disproportionate majority of individuals located in the United States, it was decided to collate all the Non-United States countries into a single value. This will assist with dimensionality reduction when the dataset is binarised down the line.

# In[ ]:


AdultDf['native.country'] = AdultDf['native.country'].replace(['Mexico', 'Greece', 'Vietnam', 'China', 'Taiwan',
       'India', 'Philippines', 'Trinadad&Tobago', 'Canada', 'South',
       'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran', 'England',
       'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba', 'Ireland',
       'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic', 'Haiti',
       'Hungary', 'Columbia', 'Guatemala', 'El-Salvador', 'Jamaica',
       'Ecuador', 'France', 'Yugoslavia', 'Portugal', 'Laos', 'Thailand',
       'Outlying-US(Guam-USVI-etc)', 'Scotland'], 'NonUS')

AdultDf['native.country'].unique()


# Binarising the dataset for use in model development and training.

# In[ ]:


categorical_columns = AdultDf.select_dtypes(exclude=np.number).columns
BinarisedDf = pd.get_dummies(data=AdultDf, prefix=categorical_columns, drop_first=True)

AdultDf = BinarisedDf
AdultDf.head()


# # Model Development and Training
# The Model was split into a training and test set with the target column removed for the test set. An initial Decision Tree Classifier was implemented to show a 'baseline' for model training. Further hyperparameter tuning and development should aim to improve upon this initial classification methodology.

# In[ ]:


X = AdultDf.drop('target', axis=1)
y = AdultDf['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

DtClas = DecisionTreeClassifier()
DtClas.fit(X_train, y_train)

DtPred = DtClas.predict(X_test)

print(classification_report(y_test, DtPred))
print("Accuracy :",accuracy_score(y_test, DtPred))


# # Logistic Regression:

# 'Baseline' Logistic Regression

# In[ ]:


LogReg = LogisticRegression(C = 0.05, max_iter = 1000)
LrMod = LogReg.fit(X_train, y_train)
LrPred = LrMod.predict(X_test)

print(classification_report(y_test, LrPred))
print("Accuracy :",accuracy_score(y_test, LrPred))


# Hyperparameter Tuning utilising 'GridSearchCV'

# In[ ]:


HypeParamLogReg = {'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio':[0.001, 0.01, 0.1]}
LogRegGrid = GridSearchCV(LogisticRegression(), param_grid=HypeParamLogReg, verbose=3)
LogRegGrid.fit(X, y)


# In[ ]:


print("Best Params : ",LogRegGrid.best_params_)
print("Accuracy : ",LogRegGrid.best_score_)


# Adjusted Logistic Regression 

# In[ ]:


LogRegAdj = LogisticRegression(C = 0.1, l1_ratio = 0.001, penalty = 'l2', max_iter = 1000)
LrModAdj = LogRegAdj.fit(X_train, y_train)
LrPredAdj = LrModAdj.predict(X_test)

print(classification_report(y_test, LrPredAdj))
print("Accuracy :",accuracy_score(y_test, LrPredAdj))


# # Random Forests:
# An initial 'baseline' Random Forest classifier was created to identify a rough estimate before conducting hyperparameter tuning

# In[ ]:


RandomForest = RandomForestClassifier(n_estimators=500,max_features=5)
RandomForest.fit(X_train, y_train)
RfPred = RandomForest.predict(X_test)

print(classification_report(y_test, RfPred))
print("Accuracy :",accuracy_score(y_test, RfPred))


# **Hyperparamater tuning** of the Random Forest Classifer utilising 'GridSearchCV'

# In[ ]:


HypeParamRf = {'criterion':['gini', 'entropy'], 'max_depth':[2, 5, 8, 11], 'n_estimators':[200, 300, 400, 500]}
RfGrid = GridSearchCV(RandomForestClassifier(), param_grid=HypeParamRf, verbose=3)

RfGrid.fit(X, y)


# In[ ]:


print("Best Params : ",RfGrid.best_params_)
print("Accuracy : ",RfGrid.best_score_)


# # Gradient Boosted Descent:

# In[ ]:


GradBoost = XGBClassifier(learning_rate = 0.35, n_estimator = 200)
GbMod = GradBoost.fit(X_train, y_train)
GbPred = GbMod.predict(X_test)


# In[ ]:


print(classification_report(y_test, GbPred))
print("Accuracy :",accuracy_score(y_test, GbPred))


# # Using folds to further improve accuracy

# Upon research for model implementation a hyperparameter training methodology was identified and implemented, the code is drawn from [Andrew Lukyanenko's](https://www.kaggle.com/artgor/bayesian-optimization-for-robots) implementation of this technique. the use of folds effectively combines a number of successful models together ultimately working together to produce a improved single classification model.

# In[ ]:


classifiers = [GradBoost,RandomForest,LogReg]

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)

scores_dict = {}

for train_index, valid_index in folds.split(X_train, y_train):
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]
    
    for classifier in classifiers:
        name = classifier.__class__.__name__
        classifier.fit(X_train_fold, y_train_fold)
        training_predictions = classifier.predict_proba(X_valid_fold)
        scores = roc_auc_score(y_valid_fold, training_predictions[:, 1])
        if name in scores_dict:
            scores_dict[name] += scores
        else:
            scores_dict[name] = scores


for classifier in scores_dict:
    scores_dict[classifier] = scores_dict[classifier]/folds.n_splits


# In[ ]:


print("Accuracy :",scores_dict)

