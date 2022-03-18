#!/usr/bin/env python
# coding: utf-8

# # Handling Missing Values in Python
# 
# ![](https://imgur.com/68u0dD2.png)
# 
# >Real world data is messy and often contains a lot of missing values. There could be multiple  reasons for the missing values but primarily the reason for missingness can be attributed to 
# 
# | Reason for missing Data |  
# |--|
# |Data doesn't exist  |  
# |Data not collected due to human error.  |  
# |Data deleted accidently  |  
# |                         |
# 
#  Either way we need to address this issue before we  proceed with the modeling stuff. It is also important to note that some algorithms like `XGBoost` and `LightGBM` can treat missing data without any preprocessing.
# <hr>
# 

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Objective" data-toc-modified-id="Objective-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Objective</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data</a></span></li><li><span><a href="#Loading-necessary-libraries-and-datasets" data-toc-modified-id="Loading-necessary-libraries-and-datasets-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Loading necessary libraries and datasets</a></span><ul class="toc-item"><li><span><a href="#Reading-in-the-dataset" data-toc-modified-id="Reading-in-the-dataset-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Reading in the dataset</a></span></li><li><span><a href="#Examining-the-Target-column" data-toc-modified-id="Examining-the-Target-column-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Examining the Target column</a></span></li></ul></li><li><span><a href="#Detecting-Missing-values" data-toc-modified-id="Detecting-Missing-values-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Detecting Missing values</a></span><ul class="toc-item"><li><span><a href="#Detecting-missing-values-numerically" data-toc-modified-id="Detecting-missing-values-numerically-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Detecting missing values numerically</a></span></li><li><span><a href="#Detecting-missing-data-visually-using-Missingno-library" data-toc-modified-id="Detecting-missing-data-visually-using-Missingno-library-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Detecting missing data visually using Missingno library</a></span><ul class="toc-item"><li><span><a href="#Visualizing-the-locations-of-the-missing-data" data-toc-modified-id="Visualizing-the-locations-of-the-missing-data-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Visualizing the locations of the missing data</a></span></li></ul></li></ul></li><li><span><a href="#Reasons-for-Missing-Values" data-toc-modified-id="Reasons-for-Missing-Values-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Reasons for Missing Values</a></span><ul class="toc-item"><li><span><a href="#Finding-reason-for-missing-data-using-matrix-plot" data-toc-modified-id="Finding-reason-for-missing-data-using-matrix-plot-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Finding reason for missing data using matrix plot</a></span></li><li><span><a href="#Finding-reason-for-missing-data-using-a-Heatmap" data-toc-modified-id="Finding-reason-for-missing-data-using-a-Heatmap-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Finding reason for missing data using a Heatmap</a></span></li><li><span><a href="#Finding-reason-for-missing-data-using-Dendrogram" data-toc-modified-id="Finding-reason-for-missing-data-using-Dendrogram-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Finding reason for missing data using Dendrogram</a></span></li></ul></li><li><span><a href="#Treating-Missing-values" data-toc-modified-id="Treating-Missing-values-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Treating Missing values</a></span><ul class="toc-item"><li><span><a href="#Deletions" data-toc-modified-id="Deletions-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Deletions</a></span><ul class="toc-item"><li><span><a href="#Pairwise-Deletion" data-toc-modified-id="Pairwise-Deletion-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Pairwise Deletion</a></span></li><li><span><a href="#Listwise-Deletion/-Dropping-rows" data-toc-modified-id="Listwise-Deletion/-Dropping-rows-6.1.2"><span class="toc-item-num">6.1.2&nbsp;&nbsp;</span>Listwise Deletion/ Dropping rows</a></span></li><li><span><a href="#Dropping-complete-columns" data-toc-modified-id="Dropping-complete-columns-6.1.3"><span class="toc-item-num">6.1.3&nbsp;&nbsp;</span>Dropping complete columns</a></span></li></ul></li><li><span><a href="#Imputations-Techniques-for-non-Time-Series-Problems" data-toc-modified-id="Imputations-Techniques-for-non-Time-Series-Problems-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Imputations Techniques for non Time Series Problems</a></span></li><li><span><a href="#Imputations-Techniques-for-Time-Series-Problems" data-toc-modified-id="Imputations-Techniques-for-Time-Series-Problems-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Imputations Techniques for Time Series Problems</a></span><ul class="toc-item"><li><span><a href="#Time-Series-dataset" data-toc-modified-id="Time-Series-dataset-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Time Series dataset</a></span></li><li><span><a href="#Imputing-using-ffill" data-toc-modified-id="Imputing-using-ffill-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>Imputing using ffill</a></span></li><li><span><a href="#Imputation-using-bfill" data-toc-modified-id="Imputation-using-bfill-6.3.3"><span class="toc-item-num">6.3.3&nbsp;&nbsp;</span>Imputation using bfill</a></span></li><li><span><a href="#Imputation-using-Linear-Interpolation-method" data-toc-modified-id="Imputation-using-Linear-Interpolation-method-6.3.4"><span class="toc-item-num">6.3.4&nbsp;&nbsp;</span>Imputation using Linear Interpolation method</a></span></li></ul></li><li><span><a href="#Advanced-Imputation-Techniques" data-toc-modified-id="Advanced-Imputation-Techniques-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Advanced Imputation Techniques</a></span><ul class="toc-item"><li><span><a href="#K-Nearest-Neighbor-Imputation" data-toc-modified-id="K-Nearest-Neighbor-Imputation-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>K-Nearest Neighbor Imputation</a></span></li><li><span><a href="#Multivariate-feature-imputation---Multivariate-imputation-by-chained-equations-(MICE)" data-toc-modified-id="Multivariate-feature-imputation---Multivariate-imputation-by-chained-equations-(MICE)-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>Multivariate feature imputation - Multivariate imputation by chained equations (MICE)</a></span></li></ul></li></ul></li><li><span><a href="#Algorithms-which-handle-missing-values" data-toc-modified-id="Algorithms-which-handle-missing-values-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Algorithms which handle missing values</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Conclusion</a></span></li><li><span><a href="#References-and-good-resources" data-toc-modified-id="References-and-good-resources-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References and good resources</a></span></li></ul></div>

# # Objective
# The objective of this notebook is to detect missing values and then go over some of the methods used for imputing them.
# <hr>
# 
# 
# # Data
# 
# There are two publically available datasets which will be used to explain the concepts:
# 
# * 1. [Titanic Dataset](https://www.kaggle.com/c/titanic) for Non Time Series problem
# * 2. [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/rohanrao/air-quality-data-in-india) for Time Series problem
# 
# 
# # Loading necessary libraries and datasets

# In[ ]:


import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

import missingno as msno


# ## Reading in the dataset
# * Reading in the Titanic Dataset.

# In[ ]:


os.listdir('../input/titanic')


# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

# First few rows of the training dataset
train.head()


# ## Examining the Target column 
# 
# >For each passenger in the test set, we need to create a model that predicts whether or not the passengers survived the sinking of the Titanic. Hence Survived is that target column in the dataset. Let's examine the Distribution of the target column

# In[ ]:



train['Survived'].value_counts()


# In[ ]:


s = sns.countplot(x = 'Survived',data = train)
sizes=[]
for p in s.patches:
    height = p.get_height()
    sizes.append(height)
    s.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/len(train)*100),
            ha="center", fontsize=14) 


# Here:
# * 0: Did not Survive while 
# * 1: Survived. 
# 
# Clearly, less people survived the accident.

# >Since the focus of the notebook is to detect and handle missing values, we'll jump directly into it. Let's now look at a step by step process to manage the missing values in a dataset.
# <hr>

# # Detecting Missing values
# 
# ## Detecting missing values numerically 
# 
# >The first step is to detect the count/percentage of missing values in every column of the dataset. This will give an idea about the distribution of missing values.

# In[ ]:


# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 
# One of the best notebooks on getting started with a ML problem.

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


train_missing= missing_values_table(train)
train_missing


# In[ ]:


test_missing= missing_values_table(test)
test_missing


# Both the train and test set have the same proportion of the missing values. 

# ## Detecting missing data visually using Missingno library
# 
# >To graphically analyse the missingness of the data, let's use a library called [Missingno](https://github.com/ResidentMario/missingno) It is a package for graphical analysis of missing values. To use this library, we need to import it as follows: `import missingno as msno`

# In[ ]:


msno.bar(train)


# >The bar chart above gives a quick graphical overview of the completeness of the dataset. We can see that Age, Cabin and embarked columns have missing values. Next,it would make sense to find out the locations of the missing data.

# ### Visualizing the locations of the missing data 
# 
# >The [msno.matrix](https://github.com/ResidentMario/missingno#matrix) nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion.
# 

# In[ ]:


msno.matrix(train)


# >* The plot appears blank(white) wherever there are missing values. For instance, in Embarked column there are only two instances of missing data, hence the two white lines.
# >
# >* The sparkline on the right gives an idea of the general shape of the completeness of the data and points out the row with the minimum nullities and the total number of columns in a given dataset, at the bottom.
# >
# >It is also possible to sample the dataset to pinpoint the exact location of the missing values. For instance let's check the first 100 rows.

# In[ ]:


msno.matrix(train.sample(100))


# This shows that the Embarked column has no nullity in the first 100 cases.

# # Reasons for Missing Values 
# 
# >Before we start treating the missing values ,it is important to understand the various reasons for the missingness in data. Broadly speaking, there can be three possible reasons:
# 
# ![](https://cjasn.asnjournals.org/content/clinjasn/early/2014/02/06/CJN.10141013/F2.large.jpg?width=800&height=600&carousel=1)
# 
# source:https://cjasn.asnjournals.org/content/early/2014/02/06/CJN.10141013/tab-figures-data?versioned=true
# 
# 
# **1. Missing Completely at Random (MCAR) **
# 
# >The missing values on a given variable (Y) are not associated with other variables in a given data set or with the variable (Y) itself. In other words, there is no particular reason for the missing values.
# 
# **2. Missing at Random (MAR) **
# 
# >MAR occurs when the missingness is not random, but where missingness can be fully accounted for by variables where there is complete information.
# 
# **3. Missing Not at Random (MNAR) **
# >Missingness depends on unobserved data or the value of the missing data itself. 
# 
# *All definitions taken from Wikipedia: https://en.wikipedia.org/wiki/Missing_data*
# 
# 
# >Now let us look at nullity matrix again to see if can find what type of missingness is present in the dataset.
# 

# ## Finding reason for missing data using matrix plot 

# In[ ]:


msno.matrix(train)


# >* The `Embarked` Column has very few missing values and donot seem to be correlated with any other column, Hence, the missingness in Embarked column can be attributed as Missing Completely at Random.
# * Both the `Age` and the `Cabin` columns have a lot of missing values.This could be a case of MAR as we cannot directly observe the reason for missingness of data in these columns.
# 
# The missingno package also let's us sort the graph by a particluar column. Let's sort the values by `Age` and `Cabin` column to see if there is a pattern in the missing values

# In[ ]:


#sorted by Age
sorted = train.sort_values('Age')
msno.matrix(sorted)


# Hence it is clear that here is no relation between the missingness in Age and Cabin column.To cement this conclusion further we can also draw a Heatmap among the different variables in the dataset.

# ## Finding reason for missing data using a Heatmap 

# In[ ]:


msno.heatmap(train)


# The heatmap function shows that there are no strong correlations between missing values of different features. This is good; low correlations further indicate that the data are MAR.

# ## Finding reason for missing data using Dendrogram 
# A dendogram is a tree diagram of missingness. It groups the highly correlated variables together.

# In[ ]:


msno.dendrogram(train)


# >Let's read the above dendrogram from a top-down perspective: 
# * Cluster leaves which linked together at a distance of zero fully predict one another's presence—one variable might always be empty when another is filled, or they might always both be filled or both empty, and so on(missingno documentation)
# >
# >![Screenshot%202020-04-25%20at%208.19.56%20AM.png](attachment:Screenshot%202020-04-25%20at%208.19.56%20AM.png)
# >
# >* the missingness of Embarked tends to be more similar to Age than to Cabin and so on.However, in this particluar case, the correlation is high since Embarked column has a very few missing values.
# 
# This dataset doesn't have much missing values but if you use the same methodology on datasets having a lot of missing values, some interesting pattern will definitely emerge.

# # Treating Missing values  
# 
# After having identified the patterns in missing values, it is important to treat them too. Here is a flowchart which could prove handy:
# 
# 
# 
# ## Deletions 
# 
# ![](https://imgur.com/tBvdfyX.png)
# 
# >Deletion means to delete the missing values from a dataset. This is however not recommended as it might result in loss of information from the dataset. We should only delete the missing values from a dataset if their proportion is very small. Deletions are further of three types: 
# 
# ### Pairwise Deletion 
# 
# Parwise Deletion is used when values are missing completely at random i.e MCAR. During Pairwise deletion, only the missing values are deleted. All operations in pandas like mean,sum etc intrinsically skip missing values.
# 
# 
# 

# In[ ]:


train.isnull().sum()


# In[ ]:


train_1 = train.copy()
train_1['Age'].mean() #pandas skips the missing values and calculates mean of the remaining values.


# 
# ### Listwise Deletion/ Dropping rows
# 
# >During Listwise deletion, complete rows(which contain the missing values) are deleted. As a result, it is also called Complete Case deletion. Like Pairwise deletion, listwise deletions are also only used for MCAR values.
# 

# In[ ]:


#Drop rows which contains any NaN or missing value for Age column
train_1.dropna(subset=['Age'],how='any',inplace=True)
train_1['Age'].isnull().sum()


# >The Age column doesn't have any missing values.A major diadvantage of Listwise deletion is that a major chunk of data and hence a lot of information is lost. Hence, it is advisable to use it only when the number of missing values is very small.

# 
# ### Dropping complete columns 
# 
# If a column contains a lot of missing values, say more than 80%, and the feature is not significant, you might want to delete that feature. However, again, it is not a good methodology to delete data.
# 

# ## Imputations Techniques for non Time Series Problems 
# 
# ![](https://imgur.com/bL0iHde.png)
# 
# >Imputation refers to replacing missing data with substituted values.There are a lot of ways in which the missing values can be imputed depending upon the nature of the problem and data. Dependng upon the nature of the problem, imputation techniques can be broadly they can be classified as follows:
# 
# 
# ### Basic Imputation Techniques
#   
#   - Imputating with a constant value
#   - Imputation using the statistics (mean, median or most frequent) of each column in which the missing values are located
# 
# For this we shall use the `The SimpleImputer` class from sklearn.

# In[ ]:


# imputing with a constant

from sklearn.impute import SimpleImputer
train_constant = train.copy()
#setting strategy to 'constant' 
mean_imputer = SimpleImputer(strategy='constant') # imputing using constant value
train_constant.iloc[:,:] = mean_imputer.fit_transform(train_constant)
train_constant.isnull().sum()


# In[ ]:


from sklearn.impute import SimpleImputer
train_most_frequent = train.copy()
#setting strategy to 'mean' to impute by the mean
mean_imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 
train_most_frequent.iloc[:,:] = mean_imputer.fit_transform(train_most_frequent)


# In[ ]:


train_most_frequent.isnull().sum()


# ## Imputations Techniques for Time Series Problems
# 
# Now let's look at ways to impute data in a typical time series problem. Tackling missing values in time Series problem is a bit different. The `fillna()` method is used for imputing missing values in such problems.
# 
# * Basic Imputation Techniques
#   - 'ffill' or 'pad' - Replace NaN s with last observed value
#   - 'bfill' or 'backfill' - Replace NaN s with next observed value
#   -  Linear interpolation method
# 
# ### Time Series dataset
# 
# The dataset is called [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/rohanrao/air-quality-data-in-india) Tand it contains air quality data and AQI (Air Quality Index) at hourly and daily level of various stations across multiple cities in India.The dataset has a lot of missing values and and is a classic Time series problem.

# In[ ]:


os.listdir('../input/air-quality-data-in-india')


# In[ ]:


city_day = pd.read_csv('../input/air-quality-data-in-india/city_day.csv',parse_dates=True,index_col='Date')
city_day1=city_day.copy(deep=True)
city_day.head()


# I won't go much into explaining the data since I have done a lot of relatedw work in my kernel titled [😷 Breathe India: COVID-19 effect on Pollution](https://www.kaggle.com/parulpandey/breathe-india-covid-19-effect-on-pollution).In this notebook, let's keep our focus on the missing values only. As id evident, city_day dataframe consists of daily pollution level data of some of the prominent cities in India.
# 
# 

# In[ ]:


#Missing Values
city_day_missing= missing_values_table(city_day)
city_day_missing


# There are a lot of missing values and some of the columns like Xylene and PM10 have more than 50% of the values missing. Let's now see how we can impute these missing values.
# 
# 

# In[ ]:


# Imputation using ffill/pad
# Imputing Xylene value

city_day['Xylene'][50:64]


# Above we see, there are 3 missing values in the Xylene column. 
# 
# ### Imputing using ffill

# In[ ]:


city_day.fillna(method='ffill',inplace=True)
city_day['Xylene'][50:65]


# We can see that all missing values have been filled with the last observed values.
# 
# ### Imputation using bfill

# In[ ]:


# Imputing AQI value

city_day['AQI'][20:30]


# In[ ]:


city_day.fillna(method='bfill',inplace=True)
city_day['AQI'][20:30]


# We can see that all missing values have been filled with the next observed values.

# ### Imputation using Linear Interpolation method
# 
# Time series data has a lot of variations against time. Hence, imputing using backfill and forward fill isn't the ebst possible solution to address the missing value problem. A more apt alternative would be to use interpolation methods, where the values are filled with incrementing or decrementing values.
# 
# [Linear interpolation](https://www.lexjansen.com/nesug/nesug01/ps/ps8026.pdf) is an imputation technique that assumes a linear relationship between data points and utilises non-missing values from adjacent data points to compute a value for a missing data point. 
# 
# Refer to the official documentation for a complete list of interpolation strategies [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)

# In[ ]:


city_day1['Xylene'][50:65]


# In[ ]:


# Interpolate using the linear method
city_day1.interpolate(limit_direction="both",inplace=True)
city_day1['Xylene'][50:65]


# ## Advanced Imputation Techniques 
# 
# Advanced imputation techniques uses machine learning algorithms to impute the missing values in a dataset unlike the previous techniques where we used other column values to predict the missing values. We shall look at the following two techniques in this notebook:
# 
# * [Nearest neighbors imputation](https://scikit-learn.org/stable/modules/impute.html#nearest-neighbors-imputation)
# * [Multivariate feature imputation](https://scikit-learn.org/stable/modules/impute.html#multivariate-feature-imputation)
# 
# ### K-Nearest Neighbor Imputation
# 
# The [KNNImputer class](https://scikit-learn.org/stable/modules/impute.html#multivariate-feature-imputation) provides imputation for filling in missing values using the k-Nearest Neighbors approach.Each missing feature is imputed using values from n_neighbors nearest neighbors that have a value for the feature. The feature of the neighbors are averaged uniformly or weighted by distance to each neighbor. 

# In[ ]:


train_knn = train.copy(deep=True)


# In[ ]:


from sklearn.impute import KNNImputer
train_knn = train.copy(deep=True)

knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")
train_knn['Age'] = knn_imputer.fit_transform(train_knn[['Age']])


# In[ ]:


train_knn['Age'].isnull().sum()


# ### Multivariate feature imputation - Multivariate imputation by chained equations (MICE)
# A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion. It performns multiple regressions over random sample ofthe data, then takes the average ofthe multiple regression values and uses that value to impute the missing value. In sklearn, it is implemented as follows:
# 
# 

# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
train_mice = train.copy(deep=True)

mice_imputer = IterativeImputer()
train_mice['Age'] = mice_imputer.fit_transform(train_mice[['Age']])


# In[ ]:


train_mice['Age'].isnull().sum()


# # Algorithms which handle missing values
# 
# Some algprithms like XGBoost and LightGBM can handle missing values without any preprocessing, by supplying relevant parameters.
# 
# * [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
# ![](https://imgur.com/GNlcWoJ.png)
# 
# * [LightGBm](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html)
# ![](https://imgur.com/GVdcx7h.png)
# 
# 
# # Conclusion
# 
# Well, there is no single best way to handle missing values. One needs to experiment with different methods and then decide which method is best for a particular problem.

# # References and good resources 
# 
# * [Dealing with Missing Data in Python](https://campus.datacamp.com/courses/dealing-with-missing-data-in-python/the-problem-with-missing-data?ex=1)
# * [How to Handle Missing Data](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)

# In[ ]:




