#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income Data Analysis

# ### **Abstract**: Predict whether income exceeds  50K /year based on census data.  Also known as "Census Income" dataset.  
# Data Set: https://archive.ics.uci.edu/ml/datasets/adult

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 1. Import Data Set

# The dataset contains featured attributes like age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week & native-country which has characterisation of categorical & integer values. The missing data is represented using '?' symbol.

# Import Pandas and Numpy Library

# In[ ]:


df = pd.read_csv("../input/adult.csv")
df.head()


# ## 2. Data Preprocessing

# During data filtering the '?' symbol is replaced with "NaN" to get a definite information from the dataset.

# ### Replace " ? " by NaN (Handling ' ? ' symbols)

# In[ ]:


# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)


# ###  Check for missing Data

# Since the symbol is now replaced with NaN which is null value, these are easily recognised and computed to check for the summation of the missing data available in the dataset.

# In[ ]:


missing_data = df.isnull()
missing_data.sum()


# From this statistics we get a total no of 1836 workclass data, 1843 occupation data, and 583 native country data missing from the existing dataset.

# ### Collect more statistics about Missing data

# Visualization of impact of data.

# In[ ]:


missing_col = []
for column in missing_data.columns.values.tolist():
    if(missing_data[column].sum() > 0):
        print("Column: ",column)
        print("Missing Data: {} ({:.2f}%)".format(missing_data[column].sum(), (missing_data[column].sum() * 100/ len(df))))
        print("Data Type: ",df[column].dtypes)
        print("")
        missing_col.append(column)


# ### As we can see from above statistics, we have 3 missing columns,  
# 1. workclass : 1836 missing data  
# 2. occupation: 1843 missing data  
# 3. native.country: 583 missing data  
#   
# ### Note:  All the missing data are categorical data  

# ### Impact of Missing Data on Data set

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig1 = plt.figure(figsize=(18,5))
i = 0
for column in missing_col:
    bad = missing_data[column].sum()
    good = len(df) - missing_data[column].sum()
    x = [bad, good]
    labels = ["Missing Data", "Good Data"]
    explode = (0.1, 0)
    i = i+1
    ax = fig1.add_subplot(1,3,i)
    ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})
    centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    ax.set_title(column, fontsize = 25)
plt.tight_layout()
plt.show()


# ## Fix Missing Data

# Now here the missing data is dealt-with accordingly based on certain rules.

# <a id="ref3"></a>
# ## Deal with missing data
# **How to deal with missing data:**
# 
#     
#     1. Drop data 
#         a. drop the whole row
#         b. drop the whole column
#     2. Replace data
#         a. replace it by mean
#         b. replace it by frequency
#         c. replace it based on other functions

# ### As we have only categorical missing data we will use " Replace by Frequency or Mode" Method

# The mode of each category is computed to fix the missing data based on frequency. The null values present in any column of dataset is replaced by the mode of the same column data.

# In[ ]:


# Calculate Mode
workclass_mode = df['workclass'].value_counts().idxmax()
occupation_mode = df['occupation'].value_counts().idxmax()
native_country_mode = df['native.country'].value_counts().idxmax()


# In[ ]:


print("Mode of workclass: ",workclass_mode)
print("Mode of Occupation: ",occupation_mode)
print("Mode of natice.country: ",native_country_mode)


# **Copy our original data frame to a dummy data frame**

# In[ ]:


df_manual = df


# ### Replace the missing categorical values by the most frequent value

# In[ ]:


#replace the missing categorical values by the most frequent value
df_manual["workclass"].replace(np.nan, workclass_mode, inplace = True)
df_manual["occupation"].replace(np.nan, occupation_mode, inplace = True)
df_manual["native.country"].replace(np.nan, native_country_mode, inplace = True)


# 1. **Check for any Null Values**

# After being replaced by the mode values effectively the algorithm doesnt find any missing values. The null value is computed zero for each attribute item.

# In[ ]:


df_manual.isnull().sum()


# **NO Null Values are present**

# # Convert Categorical Variables to Continuous Variable (Label Encoding with Binarization)

# **Without any numerical precedence**

# **Print all the categorical variables**

# In[ ]:


count = 0
for column in df_manual.columns.values.tolist():
    if df_manual[column].dtype == 'object':
        print("Column Name: ",column)
        print("Data Type: ", df_manual[column].dtypes)
        print("")
        count = count + 1
print("Count : ",count)


# ## We have above 9 Categorical Variables

# **Encode all the categorical values **

# ### Encoding "workclass"

# In[ ]:


dummy = pd.get_dummies(df_manual["workclass"])
dummy.head()


# Here we used pd.get_dummies function to convert categorical values into label encoded binary values
# **1** indicates the positive verification while, **0** indicates negative verification.
# 
# These are Dummy variables alternatively called as indicator variables that take discrete values such as 1 or 0 marking the presence or absence of a particular category.

# In[ ]:


#Rename column names
dummy.rename(columns={'Federal-gov':'work-Federal-gov', 
                      'Local-gov':'work-Local-gov',
                      'Private': 'work-Private',
                      'Self-emp-inc': 'work-Self-emp-inc',
                      'Self-emp-not-inc': 'Self-emp-not-inc',
                      'State-gov': 'work-State-gov',
                      'Without-pay' : 'work-Without-pay'}, inplace=True)
dummy.head()


# We renamed the attribute for our convinience

# # What is Dummy Variable Trap?  
# The Dummy Variable trap is a scenario in which the independent variables are multicollinear - a scenario in which two or more variables are highly correlated; in simple terms one variable can be predicted from the others.   
# <img src = "https://qph.fs.quoracdn.net/main-qimg-4445db89d9218ba35ceedcf8d1e73d35.webp"></img>   
# Lets encode the categorical column called “Country” and its values are -** [India, Germany, France]**  
# In ML regression models, predictions will do the good job if categorical values are converted into numerical (binary vectors ) values. Encoding categorical data technique to apply for the above categorical set and the values a.k.a dummy variables will become  
# <img src = "https://qph.fs.quoracdn.net/main-qimg-c9ee21fe8c9294294f81ee7d39dddedb.webp"></img>  
# Which dummy variable column do we need drop?   
# The answer is - we can drop any of one dummy variables column. It can predict the dropped column’s value based on other two columns. **Let’s take the record no 3 in the above table, both dummy variable values are ‘0’. So obviously another dummy variable column value is ‘1’ and categorical value is ‘Germany’**

# ### Drop one column to avoid Dummy Variable Trap. Here we dropped "Never-worked" column and took it as base value.  
# Here we reduced the multi-colinearity.

# In[ ]:


dummy.drop("Never-worked", axis = 1, inplace=True)
dummy.head()


# ### Add the dummy variables to Main Data Frame  
# We append the dummy variable with the main dataframe and drop the original column workplace.

# In[ ]:


# merge data frame "df" and "dummy" 
df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "workplace" from "df"
df_manual.drop("workclass", axis = 1, inplace=True)
df_manual.head()


# The above table shows the label encoded category.

# ## Encode "education"

# In[ ]:


dummy = pd.get_dummies(df_manual["education"])
dummy.head()


# In[ ]:


dummy.drop("Some-college", axis = 1, inplace=True)
dummy.head()


# In[ ]:


# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"
df_manual.drop("education", axis = 1, inplace=True)
df_manual.head()


# ## Encode "marital.status"

# In[ ]:


dummy = pd.get_dummies(df_manual["marital.status"])
dummy.head()


# In[ ]:


dummy.drop("Never-married", axis = 1, inplace=True)
# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"
df_manual.drop("marital.status", axis = 1, inplace=True)
df_manual.head()


# ## Encode "occupation"

# In[ ]:


dummy = pd.get_dummies(df_manual["occupation"])
dummy.head()


# In[ ]:


dummy.drop("Other-service", axis = 1, inplace=True)
# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"
df_manual.drop("occupation", axis = 1, inplace=True)
df_manual.head()


# ## Encode "relationship"

# In[ ]:


dummy = pd.get_dummies(df_manual["relationship"])
dummy.head()


# In[ ]:


dummy.drop("Other-relative", axis = 1, inplace=True)
# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"
df_manual.drop("relationship", axis = 1, inplace=True)
df_manual.head()


# ## Encode "race"

# In[ ]:


dummy = pd.get_dummies(df_manual["race"])
dummy.head()


# In[ ]:


dummy.drop("Other", axis = 1, inplace=True)
# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"
df_manual.drop("race", axis = 1, inplace=True)
df_manual.head()


# ## Encode "sex"

# In[ ]:


dummy = pd.get_dummies(df_manual["sex"])
dummy.head()


# In[ ]:


dummy.drop("Male", axis = 1, inplace=True)
dummy.rename(columns={ 'Female' : 'Female/Male'}, inplace = True)
# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)
# drop original column "fuel-type" from "df"
df_manual.drop("sex", axis = 1, inplace=True)
df_manual.head()


# ## Encode "native.country"

# In[ ]:


dummy = pd.get_dummies(df_manual["native.country"])
dummy.head()


# In[ ]:


# merge data frame "df" and "dummy_variable_1" 
df_manual = pd.concat([df_manual, dummy], axis=1)
# drop original column "fuel-type" from "df"
df_manual.drop("native.country", axis = 1, inplace=True)
df_manual.head()


# ## Encode "Income"  (Target Variable)
# **Income > 50K = 1, otherwise 0**

# In[ ]:


dummy = pd.get_dummies(df_manual["income"])
dummy.head()


# In[ ]:


dummy.rename(columns={ '>50K' : 'Income > 50K'}, inplace = True)
dummy.drop('<=50K', axis = 1, inplace = True)


# In[ ]:


df_manual = pd.concat([df_manual, dummy], axis=1)
df_manual.drop("income", axis = 1, inplace=True)
df_manual.head()


# ## Check for any "object" datatype  
# **object = string + numerical data**
# 
# Here we check for any remaining categorical variables.

# In[ ]:


df_manual.dtypes


# # Lets divide out dataframe into feature variable(X) and target variable(Y)

# In[ ]:


X = df_manual.iloc[:,:-1].values
y = df_manual["Income > 50K"].iloc[:].values


# ## After Encoding Categorical Data to Numeric Value

# In[ ]:


df_manual.describe(include = 'all')


# ## Data Standardization  
# **Data standardization is the process of rescaling one or more attributes so that they have a mean value of 0 and a standard deviation of 1.**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# ## See Target Variable Distribution

# In[ ]:


import seaborn as sns
sns.countplot(y)


# ### Here the observation is highly imbalanced because there are more observations where Income <= 50K than >50K

# ### Split Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # 3. Build Model

# ## Implement XGBoost (Gradient Boosting) Classifier Model

# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate = 0.1, n_estimators = 100)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Model Accuracy = {:.2f}%".format(accuracies.mean()* 100))


# In[ ]:


from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                               cmap = 'Dark2')
plt.show()


# In[ ]:


# import lightgbm as lgb
# d_train = lgb.Dataset(X_train, label = y_train)
# params = {}
# params['learning_rate'] = 0.01
# params['boosting_type'] = 'gbdt'
# params['objective'] = 'binary'
# params['metric'] = 'binary_logloss'
# params['sub_feature'] = 0.5
# params['num_leaves'] = 10
# params['min_data'] = 50
# params['max_depth'] = 10
# clf = lgb.train({},train_set = d_train)


# In[ ]:


# #Prediction
# y_pred=clf.predict(X_test)


# In[ ]:


# for i in range(0,y_pred.shape[0]):
#     if y_pred[i]>=0.5:       # setting threshold to .5
#        y_pred[i]=1
#     else:  
#        y_pred[i]=0


# ## Fixing Missing Data with ML Models

# ### 1. Take all the rows from the dataframe without any missing value and make a training dataset  
# ### 2. Build and Train a Model (Classifier / Regressor) with the Training Data  
# ### 3. Take all rows with missing values and predict for missing value using the trained model  
# ### 4. Fill the Null Values with predicted values 
# ### 5. Repeat above steps for all other values  

# ## Available Missing Values

# In[ ]:


missing_col = []
for column in missing_data.columns.values.tolist():
    if(missing_data[column].sum() > 0):
        print("Column: ",column)
        print("Missing Data: {} ({:.2f}%)".format(missing_data[column].sum(), (missing_data[column].sum() * 100/ len(df))))
        print("Data Type: ",df[column].dtypes)
        print("")
        missing_col.append(column)


# ## Replace "?" to NaN

# In[ ]:


df_dl_method = pd.read_csv("../input/adult.csv")
# replace "?" to NaN
df_dl_method.replace("?", np.nan, inplace = True)
df_dl_method.head(5)


# ## Let's make a dataframe without any Null Values

# In[ ]:


df_without_null = df_dl_method.dropna()


# ## Fix 'workclass'

# In[ ]:


# reset index, because we droped two rows
df_without_null.reset_index(drop = True, inplace = True)
df_without_null.drop(["occupation", "native.country"], axis = 1, inplace = True)
df_without_null.head()


# In[ ]:


def encoder(dataframe, col, drop_dummy_trap = ""):
    dummy = pd.get_dummies(dataframe[col])
    dataframe = pd.concat([dataframe, dummy], axis=1)
    if(len(drop_dummy_trap) != 0):
        # drop original column "fuel-type" from "df"
        dataframe.drop(drop_dummy_trap, axis = 1, inplace=True)
    dataframe.drop(col, axis = 1, inplace = True)
    return dataframe


# In[ ]:


df_without_null = encoder(dataframe = df_without_null, col = "education", drop_dummy_trap = "Some-college")
#df_without_null = encoder(df = df_without_null, col = "occupation", drop_dummy_trap = "Other-service")
df_without_null = encoder(dataframe = df_without_null, col = "marital.status", drop_dummy_trap = "Never-married")
df_without_null = encoder(dataframe = df_without_null, col = "relationship", drop_dummy_trap = "Other-relative")
df_without_null = encoder(dataframe = df_without_null, col = "race", drop_dummy_trap = "Other")
df_without_null = encoder(dataframe = df_without_null, col = "sex", drop_dummy_trap = "Male")
df_without_null = encoder(dataframe = df_without_null, col = "income", drop_dummy_trap = "<=50K")


# In[ ]:


X_train = df_without_null.drop(["workclass"], axis = 1).iloc[:].values
y_train = df_without_null.iloc[:,1].values


# In[ ]:


df_test=df_dl_method.loc[pd.isnull(df_dl_method["workclass"])]
df_test.head()


# In[ ]:


df_test.drop(["workclass", "occupation", "native.country"], axis = 1, inplace = True)
df_test.head()


# In[ ]:


df_test = encoder(df_test, col = "education", drop_dummy_trap = "Some-college")
#df_test = encoder(df_test, col = "occupation", drop_dummy_trap = "Other-service")
df_test = encoder(df_test, col = "marital.status", drop_dummy_trap = "Never-married")
df_test = encoder(df_test, col = "relationship", drop_dummy_trap = "Other-relative")
df_test = encoder(df_test, col = "race", drop_dummy_trap = "Other")
df_test = encoder(df_test, col = "sex", drop_dummy_trap = "Male")
#df_test = encoder(df_test, col = "native.country", drop_dummy_trap = "")
df_test = encoder(df_test, col = "income", drop_dummy_trap = "<=50K")


# In[ ]:


X_test = df_test.iloc[:].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# In[ ]:


y_train[:] = labelencoder.fit_transform(y_train[:])
y_train


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Predict Missing Values using XGBoost

# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate = 0.1, n_estimators = 100)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


decode = dict(zip(labelencoder.transform(labelencoder.classes_), labelencoder.classes_))


# In[ ]:


for i in range(0,y_pred.shape[0]):
    y_pred[i] = decode[y_pred[i]]
y_pred    


# In[ ]:


df_dl_method.head()


# ## Fill the Missing Values of "workclass" using our Predicted Values

# In[ ]:


fill = pd.DataFrame(y_pred, columns = ["workclass"])


# In[ ]:


j = 0
for i in range(0, df_dl_method.shape[0]):
    if(pd.isnull(df_dl_method.workclass[i])):
        df_dl_method.workclass[i] = y_pred[j]
        j = j+1


# In[ ]:


df_dl_method.workclass.isnull().sum()


# In[ ]:


df_viz = pd.read_csv("../input/adult.csv")
# replace "?" to NaN
df_viz.replace("?", np.nan, inplace = True)


# In[ ]:


fig1 = plt.figure(figsize=(20,5))
i = 1
column = "workclass"

bad = df_viz[column].isnull().sum()
good = len(df_viz) - df_viz[column].isnull().sum()
x = [bad, good]
labels = ["Missing Data", "Good Data"]
explode = (0.1, 0)
ax = fig1.add_subplot(1,2,i)
ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})
centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal')
ax.set_title(column + "(before)", fontsize = 25) 
i = i+1

bad = df_dl_method[column].isnull().sum()
good = len(df) - df_dl_method[column].isnull().sum()
x = [bad, good]
labels = ["Missing Data", "Good Data"]
explode = (0.1, 0)
ax = fig1.add_subplot(1,2,i)
ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})
centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal')
ax.set_title(column + "(after)", fontsize = 25)


plt.tight_layout()
plt.show()


# # Thank You!
