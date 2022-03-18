#!/usr/bin/env python
# coding: utf-8

# Problem statement:
# 
# Given the data of citizens, we have to understand the data extensively, perform EDA to better understand the features and predict if an individual's income is above 50k or no by using the relevant features
# 

# Importing the necessary libraries

# In[ ]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# Visualizations
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[ ]:


df= pd.read_csv('../input/adult-census-income/adult.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


#Mapping binary values to the expected output

df['income']=df['income'].map({'<=50K': 0, '>50K': 1})


# In[ ]:


#Replacing question marks in dataset with null values

df.replace('?',np.nan )


# In[ ]:


#Finding what percentage of data is missing from the dataset

total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[ ]:


#Since a very small amount of data is missing, we can replace the null values with the mode of each column

df['occupation'].describe()


# In[ ]:


#Since mode is Prof-specialty, replacing null values with it

df['occupation'] = df['occupation'].fillna('Prof-specialty')


# In[ ]:


df['workclass'].describe()


# In[ ]:


#Since mode is Private, replacing null values with it

df['workclass'] = df['workclass'].fillna('Private')


# In[ ]:


df['native.country'].describe()


# In[ ]:


#Since mode is United-States, replacing null values with it

df['native.country'] = df['native.country'].fillna('United-States')


# In[ ]:


#Mean, Median, Minimum , Maximum values etc can be found

df.describe()


# In[ ]:


df.describe(include=["O"])


# In[ ]:


#Visualizing the numerical features of the dataset using histograms to analyze the distribution of those features in the dataset

rcParams['figure.figsize'] = 12, 12
df[['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']].hist()

#Can visualise that data such as capital gain, capitaln loss, fnlwgt is right skewed an other columns can be grouped for better visualisation


# In[ ]:


#Ploting the correlation between the output(income) and individual features

plt.matshow(df.corr())
plt.colorbar()
plt.xticks(np.arange(len(df.corr().columns)), df.corr().columns.values, rotation = 45) 
plt.yticks(np.arange(len(df.corr().columns)), df.corr().columns.values) 
for (i, j), corr in np.ndenumerate(df.corr()):
    plt.text(j, i, '{:0.1f}'.format(corr), ha='center', va='center', color='white', fontsize=14)


# In[ ]:


#Since it has 0 correlation, it can be dropped

df.drop(['fnlwgt'], axis = 1, inplace = True)


# In[ ]:


dataset=df.copy()


# In[ ]:


#Distributing Age column in 3 significant parts and plotting it corresponding to the output feature(income)

dataset['age'] = pd.cut(dataset['age'], bins = [0, 25, 50, 100], labels = ['Young', 'Adult', 'Old'])


# In[ ]:


sns.countplot(x = 'income', hue = 'age', data = dataset)


# In[ ]:


#Capital gain and capital loss can be combined and transformed into a feature capital difference. Plotting the new feature corresponding to income

dataset['Capital Diff'] = dataset['capital.gain'] - dataset['capital.loss']
dataset.drop(['capital.gain'], axis = 1, inplace = True)
dataset.drop(['capital.loss'], axis = 1, inplace = True)


# In[ ]:


dataset['Capital Diff'] = pd.cut(dataset['Capital Diff'], bins = [-5000, 5000, 100000], labels = ['Minor', 'Major'])
sns.countplot(x = 'income', hue = 'Capital Diff', data = dataset)


# In[ ]:


#Dividing hours of week in 3 major range and plotting it corresponding to the income

dataset['Hours per Week'] = pd.cut(dataset['hours.per.week'], 
                                   bins = [0, 30, 40, 100], 
                                   labels = ['Lesser Hours', 'Normal Hours', 'Extra Hours'])


# In[ ]:


sns.countplot(x = 'income', hue = 'Hours per Week', data = dataset)


# In[ ]:


#Plotting workclass corresponding to the income

sns.countplot(x = 'income', hue = 'workclass', data = dataset)


# In[ ]:


#Plot of education corresponding to income

sns.countplot(x = 'income', hue = 'education', data = dataset)


# In[ ]:


#Combining the lower grades of education together

df.drop(['education.num'], axis = 1, inplace = True)
df['education'].replace(['11th', '9th', '7th-8th', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
                             ' School', inplace = True)
df['education'].value_counts()


# In[ ]:


sns.countplot(x = 'income', hue = 'education', data = df)


# In[ ]:


#Plot of occupation corresponding to the income

plt.xticks(rotation = 45)
sns.countplot(x = 'income', hue = 'occupation', data = dataset)


# In[ ]:


sns.countplot(x = 'income', hue = 'race', data = dataset)


# In[ ]:


#Since majority of race is white, the rest of races can be combined together to form a new group

df['race'].unique()
df['race'].replace(['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],' Other', inplace = True)


# In[ ]:


#Plot of sex corresponding to income

sns.countplot(x = 'income', hue = 'sex', data = dataset)


# In[ ]:


count = dataset['native.country'].value_counts()
count


# In[ ]:


#Plot of Country corresponding to income


plt.bar(count.index, count.values)
plt.xlabel('Countries')
plt.ylabel('Count')
plt.title('Count from each Country')


# In[ ]:


#Combining all other into one class

countries = np.array(dataset['native.country'].unique())
countries = np.delete(countries, 0)


# In[ ]:


dataset['native.country'].replace(countries, 'Other', inplace = True)
df['native.country'].replace(countries, 'Other', inplace = True)


# In[ ]:


sns.countplot(x = 'native.country', hue = 'income', data = dataset)


# In[ ]:


#Splitting the data set into features and outcome

X = df.drop(['income'], axis=1)
Y = df['income']


# In[ ]:


X.head()


# In[ ]:


#Splitting the data into test data and training data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[ ]:


from sklearn import preprocessing

categorical = ['workclass','education', 'marital.status', 'occupation', 'relationship','race', 'sex','native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[ ]:


#Using StandardScalar to normalise the dataset

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[ ]:


X_train.head()


# In[ ]:


#Applying the random forest algorithm

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Applying the Logistic Regression algorithm

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# In[ ]:


# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train) 
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Applying the GaussianNB algorithm

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Applying the Support Vector Machine algorithm

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Applying the Decision Tree algorithm

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Plotting the accuracy of the used algorithms to find the best fit

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(7)


# In[ ]:


#Finding significance of each feature in t5he best fit model

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[ ]:


#Plotting the significance of each feautre

importances.plot.bar()


# In[ ]:


#Since they hardly have any significance, can drop these columns to avoid overfitting

df  = df.drop("sex", axis=1)
df  = df.drop("race", axis=1)
df  = df.drop("native.country", axis=1)


# In[ ]:


#The accuracy remains the same even after dropping the columns

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# Thus, in this project, we cleaned the data, carried out EDA extensively and understood all the parameters in this process and successfully applied multiple classification algorithms to predict the required output!
