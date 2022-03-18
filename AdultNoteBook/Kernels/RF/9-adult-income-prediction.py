#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income dataset

# **I hope you find this kernel useful**
# # Your UPVOTES would be highly appreciated

# The dataset named Adult Census Income is available in kaggle and UCI repository. This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics).

# **The prediction task is to determine whether a person makes over $50K a year or not.**

# The dataset provides 14 input variables that are a mixture of categorical, ordinal, and numerical data types. The complete list of variables is as follows:
# 
# - age: continuous.
# - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - fnlwgt: continuous.
# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - sex: Female, Male.
# - capital-gain: continuous.
# - capital-loss: continuous.
# - hours-per-week: continuous.
# - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# - salary: >50K,<=50K[](http://)

# ### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
sns.set(style='white', context='notebook', palette='deep')

# Import scikit_learn module for the algorithm/model: Linear Regression
from sklearn.linear_model import LogisticRegression
# Import scikit_learn module to split the dataset into train.test sub-datasets
from sklearn.model_selection import train_test_split 

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# import the metrics class
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# ### Import dataset

# In[ ]:


df = pd.read_csv("../input/adult-census-income/adult.csv")
df.head()


# ## Exploratory data analysis with Pandas

# Let’s have a look at data dimensionality, feature names, and feature types

# In[ ]:


print(df.shape)


# From the output, we can see that the table contains 32561 rows and 15 columns.
# 
# 

# Now let's try printing out column names using columns:

# In[ ]:


print(df.columns)


# We can use the info() method to output some general information about the dataframe:

# In[ ]:


print(df.info())


# The describe method shows basic statistical characteristics of each numerical feature (int64 and float64 types): number of non-missing values, mean, standard deviation, range, median, 0.25 and 0.75 quartiles.

# In[ ]:


df.describe()


# Let's see statistics on non-numerical features.

# In[ ]:


df.describe(include=['object'])


# Let's check the repartition between male and female

# In[ ]:


df.sex.value_counts()


# In[ ]:


sns.countplot(x="sex", data=df)


# Let's check the race's repartition

# In[ ]:


df.race.value_counts()


# In[ ]:


sns.countplot(x="race", data=df)


# #### Reformating the target columns

# In[ ]:


df['income']=df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
df.head(4)


# *Let's have a look at our target*

# In[ ]:


df.income.value_counts()


# In[ ]:


sns.countplot(x="income", data=df)


# We have 24720 people with incomes below 50k and 7841 with incomes above 50k

# In[ ]:


# Identify Numeric features
numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']

# Identify Categorical features
cat_features = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']


# #### Analyzing numerical features

# Age 

# In[ ]:



df.age.plot.hist(grid=True)


# In[ ]:


df.fnlwgt.plot.hist(grid=True)


# In[ ]:


df[numeric_features].hist()


# #### Correlation

# In[ ]:


cor_mat = sns.heatmap(df[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()


# In[ ]:


df['education.num'].value_counts()


# In[ ]:


sns.countplot(x="education.num", data=df)


# Let's have a look at the income according to the gender

# In[ ]:


sns.countplot(x="sex", hue="income", data=df)


# #### Exploring Education Num vs Income

# In[ ]:


ed_inc= sns.factorplot(x="education.num",y="income",data=df,kind="bar",size = 6,palette = "muted")
ed_inc.despine(left=True)
ed_inc = ed_inc.set_ylabels(">50K probability")


# #### Exploring Hours Per Week vs Income
# 

# In[ ]:


hour_inc  = sns.factorplot(x="hours.per.week",y="income",data=df,kind="bar",size = 6,palette = "muted")
hour_inc.despine(left=True)
hour_inc = hour_inc.set_ylabels(">50K probability")


# #### Exploring Relationship vs Income

# In[ ]:


rel_inc= sns.factorplot(x="relationship",y="income",data=df,kind="bar", size = 6 ,
palette = "muted")
rel_inc.despine(left=True)
rel_inc = rel_inc.set_ylabels("Income >50K Probability")


# #### Checking missing values

# In[ ]:


df.isna().sum()


# In[ ]:


# Fill Missing Category Entries
df["workclass"] = df["workclass"].fillna("X")
df["occupation"] = df["occupation"].fillna("X")
df["native.country"] = df["native.country"].fillna("United-States")

# Confirm All Missing Data is Handled
df.isnull().sum()


# Taking a glance at the data provided, we can see that there are some special characters in the data like ‘?’. So, let’s get the count of special characters present in the data.

# In[ ]:


#Finding the special characters in the data frame 
df.isin(['?']).sum(axis=0)


# ## Feature engineering

# In[ ]:


# code will replace the special character to nan and then drop the columns 
df['native.country'] = df['native.country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)


# In[ ]:


df.isna().sum()


# We will make a copy of the dataset in order to deal with this issue

# In[ ]:


print(df.shape)


# In[ ]:


df_new = df.copy()


# In[ ]:


#dropping the NaN rows now 
df_new.dropna(how='any',inplace=True)


# In[ ]:


print(df_new.shape)


# Machine Learning model requires input data in numerical notations to extract patterns from it and make predictions. But, not all the data provided in our source dataset is numerical. Some of the data provided are Categorical data like WorkClass, Education, Marital-Status, Occupation, Relationship, etc. we need to convert these into numerical notations.
# Here data is nothing but a feature that our model uses as an input. So, we perform Feature Engineering on our data to create meaningful numerical data out of the source dataset.
# 

# **value_counts for categorical features**

# In[ ]:


for c in df_new[cat_features]:
    print ("---- %s ---" % c)
    print (df[c].value_counts())


# Here we ran a for loop over all the columns using the .value_counts() function of Pandas which gets us the count of unique values. We can see that some of the data provided are unique like the ‘workclass’ attribute which has only 7 distinct values and some columns have a lot of distinct values.

# *Printing the number of uniques values in each categories*

# In[ ]:


for i in df_new:
    print ("---- %s ---" % i)
    print (df[i].nunique())


# In[ ]:


#dropping based on uniquness of data from the dataset 
df_new.drop(['age', 'fnlwgt', 'capital.gain','capital.loss', 'native.country','education.num'], axis=1, inplace=True)


# Printing the new columns

# In[ ]:


list(df_new.columns)


# In[ ]:


df_new.shape


# In[ ]:


df_new.dtypes


# ### Handling categorical features

# We have a lot of different ways to handle this kind of issue. Let's use the map function, we can convert all the other categorical data in the dataset to numerical data.
# 

# In[ ]:


#gender
df_new['sex'] = df_new['sex'].map({'Male': 0, 'Female': 1}).astype(int)
df_new['race'] = df_new['race'].map({'Black': 0, 'Asian-Pac-Islander': 1,'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
df_new['marital.status'] = df_new['marital.status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
df_new['workclass']= df_new['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
df_new['relationship'] = df_new['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)
df_new['education']= df_new['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)
df_new['occupation'] = df_new['occupation'].map({ 'Farming-fishing': 0, 'Tech-support': 1, 'Adm-clerical': 2, 'Handlers-cleaners': 3, 
 'Prof-specialty': 4,'Machine-op-inspct': 5, 'Exec-managerial': 6,'Priv-house-serv': 7,'Craft-repair': 8,'Sales': 9, 'Transport-moving': 10, 'Armed-Forces': 11, 'Other-service': 12,'Protective-serv':13}).astype(int)


# In[ ]:


df_new.race.value_counts()


# In[ ]:


df_new.sex.value_counts()


# In[ ]:


df_new['marital.status'].value_counts()


# In[ ]:


df_new['workclass'].value_counts()


# In[ ]:


df_new['education'].value_counts()


# In[ ]:


df_new['occupation'].value_counts()


# In[ ]:


df_new['relationship'].value_counts()


# In[ ]:


df_new.dtypes


# We have handled all categorical variables

# ### The new dataframe look like this

# In[ ]:


df_new.head(10)


# Now that we have just continuous variables we can analyze the correlation between them

# In[ ]:


#plotting a bar graph for Education against Income to see the co-relation between these columns 
df_new.groupby('education').income.mean().plot(kind='bar')


# Adults with an educational background of Prof-school (7) and Doctorate (10) will have a better income and it is likely possible that their income is higher than 50K.

# In[ ]:


df_new.groupby('occupation').income.mean().plot(kind='bar')


# Our data suggest that people with occupation Prof-specialty (5) and Exec-managerial (7) will have a better chance of earning an income of more than 50K.

# In[ ]:


df_new.groupby('sex').income.mean().plot(kind='bar')


# The gender bar chart provides us some useful insight into the data that Men (0) are more likely to have a higher income.

# In[ ]:


df_new.groupby('relationship').income.mean().plot(kind='bar')


# relationship chart shows us that wife (1) and husband (4) has a higher income. A married couple would most likely earn >50K.

# In[ ]:


df_new.groupby('race').income.mean().plot(kind='bar')


# As per the data, an Asian-Pac-Islander (1) or a white (3) have more chances of earning more than 50K.
# 

# In[ ]:


df_new.groupby('workclass').income.mean().plot(kind='bar')


# Self-emp-in (0), Federal-gov(2) workclass groups have a higher chance of earning more than 50K.

# ## Train test split

# In[ ]:


df_new.columns


# In[ ]:


X = df_new.drop('income',axis=1)
y = df_new.income


# In[ ]:


print("X shape : ", X.shape)
print("y shape : ", y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)


# In[ ]:


print("X_train shape : ", X_train.shape)
print("X_test shape : ", X_test.shape)
print("===============================")
print("y_train shape : ", y_train.shape)
print("y_test shape : ", y_test.shape)


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# ## Model Selection

# ### Logistic Regression

# Logistic Regression is one of the easiest and most commonly used supervised Machine learning algorithms for categorical classification. The basic fundamental concepts of Logistic Regression are easy to understand and can be used as a baseline algorithm for any binary (0 or 1) classification problem.

# In[ ]:


log_reg = LogisticRegression()
#Train our model with the training data
log_reg.fit(X_train, y_train)


# In[ ]:


#print our price predictions on our test data
pred_log = log_reg.predict(X_test)


# In[ ]:


accuracy_log_reg = metrics.accuracy_score(y_test, pred_log)


# In[ ]:


print(f"The accuracy of the model is {round(metrics.accuracy_score(y_test,pred_log),3)*100} %")


# In[ ]:


auc_log = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1])
print(f"The AUC Score  is {round(auc_log,3)*100} %")


# ### Naive Bayes

# A naive Bayes classifier assumes that the presence (or absence) of a particular feature of a class is unrelated to the presence (or absence) of any other feature, given the class variable. Basically, it’s “naive” because it makes assumptions that may or may not turn out to be correct.
# 

# In[ ]:


model = GaussianNB()

# Train the model using the training sets
gnb = model.fit(X_train,y_train)
predictions = gnb.predict(X_test)


# In[ ]:


accuracy_log_naive_bayes = metrics.accuracy_score(y_test, predictions)


# In[ ]:


#printing the accuracy values 
print(f"The accuracy of the model is {round(metrics.accuracy_score(y_test,predictions),3)*100} %")


# In[ ]:


auc_nb = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f"The AUC Score  is {round(auc_nb,3)*100} %")


# ### Decision Tree

# A decision tree is a branched flowchart showing multiple pathways for potential decisions and outcomes. The tree starts with what is called a decision node, which signifies that a decision must be made. From the decision node, a branch is created for each of the alternative choices under consideration.

# In[ ]:


clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=8,max_depth=10)


# In[ ]:


# Train Decision Tree Classifer
clf.fit(X_train,y_train)
prediction_clf = clf.predict(X_test)


# In[ ]:


accuracy_dec_tree = metrics.accuracy_score(y_test, prediction_clf)


# In[ ]:


print(f"The accuracy of the model is {round(metrics.accuracy_score(y_test,prediction_clf),3)*100} %")


# In[ ]:


auc_clf = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print(f"The AUC Score  is {round(auc_clf,3)*100} %")


# ### Random Forest
# 

# Random Forests are a combination of tree predictors where each tree depends on the values of a random vector sampled independently with the same distribution for all trees in the forest. The basic principle is that a group of “weak learners” can come together to form a “strong learner”.

# In[ ]:


rf=RandomForestClassifier(min_samples_split=30)
# Train the model using the training sets
rf.fit(X_train,y_train)
predictions_rf =rf.predict(X_test)


# In[ ]:


accuracy_rf = metrics.accuracy_score(y_test, predictions_rf)


# In[ ]:


print(f"The accuracy of the model is {round(metrics.accuracy_score(y_test,predictions_rf),3)*100} %")


# In[ ]:


auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"The AUC Score  is {round(auc_rf,3)*100} %")


# Let's resume all the result in a dataframe

# In[ ]:


model = ['LR',"NB","DT","RF"]


# In[ ]:


result = {'Model':['Logistic Regression',"Naive Bayes","Decision Tree","Random Forest"],
          'Accuracy':[accuracy_log_reg,accuracy_log_naive_bayes,accuracy_dec_tree,accuracy_rf],
         'AUC':[accuracy_log_reg,auc_nb,auc_clf,auc_rf]}


# In[ ]:


result_df = pd.DataFrame(data=result,index=model)
result_df


# **Our best model is the random forest.**
# 
# 
# 

# Let's dive into the random forest's metrics

# ### AUC Random Forest
# 

# AUC stands for Area under the curve. AUC gives the rate of successful classification by the logistic model. The AUC makes it easy to compare the ROC curve of one model to another.
# 

# In[ ]:


train_probs = rf.predict_proba(X_train)[:,1] 
probs = rf.predict_proba(X_test)[:, 1]
train_predictions = rf.predict(X_train)


# In[ ]:


def evaluate_model(y_pred, probs,train_predictions, train_probs):
   
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


# In[ ]:


evaluate_model(predictions_rf,probs,train_predictions,train_probs)


# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color 
    plt.figure(figsize = (5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
             plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Let's plot it out
cm = confusion_matrix(y_test, predictions_rf)
plot_confusion_matrix(cm, classes = ['0 - <50k', '1 - >50k'],
                      title = 'Confusion Matrix')


# Our best model is random forest !

# ### Feature importance of our best model

# In[ ]:


feature_importances = list(zip(X_train, rf.feature_importances_))
# Then sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked];


# In[ ]:


feature_names_8 = [i[0] for i in feature_importances_ranked[:8]]
y_ticks = np.arange(0, len(feature_names_8))
x_axis = [i[1] for i in feature_importances_ranked[:8]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_8, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 25)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()


# ## Conclusion

# As we said previously, we are more likely to have a salary above 50k if we are married and wife, and if we have a high level of education.

# # If you find this notebook useful then please upvote

# #### END
