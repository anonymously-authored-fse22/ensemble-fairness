#!/usr/bin/env python
# coding: utf-8

# ## Detection of Adult Census Income with Machine Learning & Deep Learning
# ![census.PNG](attachment:536303b8-cb98-46d7-af88-50ab092c88ca.PNG)
# ###### Dataset information:
# 
# - This data was extracted from the [1994 Census bureau database](https://www.census.gov/en.html) by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year.
# 
# The dataset can be found on the `` Kaggle`` platform at the link below:
# 
# https://www.kaggle.com/uciml/adult-census-income

# ## 1. Imports from libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense


# ## 2. Starting...

# In[ ]:


df = pd.read_csv("../input/adult-census-income/adult.csv")


# In[ ]:


df.head(3)


# We have 14 columns present in the dataset provided, 13 of which are characteristic variables (input data) and one of them is a target variable (which we want our model to be able to predict).
# 
# The characteristic variables are:
# 
#      age - The age of the user
#      Workclass - User Profession
#      final-weight - Final user income
#      education - user education
#      education-num - user education ID
#      marital-status - user's civil status
#      occupation - User occupation
#      relationship - User relationship
#      race - user race
#      Fri - User Gender
#      capital-gain - Capital gained
#      capital-loss - lost capital
#      hour-per-week - Hours per week
#      native-country - hometown
# 
# The target variable is:
# 
#      income - a *binary* type that indicates the user's income:
#              <=50k - User with income less than or equal to 50000
#               >50k - User with income over 50000

# In[ ]:


df.info()


# Note that there are variables of type ``float64`` ("decimal" numbers), variables of type ``int64`` (integers) and variables of type ``object`` (in this case they are *strings*, or text) .
# 
# Since most supervised statistical learning algorithms only accept numerical values as input, it is necessary then to preprocess variables of type "object" before using this dataset as input for training a model.

# The ``describe()`` function generates a lot of information about numeric variables that can also be useful:

# In[ ]:


df.describe()


# ###### Defining the features of our model
# For this, we will create the variable X that will receive the characteristic variables of our model, and the variable y that will receive the target variable of our model.
# 
# We will also remove the 'clientid' columns that will not be relevant in our model.

# In[ ]:


df.columns


# In[ ]:


# Definition of the columns that will be features (note that the column 'clientid' is not present)
features = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num',
    'marital.status', 'occupation', 'relationship', 'race', 'sex',
    'capital.gain', 'capital.loss', 'hours.per.week', 'native.country'
]

# Preparation of arguments for ``scikit-learn`` library methods
X = df[features].values


# ###### Categorical variable handling
# 
# As mentioned before, computers aren't good with "categorical" variables (or strings).
# 
# Given a column with categorical variable, what we can do is encoding that column into multiple columns containing binary variables. This process is called "one-hot-encoding" or "dummy encoding".

# In[ ]:


from sklearn.preprocessing import LabelEncoder

lbp = LabelEncoder()


# In[ ]:


# Part of transforming categorical to integer

X[:, 1] = lbp.fit_transform(X[:, 1])

X[:, 3] = lbp.fit_transform(X[:, 3])

X[:, 5] = lbp.fit_transform(X[:, 5])

X[:, 6] = lbp.fit_transform(X[:, 6])

X[:, 7] = lbp.fit_transform(X[:, 7])
X
X[:, 8] = lbp.fit_transform(X[:, 8])

X[:, 9] = lbp.fit_transform(X[:, 9])

X[:, 13] = lbp.fit_transform(X[:, 13])


# ###### Checking if the data has changed from the first three lines

# In[ ]:


df.head(3)


# In[ ]:


X[0:3]


# In[ ]:


# converting the Label to a numeric format for testing later...
LE = LabelEncoder()

y = LE.fit_transform(df["income"])


# ###### Scaling of numerical data
# As we can see in the data there is a big difference between high numbers and low numbers, so we must scale the data to keep them on the same scale.

# In[ ]:


scaler = StandardScaler()


# In[ ]:


X = scaler.fit_transform(X)
X


# ## 3. Dividing into training and testing sets
# Now we need to convert our data into training and testing sets. We will use 75% as our training data and test our model on the remaining 25% with Scikit-learn's train_test_split function.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## 4. Creation of models

# ###### 1. Naive Bayes
# The Naive Bayes algorithm is a simple classification algorithm that uses historical data to predict the classification of new data. It works by calculating the probability of an event occurring given that another event has already occurred.

# In[ ]:


nb = GaussianNB()


# In[ ]:


nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)


# In[ ]:


y_pred_nb = nb.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
roc_nb = roc_auc_score(y_test, y_pred_nb)


# In[ ]:


print(classification_report(y_test, y_pred_nb))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_nb),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ###### 2. Decision Tree
# The Decision Tree algorithm are statistical models that use supervised training for data classification and prediction. These models use the divide-and-conquer strategy: a complex problem is decomposed into simpler sub-problems and recursively this technique is applied to each sub. -problem

# In[ ]:


dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)


# In[ ]:


dt.fit(X_train, y_train)


# In[ ]:


y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_dt = roc_auc_score(y_test, y_pred_dt)


# In[ ]:


print(classification_report(y_test, y_pred_dt))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_dt),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ###### 3. Random Forest
# The Random Forest algorithm creates a forest in a random way, creating several decision trees and combining them, each tree tries to estimate a ranking and this is called as “vote”, thus to obtain a more accurate and more stable prediction.

# In[ ]:


rf = RandomForestClassifier(n_estimators = 40, criterion= 'entropy', random_state= 0)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_pred_rf)


# In[ ]:


print(classification_report(y_test, y_pred_rf))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_rf),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ###### 4. kNN
# The KNN or k-nearest neighbor algorithm is a very simple machine learning algorithm. It uses some sort of similarity measure to tell which class the new data falls into, in which case we'll use 5 nearest neighbors.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
roc_knn = roc_auc_score(y_test, y_pred_knn)


# In[ ]:


print(classification_report(y_test, y_pred_knn))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_dt),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ###### 5. Logistic Regression
# Logistic regression algorithm is used where a discrete output is expected, (eg Predict whether a user is a good or bad payer). Typically, logistic regression uses some function to squeeze values into a given range.

# In[ ]:


rl = LogisticRegression(random_state=0)


# In[ ]:


rl.fit(X_test, y_test)


# In[ ]:


y_pred_rl = rl.predict(X_test)

accuracy_rl = accuracy_score(y_test, y_pred_rl)
recall_rl = recall_score(y_test, y_pred_rl)
precision_rl = precision_score(y_test, y_pred_rl)
f1_rl = f1_score(y_test, y_pred_rl)
roc_rl = roc_auc_score(y_test, y_pred_rl)


# In[ ]:


print(classification_report(y_test, y_pred_rl))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_rl),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ###### 6. SVM (Support Vector Machines)
# The SVM algorithm separates data points using a line. This line is chosen in such a way that it will be the most important of the closest data points in 2 categories.

# In[ ]:


svm = SVC(kernel = 'linear', random_state=0)


# In[ ]:


svm.fit(X_train, y_train)


# In[ ]:


y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_svm = roc_auc_score(y_test, y_pred_svm)


# In[ ]:


print(classification_report(y_test, y_pred_svm))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_svm),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ###### 7. Neural networks
# The purpose of the Neural Networks algorithm is to imitate the nervous system of humans in the learning process, it is inspired by biological neural networks

# In[ ]:


rn = MLPClassifier(verbose = True, max_iter= 250, tol = 0.000010)


# In[ ]:


rn.fit(X_train, y_train)


# In[ ]:


y_pred_rn = rn.predict(X_test)

accuracy_rn = accuracy_score(y_test, y_pred_rn)
recall_rn = recall_score(y_test, y_pred_rn)
precision_rn = precision_score(y_test, y_pred_rn)
f1_rn = f1_score(y_test, y_pred_rn)
roc_rn = roc_auc_score(y_test, y_pred_rn)


# In[ ]:


print(classification_report(y_test, y_pred_rn))


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred_rn),annot = True, fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# ## 5. Viewing the results of all models

# In[ ]:


models = [('Naive Bayes', accuracy_nb, recall_nb, precision_nb, f1_nb, roc_nb),
          ('Decision Tree', accuracy_dt, recall_dt, precision_dt, f1_dt, roc_dt),
          ('Random Forest', accuracy_rf, recall_rf, precision_rf, f1_rf, roc_rf),
          ('kNN', accuracy_knn, recall_knn, precision_knn, f1_knn, roc_knn),
          ('Logistic Regression', accuracy_rl, recall_rl, precision_rl, f1_rl, roc_rl),
          ('SVM', accuracy_svm, recall_svm, precision_svm, f1_svm, roc_svm),
          ('Neural Networks', accuracy_rn, recall_rn, precision_rn, f1_rn, roc_rn)]

df_all_models = pd.DataFrame(models, columns = ['Model', 'Accuracy (%)', 'Recall (%)', 'Precision (%)', 'F1 (%)', 'AUC'])

df_all_models


# In[ ]:


plt.style.use("dark_background")

plt.subplots(figsize=(12, 10))
sns.barplot(y = df_all_models['Accuracy (%)'], x = df_all_models['Model'], palette = 'icefire')
plt.xlabel("Models")
plt.title('Accuracy')
plt.show()


# In[ ]:


r_probs = [0 for _ in range(len(y_test))]
r_auc = roc_auc_score(y_test, r_probs)
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)

fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_nb)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn)
fpr_rl, tpr_rl, _ = roc_curve(y_test, y_pred_rl)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
fpr_rn, tpr_rn, _ = roc_curve(y_test, y_pred_rn)


# In[ ]:


sns.set_style('darkgrid')

plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)

plt.plot(fpr_nb, tpr_nb, marker='.', label='Naive Bayes (AUROC = %0.3f)' % roc_nb)
plt.plot(fpr_dt, tpr_dt, marker='.', label='Decision Tree (AUROC = %0.3f)' % roc_dt)
plt.plot(fpr_rf, tpr_rf, marker='.', label='Random Forest (AUROC = %0.3f)' % roc_rf)
plt.plot(fpr_knn, tpr_knn, marker='.', label='kNN (AUROC = %0.3f)' % roc_knn)
plt.plot(fpr_rl, tpr_rl, marker='.', label='Logistic Regression (AUROC = %0.3f)' % roc_rl)
plt.plot(fpr_svm, tpr_svm, marker='.', label='SVM (AUROC = %0.3f)' % roc_svm)
plt.plot(fpr_rn, tpr_rn, marker='.', label='Neural Networks (AUROC = %0.3f)' % roc_rn)

plt.title('ROC Plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()

