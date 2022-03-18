#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# **Portugese Bank term Deposit marketing campaign DataSet description:**
# 
# *The data is related with direct marketing campaigns of a Portuguese banking institution.
# The different marketing campaigns were based on phone calls.
# Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.*

# **Features Description:**
# 
# - Features 1 to 4 is information on customer's age, job, marital status and educational qualification.
# - Features 5 to 8 is information on customer's credit information - Loan default status - housing, personal loan and A/c Bal value
# - Features 9 to 12 is information on customer contact in current campaign - contact mode, day of week, last contact month, duration in sec. , campaign - num of contacts during campaign.
# - Duration variable to be removed(or weightage to be reduced) as the output of the call determines the outcome for that specific customer and our model should be good in predicting with customer profile rather than duration of call with specific customer.
# - Features 13 to 15 is information on previous campaign: pdays- num of days of contact from previous campaign;
#   previous - num of contacts performed before this campaign; poutcome- outcome of previous marketing campaign
# - Target variable 'y' conatins the values of the current campaign outcome.

# In[ ]:


X = pd.read_csv('./X_Bank_portugese_dataset.csv')


# In[ ]:


X.columns


# In[ ]:


X.shape


# In[ ]:


y = pd.read_csv('./y_Bank_portugese_dataset.csv')


# In[ ]:


y.value_counts()


# In[ ]:


y = np.array(y)


# In[ ]:


y = y.ravel()


# In[ ]:


y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify = y)


# In[ ]:


print( X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ***Applying Random Forest Classifier for this dataset and study the model explainability with SHAP values and dependance plots.***

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators = 150).fit(X_train, y_train)


# * **Applying *permutation importance method* to compute the feature importances**
# * *This method works as follows:*
# *    * Alogrithm selects a feature from a single row
# *    * Permutates over the range of values available from the dataset for that feature and calculates the impact on the * target variable.
# *    * And same is repeated row-wise and for all features.

# In[ ]:


from eli5.sklearn import PermutationImportance
import eli5


# In[ ]:


perm = PermutationImportance(rf_clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# **Applying SHAP summary report to calculate Feature values and depndancy (interaction) plots**

# In[ ]:


import shap


# In[ ]:


explainer = shap.TreeExplainer(rf_clf)

shap_values = explainer.shap_values(X_test)


# In[ ]:


shap_values


# In[ ]:


shap.summary_plot(shap_values[1], X_test)


# * From above shap value summary plot, below are some of the inferences about dataset on *target variable (make new deposit in current campaign y/n)*:
# * *Ranked Feature Permutation-importance wise:*
# *    
# *    **Feature**:
# *     **1.** **poutcome_enc**: *Previous campaign outcome for deposit conversion - unknown/failure/other/success*
# *             * range of the **purple-pink(feat value higher side) dots** in shap value is from **-0.1 to 0.6**
# *             * distribution of those dots are dense near -0.2 to 0 and sparse in  0.3- 0.5 range of shap value
# *             * poutcome feature on most of the data have little -ve impact and more +ve on target variable
# *             * previous campaign outcome has **more influence in current campaign deposit converion** from customers than other features in the dataset.
# *             * permutation importance value: *0.0228 +/- 0.0007*
# *
# *    **2.** **job_enc**: *Job profile - managment/entrpreuner/technician/blue-collar/other*
# *           * range of **feat values in higher side: -0.2 to 0.1** (with some outliers)
# *           * distribution : dense in -0.1 to 0.1 range & sparse towards -0.2
# *           * job profile of the customers have **lesser impact on outome**(closer to 0 SHAP val) compared mode of poutcome feature.
# *           * permutation importance value: *0.0029 +/- 0.0018*
# *
# *   **3.** **contact_enc**: *Contact - mode of contact during the campaign - cell/ telephone / other*
# *           * range of **feat values in higher side: -0.2 to 0.25**
# *           * distribution : dense in -0.05 to 0.1 range & sparse towards 0.2
# *           * mode of contact has **less impact value on the outcome** as poutcome and is biased towards +ve impact on outcome.
# *           * permutation importance value: *0.0023 +/- 0.0014*
# *
# *   **4.** **housing_enc**: *Housing loan - Yes / No*
# *           * range of feat values in higher side:**0 to -0.2**
# *           * distribution: dense in -0.1 to 0 range & sparse towards -0.15
# *           * Housing loan status has **equal impact value on the outcome** as contact_enc. but on the -ve side.
# *           * permutation importance value: *0.0023 +/- 0.0010*
# *
# *   **5.** **educ_ord**: *Education qualification - tertiary, secondary, primary, other*
# *           * range of feat values in higher side: **-0.15 to 0.2**
# *           * distribution: dense near 0 (shap value) and sparse when away from centre
# *           * education qualification has **almost equal impact on the outcome** as job profile
# *           * permutation importance value: *0.0016 +/- 0.0011*
# *
# *   **6.** **marital_enc**:  *Marital status - Single/Married/Divorced*
# *           - range of feat values in higher side: **-0.2 to 0.1**
# *           - distribution: dense in -0.1 to 0 range & sparse towards -0.15
# *           - marital_enc has **similar impact value on the outcome** compared to Educ_ord.
# *           * permutation importance value: *0.0013 +/- 0.0010*
# *
# *    **7.** **loan_enc**: *personal loan - Yes/No*
# *           - range of feat values in higher side: **-0.2 to 0.1**
# *           - distribution: sparse throughout the range
# *           - loan_enc has **lesser impact on outcome** compared to marital_enc
# *           * permutation importance value: *0.0006 +/- 0.0006*
# *
# *    **8.** **default_enc**: *Loan default - Yes/No*
# *           - range of feat values in higher side: **-0.1 to 0.1**
# *           - distribution: sparse throughout the range
# *           - deafult_enc has **less impact on outcome** compared to other features
# *           * permutation importance value: *0.0003 +/- 0.0003*

# * **Further we will explore the interactions between the top features identified**

# **(Poutcome vs Job category on Target variable)**

# In[ ]:


shap.dependence_plot('job_enc', shap_values[1], X_test, interaction_index="poutcome_enc")


# * Below summary is inferred from interaction plot of top 2 important features *(Poutcome vs Job category on Target variable)* :
# *    * Interdependancy of the top 2 features is found to be **less across all categories of Job  type** (range 0.15 to -0.1)
# *    * Unemployed and Low income categories(1 - 3)* have **more positive impact on target variable** compared to *Higher income categories(4-6)* supported by *    * previous outcome feature values*. (more pink/purple dots in lower side for categ 4-6).

# **(Poutcome vs (*Higher interaction feat calculated by algorithm*) on Target variable)**

# In[ ]:


shap.dependence_plot('poutcome_enc', shap_values[1], X_test, interaction_index='auto')


# * Below summary is inferred from *interaction plot of **important feature and higher interaction feature** chosen by algorithm (Poutcome vs housing_enc on Target variable)* :
# *  *  Previous outcome ***success rate has high influence on target variable***

# *From **interaction feature *housing_enc*** - we can infer that the ***customers with housing loan(value 2) are less likely to make new deposits in the bank compared to without housing loan (value 1).***

# **This notebook demonstrates the application of *Permutation importance method from Eli5 library* for calculating feature importances based on model fit.**
#  **Also, the various features of *SHAP library - Summary plot, dependency plot* have been applied to explore and understand the dataset's feature impact and feature interdependencies in contributing to the prediction of the outcome (target variable).**

# 
