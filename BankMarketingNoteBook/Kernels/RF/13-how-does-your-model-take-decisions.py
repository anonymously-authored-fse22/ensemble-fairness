#!/usr/bin/env python
# coding: utf-8

# **OPEN THE BLACK BOX - INTRODUCTION TO MODEL INTERPRETABILITY**
# > Why do we need it??
# > Lets say you have done your job as building the pipeline
# 1. Cleaned and processed messy data
# 2. Engineered Fancy new features
# 3. Selected the best model and tuned parametres
# 4. Trained Your Final Model
# 5. Got Great Performance On the Test Set
# > And Suddenly Your Boss or Any Non-Technical Guy asks -`Can you explain How Your Model Works??`
# > In this situation we should understand the importance of model interpretability->
# `Algorithms are everywhere,sometime automating important decisions that have an impact on people`
#    *   *Insurance* - model to predict the best price to charge the client
#    *    *Bank* - model to predict who should get loan or not
#    *    *Police* - model to predict who is most likely to buy a product
#    `In these types of situations we need to understand the underlying principle decisions that are taken by the algorithm to predict an outcome`
#    
#    
#     It is  also helpful to capture bias in the data, for example:
#     ->Predict employees' performance at a big company
#    Data Available
#    *past performance reviews of indiviual employees for the last 10 years*
#     
#     'What if the company tends to promote men more than women?'
#     The model will learn the bias, and predict that men are more likely to be performant'
#     
#    In these situations we need to understand the model behaviour and decisions that it is taking to predict and 
#    
#    So to interpret models we have several packages made available to us-
#   >[ELI5](https://github.com/TeamHG-Memex/eli5) -`Useful to debug skelearn-models and communicate with domain experts(usually used for white-box  models)`
#  
#    > [LIME](https://github.com/marcotcr/lime)-`Explains why a single datapoint was classified as a specific class(used for black box algorithms)`
#   
#   > [SHAP](https://github.com/slundberg/shap)-`Tree explainer(only for tree based models-used with scikit-learn,xgboost,lightgbm,catboost) and kernel explainer-(Model Agnostic explainer)`
#   
#   
#   
#  
#  `SO LETS GET STARTED` 
# 
#  
#       
# 
# 

# <a id="contents"></a>
#  # Contents
#  
# 1.[Import the necessary Libraries](#imports)<br>
#  
# 2.[Load dataset and process it](#dataloading)<br>
# 
# 3.[ELI5 to interpret white box models](#eli)<br>
#   > 3.a)[With a Logistic Regression](#e)<br>
#   3.b)[With a Decision Classifier](#d)<br>
#      
# 4.[Black Box Model Interpretation->>Random Forests](#rand)<br>
#   > 4.a)[Confidence based on Tree Variance](#int)<br>
#    4.b)[Permutation Importance(Why not to trust scikit learns' feature importance)](#it)<br>
#    4.c)[Removing Redundant Features](#itc )<br>
#    4.d)[Partial Dependence Plots](#ts)<br>
#    4.e) [WaterFall Models(Useful in buisness case situations)](#interpret results)<br>
#    
# 5.[Black Box Model Local Interpretation with Shap ->>Tree based models(Xgboost,Catboost,LightGbm)](#interpret results)<br>
# 
# 6.[Interpreting models with Non -Tabular Data](#interpret results)<br>
#     
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# <a id='imports'></a>
# ## 1. Import the required libraries

# In[ ]:


# Obviously
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy 
# Some sklearn tools for preprocessing and building a pipeline. 
# ColumnTransformer was introduced in 0.20 so make sure you have this version
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Our algorithms, by from the easiest to the hardest to intepret.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


# <a id='dataloading'></a>
# ## 2. Load dataset and process it
#   >The dataset is from the UCI machine learning repo containing information about the  marketing campaigns of a Portuguese bank. We will try to build classifiers that can predict whether or not the client targeted   by the campaign ended up subscribing to a term deposit (column y)

# In[ ]:


df = pd.read_csv('../input/bank-additional-full.csv',sep=";")


# In[ ]:


df.y.value_counts()


# > The dataset is imbalanced, we will need to keep that in mind when building our models!

# In[ ]:


# Get X, y
y = df["y"].map({"no":0, "yes":1})
X = df.drop("y", axis=1)


# In[ ]:


X.head()


# Let's look at the features in the X matrix:
# 
# >age (numeric)
# 
# >job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 
# >marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 
# >education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 
# >default: has credit in default? (categorical: 'no','yes','unknown')
# 
# >housing: has housing loan? (categorical: 'no','yes','unknown')
# 
# >loan: has personal loan? (categorical: 'no','yes','unknown')
# 
# >contact: contact communication type (categorical: 'cellular','telephone') 
# 
# >month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 
# >day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 
# >duration: last contact duration, in seconds (numeric).
# 
# `Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.`
# >campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# >pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# >previous: number of contacts performed before this campaign and for this client (numeric)
# 
# >poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# 
# >emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 
# >cons.price.idx: consumer price index - monthly indicator (numeric) 
# 
# >cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# 
# >euribor3m: euribor 3 month rate - daily indicator (numeric)
# 
# >nr.employed: number of employees - quarterly indicator (numeric)
# 
# Note the comment about duration feature. We will exclude it from our analysis.

# In[ ]:


X.drop("duration", inplace=True, axis=1)
X.dtypes


# In[ ]:


# Some such as default would be binary features, but since
# they have a third class "unknown" we'll process them as non binary categorical
num_features = ["age", "campaign", "pdays", "previous", "emp.var.rate", 
                "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]

cat_features = ["job", "marital", "education","default", "housing", "loan",
                "contact", "month", "day_of_week", "poutcome"]


# `We'll define a new ColumnTransformer object that keeps our numerical features and apply one hot encoding on our categorical features. That will allow us to create a clean pipeline that includes both features engineering (one hot encoding here) and training the model (a nice way to avoid data leakage)`

# In[ ]:


preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 
                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                   cat_features)])


# Now we can define our 2 models as sklearn Pipeline object, containing our preprocessing step and training of one given algorithm.[](http://)

# In[ ]:


# Logistic Regression
lr_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42))])

# Decision Tree
dt_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", DecisionTreeClassifier(class_weight="balanced"))])


# Let's split the data into training and test sets.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)


# <a id='eli'></a>
# ## 3. ELI5 to interpret white box models
# <a id='e'></a>
# >   **3.a) With Logistic Regression**

# First let's fine tune our logistic regression and evaluate its performance.

# In[ ]:


gs = GridSearchCV(lr_model, {"model__C": [1, 1.3, 1.5]}, n_jobs=-1, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)


# Let's see our best parameters and score

# In[ ]:


print(gs.best_params_)
print(gs.best_score_)


# In[ ]:


lr_model.set_params(**gs.best_params_)


# In[ ]:


lr_model.get_params("model")


# Now we can fit the model on the whole training set and calculate accuracy on the test set.

# In[ ]:


lr_model.fit(X_train, y_train)


# Generate Predictions

# In[ ]:


y_pred = lr_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# y = 1 being the minority class has lower precision and recall so the accuracy doesnt help us
# >Let's use eli5 to visualise the weights associated to each feature:

# In[ ]:


import eli5
eli5.show_weights(lr_model.named_steps["model"])


# That gives us the weights associated to each feature, that can be seen as the contribution of each feature into predicting that the class will be y=1 (the client will subscribe after the campaign).
# 
# The names for each features aren't really helping though, we can pass a list of column names to eli5 but we'll need to do a little hard work first to extract names from our preprocessor in the pipeline (since we've generated new features on the fly with the one hot encoder)

# In[ ]:


preprocessor = lr_model.named_steps["preprocessor"]
ohe_categories = preprocessor.named_transformers_["categorical"].categories_


# In[ ]:


new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]


# In[ ]:


all_features = num_features + new_ohe_features
pd.DataFrame(lr_model.named_steps["preprocessor"].transform(X_train), columns=all_features).head()


# we have a nice list of columns after processing

# In[ ]:


eli5.show_weights(lr_model.named_steps["model"], feature_names=all_features)


# `Looks like it's picking principally on whether the month is march or not, the marketing campaign seem to have been more efficient in march?`
# `So once we know the important feature the model is spitting out, we can communicate with the domain expert or the marketing head of the campaign and ask questions specifically whether there was something different they had done in the month of march and so on, commmunication is also the key`

# We can also use eli5 to explain a specific prediction, let's pick a row in the test data:

# In[ ]:


i = 4
X_test.iloc[[i]]


# In[ ]:


y_test.iloc[i]


# >Our client subsribed to the term deposit after the campaign! Let's see what our model would have predicted and how it would explain it.

# We'll need to first transform our row into the format expected by our model as eli5 cannot work directly with our pipeline.

# In[ ]:


eli5.show_prediction(lr_model.named_steps["model"], 
                     lr_model.named_steps["preprocessor"].transform(X_test)[i],
                     feature_names=all_features, show_feature_values=True)


# For this particular client ,it has predicted with 0.963 probability that he/she will subscribe, and  most importantly the model is principally
# picking conumer_price_index as one of the important predictors but according to our data dictionary this particular feature is not actually linked 
# with the client but is linked to the company so it  is trying to use as a proxy whatever the company was upto at that time so the model is not looking at
# the client characteristics to make its decision rather it is depending upon the companys' consumer_index around that time of marketing campaign, 
# so you could easily tell that the model is not making decision based on clients characteristics but rather something happening in the company
# So this model is probably not that great`

# <a id='d'></a>
# ## 3.b) With a Decision Classifier

# In[ ]:


gs = GridSearchCV(dt_model, {"model__max_depth": [3, 5, 7], 
                             "model__min_samples_split": [2, 5]}, 
                  n_jobs=-1, cv=5, scoring="accuracy")

gs.fit(X_train, y_train)


# In[ ]:


print(gs.best_params_)
print(gs.best_score_)


# In[ ]:


dt_model.set_params(**gs.best_params_)


# In[ ]:


dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# Accuracy improved a bit  with decision classifier

# In[ ]:


print(classification_report(y_test, y_pred))


# For Decision Trees, eli5 only gives feature importance, which does not say in what direction a feature impact the predicted outcome.

# In[ ]:


eli5.show_weights(dt_model.named_steps["model"], feature_names=all_features)


# Here the most important feature seems to be nr.employed.(which again is not linked with clients characteristics)-> Not a very great model . We can also get an explanation for a given prediction, this will calculate the contribution of each feature in the prediction:

# In[ ]:


#This is for the same client that we had calculated for logistic regression
#Local interpretation of a particular row
eli5.show_prediction(dt_model.named_steps["model"], 
                     dt_model.named_steps["preprocessor"].transform(X_test)[i],
                     feature_names=all_features, show_feature_values=True)


# Here the explanation for a single prediction is calculated by following the decision path in the tree, and adding up contribution of each feature from each node crossed into the overall probability predicted.
# Again the model is not so great as it is taking the decision on the basis of companys' characteristics from that time .i.e here it is number of employees , which is pretty vague in predicted whether the customer subscribed or not

# bbb<a id='rand'></a>
# ## 4. Black Box Model Interpretation ->Random Forest

# In[ ]:


# Random Forest
rf_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])


# In[ ]:


gs = GridSearchCV(rf_model, {"model__max_depth": [10, 15], 
                             "model__min_samples_split": [5, 10]}, 
                  n_jobs=-1, cv=5, scoring="accuracy")

gs.fit(X_train, y_train)


# In[ ]:


print(gs.best_params_)
print(gs.best_score_)


# In[ ]:


rf_model.set_params(**gs.best_params_)


# In[ ]:


rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# Accuracy has improved a bit using Random Forests

# In[ ]:


print(classification_report(y_test, y_pred))


# <a id='int'></a>
# ## 4.1 Confidence Based on Tree Variance 

# In order to explain why the model classifies invidividual observations as class 0 or 1, we are going to use the `LimeTabularExplainer` from the library lime, this is the main explainer to use for tabular data. Lime also provides an explainer for text data, for images and for time-series.
# When using the tabular explainer, we need to provide our training set as parameter so that lime can compute statistics on each feature, either mean and std for numerical features, or frequency of values for `categorical features`. Those statistics are used to scale the data and generate new perturbated data to train our local linear models on.

# In[ ]:


from lime.lime_tabular import LimeTabularExplainer


# The parameters passed to the explainer are:
# >1.`our training set`, we need to make sure we use the training set without one hot encoding
# 
# >2`mode`: the explainer can be used for classification or regression
# 
# >3.`feature_names`: list of labels for our features
# 
# >4.`categorical_features`: list of indexes of categorical features
# 
# >5.`categorical_names`: dict mapping each index of categorical feature to a list of corresponding labels
# 
# >6.`dicretize_continuous`: will discretize numerical values into buckets that can be used for explanation. For instance it can tell us that the decision was made because distance is in bucket [5km, 10km] instead of telling us distance is an importante feature.
# 
# First, in order to get the categorical_names parameter we need to build a dictionary with indexes of categorical values in original dataset as keys and lists of possible categories as values:

# In[ ]:


categorical_names = {}
for col in cat_features:
    categorical_names[X_train.columns.get_loc(col)] = [new_col.split("__")[1] 
                                                       for new_col in new_ohe_features 
                                                       if new_col.split("__")[0] == col]


# In[ ]:


categorical_names


# In[ ]:


import pandas as pd
def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):

    """Converts data with categorical values as string into the right format 
     for LIME, with categorical values as integers labels.
    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency. 
    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
    """
    # If the data isn't a dataframe, we need to be able to build it

    if not isinstance(X, pd.DataFrame):

        X_lime = pd.DataFrame(X, columns=col_names)

    else:

        X_lime = X.copy()

    for k, v in categorical_names.items():

        if not invert:

            label_map = {

                str_label: int_label for int_label, str_label in enumerate(v)
            }

        else:

            label_map = {

                int_label: str_label for int_label, str_label in enumerate(v)

            }

        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

    return X_lime


# In[ ]:


convert_to_lime_format(X_train, categorical_names).head()


# In[ ]:


explainer = LimeTabularExplainer(convert_to_lime_format(X_train, categorical_names).values,
                                 mode="classification",
                                 feature_names=X_train.columns.tolist(),
                                 categorical_names=categorical_names,
                                 categorical_features=categorical_names.keys(),
                                 discretize_continuous=True,
                                 random_state=42)


# Great, our explainer is ready. Now let's pick an observation we want to explain.

# `Explain new observations`
# 
# We'll create a variable called observation that contains our ith observation in the test dataset.

# In[ ]:


i = 2
X_observation = X_test.iloc[[i], :]
X_observation


# Compare every model predictions for a specific client 

# In[ ]:


print(f"""* True label: {y_test.iloc[i]}
* LR: {lr_model.predict_proba(X_observation)[0]}
* DT: {dt_model.predict_proba(X_observation)[0]}
* RF: {rf_model.predict_proba(X_observation)[0]}
""")


# So for this particular client,its true lable is 0 i.e he/she didnt subscribe, Random Forest seems to be the predicting with more confidence than any other model 

# Let's convert our observation to lime format and convert it to a numpy array.

# In[ ]:


observation = convert_to_lime_format(X_test.iloc[[i], :],categorical_names).values[0]
observation


# In order to explain a prediction, we use the explain_instance method on our explainer. This will generate new data with perturbated features around the observation and learn a local linear model. It needs to take:
# *  >our observation as a numpy array
# *  >a function that uses our model to predict probabilities given the data (in same format we've passed in our explainer). That means we cannot pass directly our rf_model.predict_proba because our pipeline expects string labels for categorical values. We will need to create a custom function rf_predict_proba that first converts back integer labels to strings and then calls rf_model.predict_proba.
# * >num_features: number of features to consider in explanation

# In[ ]:


# Let write a custom predict_proba functions for our models:
from functools import partial

def custom_predict_proba(X, model):
    X_str = convert_to_lime_format(X, categorical_names, col_names=X_train.columns, invert=True)
    return model.predict_proba(X_str)

lr_predict_proba = partial(custom_predict_proba, model=lr_model)
dt_predict_proba = partial(custom_predict_proba, model=dt_model)
rf_predict_proba = partial(custom_predict_proba, model=rf_model)


# Let's test our custom function to make sure it generates propabilities properly

# In[ ]:


explanation = explainer.explain_instance(observation, lr_predict_proba, num_features=5)


# Now that we have generated our explanation, we have access to several representations. The most useful one when working in a notebook is show_in_notebook.
# On the left it shows the list of probabilities for each class, here the model classified our observation as 0 (non subsribed) with a high probability.
# >If you set show_table=True, you will see the table with the most important features for this observation on the right.

# In[ ]:


explanation.show_in_notebook(show_table=True, show_all=False)


# LIME is fitting a linear model on a local perturbated dataset. You can access the coefficients, the intercept and the R squared of the linear model by calling respectively .local_exp, .intercept and .score on your explanation.

# In[ ]:


print(explanation.local_exp)
print(explanation.intercept)
print(explanation.score)


# If your R-squared is low, the linear model that LIME fitted isn't a great approximation to your model, which means you should not rely too much on the explanation it provides.

# In[ ]:


explanation = explainer.explain_instance(observation, dt_predict_proba, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)
print(explanation.score)


# In[ ]:


explanation = explainer.explain_instance(observation, rf_predict_proba, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)
print(explanation.score)


# Here we can see how each of features influence the model to choose and decide the probability score

# <a id='it'></a>
# ## 4.2 Permutation Importance

# The idea of calculating feature_importances is simple, but great.
# Splitting down the idea into easy steps:
# 1. train random forest model (assuming with right hyper-parameters)
# 2. find prediction score of model (call it benchmark score)
# 3. find prediction scores p more times where p is number of features, each time randomly shuffling the column of i(th) feature
# 4. compare all p scores with benchmark score. If randomly shuffling some i(th) column is hurting the score, that means that our model is bad without that feature.
# 5. remove the features that do not hurt the benchmark score and retrain the model with reduced subset of features.

# Permutation importance for Random forest from scratch 

# In[ ]:


# defining roc under the curve as scoring criteria (any other criteria can be used in a similar manner)
def score(x1,x2):
    return metrics.auc(x1,x2)
# defining feature importance function based on above logic
def feat_imp(m, x, y, small_good = True): 

     """
       m: random forest model
       x: matrix of independent variables
       y: output variable
       small__good: True if smaller prediction score is better
     """  

     score_list = {} 

     score_list['original'] = score(m.predict(x.values), y) 

     imp = {} 

     for i in range(len(x.columns)): 

            rand_idx = np.random.permutation(len(x)) # randomization

            new_coli = x.values[rand_idx, i] 

            new_x = x.copy()            

            new_x[x.columns[i]] = new_coli 

            score_list[x.columns[i]] = score(m.predict(new_x.values), y) 

            imp[x.columns[i]] = score_list['original'] - score_list[x.columns[i]] # comparison with benchmark

     if small_good: 
          return sorted(imp.items(), key=lambda x: x[1]) 

     else: return sorted(imp.items(), key=lambda x: x[1], reverse=True)


# Suprisingly ELI5 also has an api for random forest model feature importance

# In[ ]:


eli5.show_weights(rf_model.named_steps["model"], 
                  feature_names=all_features)


# We can explain roughly what our model seems to focus on mostly. We also get the standard deviation of feature importance accross the multiple trees in our ensemble.

# <a id='itw'></a>
# ## 4.3 Removing Redundant Features

# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(X).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X.columns, orientation='left', leaf_font_size=16)
plt.show()


# We can remove  some of these related features to see if the model can be simplified without impacting the accuracy.This is called aggloremative clustering,which can be used to remove redudant features from you model and train  again with less features to see whether it will impact your accuracy

# <a id='ts'></a>
# ## 4.4 Partial Dependence Plots

#  Partial dependence plots that can be viewed as graphical representation of linear model coefficients, but can be extended to seemingly black box models also. The idea is to isolate the changes made in predictions to solely come from a specific feature. It is different than scatter plot of X vs. Y as scatter plot does not isolate the direct relationship of X vs. Y and can be affected by indirect relationships with other variables on which both X and Y depend.

# `Steps for PDP Plots

# The steps to make PDP plot are as follows:
# 1. Train a random forest model (let’s say F1…F4 are our features and Y is target variable. Suppose F1 is the most important feature). 
# 2. we are interested to explore the direct relationship of Y and F1
# 3. replace column F1 with F1(A) and find new predictions for all observations. take mean of predictions. (call it base value)
# 4. repeat step 3 for F1(B) … F1(E), i.e. for all distinct values of feature F1. 
# 5. PDP’s X-axis has distinct values of F1 and Y-axis is change in mean prediction for that F1 value from base value.

# ## MORE TO COME STAY TUNED 

# 

# In[ ]:





# In[ ]:




