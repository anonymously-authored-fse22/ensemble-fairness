#!/usr/bin/env python
# coding: utf-8

# # **商务智能课程设计**
# 
# ###   题目：对美国成年人收入的数据分析
# ###   关键词：机器学习、网格调参、数据分析
# 
# 
# ####   学生：  江\*\*
# 
# ####   学号：  2018\*\*\*\*\*\*048
#          
# ####   教师：  吴\*\*

# # 1.基本配置和数据导入

# ## 1.1. 导入相关包

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

print("Setup Complete")


# ## 1.2. 从csv导入数据，并查看基本信息

# In[ ]:


path_data = '../input/adult-census-income/adult.csv'
data_origin = pd.read_csv(path_data)
print(data_origin.info())
print('\n\n\n==========\n\n\n'+data_origin.head().to_string())


# 数据说明:从上到下依次为:年龄、类型、最终分析权重、教育程度、受教育时间、婚姻状况、职业、关系、种族、性别、资本收益、资本损失、每周工作小时数、原籍、收入

# # 2. 使用XGB分类器进行预测

# ## 2.1 数据简单处理

# 观察数据

# In[ ]:


data_origin.head(10)


# 虽然info函数说没有空值，但是可以明显看到有很多项有无效值?存在
# 
# 暂时先简单替换这些行，让X填充或使用最多的值进行填充

# In[ ]:


data_use = data_origin.replace('?', np.nan)
data_use["workclass"] = data_use["workclass"].fillna("X")
data_use["occupation"] = data_use["occupation"].fillna("X")
data_use["native.country"] = data_use["native.country"].fillna("United-States")


# 对income的值进行更改

# In[ ]:


data_use['income'].unique()


# 只有这两种情况，那就只处理这两种情况，将这个数据类型改成方便我们处理的

# In[ ]:


data_use['income']=data_use['income'].map({'<=50K': 0, '>50K': 1})
#data_use.head(10)


# 性别这里显然只有两种，直接处理

# In[ ]:


data_use["sex"] = data_use["sex"].map({"Male": 0, "Female":1})


# 我们还有很多字符串类型数据需要处理，我们将他们转换为数值类型
# 
# 这些属性全部可以使用One hot方式编码

# In[ ]:


for feature in data_use.columns:
    if data_use[feature].dtype == 'object':
        #data_temp = pd.Categorical(data_use[feature]).codes
        data_temp = np.array(data_use[feature])
        #print(data_temp)
        enc_lab = LabelEncoder()
        data_lab = enc_lab.fit_transform(data_temp)
        enc_oht = OneHotEncoder(sparse=False)
        data_lab = data_lab.reshape(len(data_lab), 1)
        data_oht = enc_oht.fit_transform(data_lab)
        #print(data_oht[:,0].A)
        #print(data_oht.shape)
        data_use.drop(feature, axis=1, inplace=True)
        for i in range(data_oht.shape[1]):
        #    print(feature+str(i))
        #    print(data_oht)
            data_use.insert(data_use.shape[1]-1, feature+str(i), data_oht[:,i])
print(data_use.shape)
print(data_use.columns)


# ## 2.2 划分数据集

# In[ ]:


y = data_use['income']
features = data_use.columns[:-1]
X = data_use[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print('Done.')


# ## 2.3 进行预测

# In[ ]:


model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


# ## 2.4 查看准确率

# In[ ]:


print("Accuracy: %s%%" % (100*accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))


# 有87.8%的准确率。

# # 3. 使用随机森林进行预测

# 这次要加上一个调参的步骤，使用的是GridSearchCV进行调参

# ## 3.1 数据处理

# 这些步骤和之前类似，但是因为硬件的限制这里就让参数稍微少一些

# In[ ]:


#由于要调参，参数太多跑起来很慢，所以重新处理数据，让数据量更少
data_use = data_origin.replace('?', np.nan).dropna()
data_use['income']=data_use['income'].map({'<=50K': 0, '>50K': 1})
data_use["sex"] = data_use["sex"].map({"Male": 0, "Female":1})
#看一下减少的行数会不会太多了
print(data_use.shape)
print(data_origin.shape)


# 减少的行数其实并没有太多，可以接受

# 特征工程，分析整合属性
# 
# 另外，由于转为One hot后参数数量过多，所以对于婚姻状况属性还是用类型编码，其他的直接删掉

# In[ ]:


data_use["marital.status"] = data_use["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
data_use["marital.status"] = data_use["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
data_use["marital.status"] = data_use["marital.status"].map({"Married":1, "Single":0})
data_use["marital.status"] = data_use["marital.status"].astype(int)

#这里只使用部分属性，删掉其他的
data_use.drop(labels=["workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
print(data_use.head())


# ## 3.2 划分数据集

# In[ ]:


y = data_use['income']
features = data_use.columns[:-1]
X = data_use[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print('Done.')


# ## 3.3 调参

# 这次使用GridSearchCV调参,这一块花的时间很久，所以注释掉了
# 
# 运行结果中最好的参数是
# 
# * n_estimator = 250
# * max_feature = 5

# In[ ]:


# model_rf = RandomForestClassifier()
# n_estimators = np.array([50, 100, 150, 200, 250])
# max_features = np.array([1, 2, 3, 4, 5])
# param_grid = dict(n_estimators = n_estimators, max_features = max_features )
# #将训练集和测试集划分为10个互斥子集，交叉验证
# kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=1)
# search_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# result_grid = search_grid.fit(X_train, y_train)
# print(grid_result.best_params_)


# 使用随机森林分类器，参数为尝试出的最佳参数

# In[ ]:


model_rf = RandomForestClassifier(n_estimators=250, max_features=5)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


# 查看准确率

# In[ ]:


print("Accuracy: %s%%" % (100*accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))


# 有83%左右的准确率

# # 4. 数据分析

# 针对原始数据，除了可以建立模型进行预测之外，还可以用简单统计或画图等方式进行简单的分析

# ## 4.1 查看工作时间和薪酬的关系

# 重新加载原始数据
# 
# 改变income的数据类型便于统计

# In[ ]:


data_use = data_origin.copy()
data_use['income']=data_use['income'].map({'<=50K': 0, '>50K': 1})


# In[ ]:


rel = pd.crosstab(data_use['hours.per.week'], data_use['income'], rownames= ['hours.per.week'])
rel


# 这样看似乎不够直观，试试画张折线图

# In[ ]:


sns.lineplot(x=data_use['hours.per.week'], y=rel.loc[:,1] )


# 可以看到薪资高的人多数都每周工作40到50小时，也有很大一部分薪资高的人每周工作70小时

# ## 4.2 看看薪资和性别的关系 

# In[ ]:


rel_s = pd.crosstab(data_use['sex'], data_use['income'], rownames= ['sex'])
plt.title('female')
plt.pie(rel_s.iloc[0,:],labels=['low paid', 'high paid'])
plt.show()
plt.title('male')
plt.pie(rel_s.iloc[1,:],labels=['low paid', 'high paid'])
plt.show()


# 可以看出，这些女性的高薪比例要远低于男性

# ## 4.3 受教育时间和薪资的关系

# In[ ]:


sns.lineplot(x="education.num",y="income",data=data_use)


# 显而易见，受教育时间越长，工资高的可能性就越高

# ## 4.4 家庭关系与薪资的关系

# In[ ]:


sns.factorplot(x="relationship",y="income",data=data_use,kind="bar", size = 10)


# 似乎结了婚的人有更大的可能性是高薪
