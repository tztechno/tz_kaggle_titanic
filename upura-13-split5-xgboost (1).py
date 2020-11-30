#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


get_ipython().system('ls ../input/titanic')


# In[3]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[4]:


gender_submission.head()


# In[5]:


data = pd.concat([train, test], sort=False)


# In[6]:


print(len(train), len(test), len(data))


# In[7]:


data.isnull().sum()


# In[8]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# In[9]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[10]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['fare_value']=data['Fare']/50


# In[11]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['age_value']=data['Age']/50


# In[12]:


data['family'] = (data['SibSp'] + data['Parch'])/5 


# In[13]:


data['isAlone'] = 0
data.loc[data['family'] > 0, 'isAlone'] = 1


# In[14]:


delete_columns = ['Name','PassengerId','SibSp','Parch','Ticket','Cabin','Age','Fare']
data.drop(delete_columns, axis=1, inplace=True)


# In[15]:


train = data[:len(train)]
test = data[len(train):]


# In[16]:


y_train0 = train['Survived']
X_train0 = train.drop('Survived', axis = 1)
X_test0 = test.drop('Survived', axis = 1)


# In[17]:


from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=5,    # 分割を1個生成
                  train_size=0.9,  # 学習
                  test_size =0.1,  # テスト
                  random_state=1)  # 乱数種（再現用）


# In[18]:


X = np.array(X_train0)
y = np.array(y_train0)


# In[19]:


X.shape


# In[20]:


y.shape


# In[21]:


from xgboost import XGBClassifier
clf = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)


# In[22]:


from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
    
ss = ShuffleSplit(n_splits=5,    # 分割を1個生成
                  train_size=0.8,  # 学習
                  test_size =0.2,  # テスト
                  random_state=0)  # 乱数種（再現用）

for train_index, test_index in ss.split(X): 

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    clf.fit(X_train, Y_train) 
    print(clf.score(X_test, Y_test))


# In[23]:


type(X_train)


# In[24]:


type(X_test0)


# In[25]:


#np.array(X_test0)


# In[26]:


y_pred = clf.predict(np.array(X_test0))


# In[27]:


y_pred[:20]


# In[28]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[29]:


sub

