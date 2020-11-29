#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


get_ipython().system('ls ../input/titanic')


# In[3]:


train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
gender_submission = pd.read_csv("./input/gender_submission.csv")


# In[4]:


gender_submission.head()


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


data = pd.concat([train, test], sort=False)


# In[8]:


data.head()


# In[9]:


print(len(train), len(test), len(data))


# In[10]:


data.isnull().sum()


# In[11]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# In[12]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[13]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['fare_value']=data['Fare']/50


# In[14]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['age_value']=data['Age']/50


# In[15]:


data['family'] = (data['SibSp'] + data['Parch'])/5 


# In[16]:


data['isAlone'] = 0
data.loc[data['family'] > 0, 'isAlone'] = 1


# In[17]:


delete_columns = ['Name','PassengerId','SibSp','Parch','Ticket','Cabin','Age','Fare']
data.drop(delete_columns, axis=1, inplace=True)


# In[18]:


train = data[:len(train)]
test = data[len(train):]


# In[19]:


y_train0 = train['Survived']
X_train0 = train.drop('Survived', axis = 1)
X_test0 = test.drop('Survived', axis = 1)


# In[20]:


X = np.array(X_train0)
y = np.array(y_train0)


# In[21]:


X.shape


# In[22]:


y.shape


# In[23]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()


# In[24]:


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


# In[25]:


y_pred = clf.predict(X_test0)


# In[26]:


y_pred[:20]


# In[27]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[28]:


sub

