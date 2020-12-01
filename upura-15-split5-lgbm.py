#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd


# In[52]:


get_ipython().system('ls ../input/titanic')


# In[53]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[54]:


gender_submission.head()


# In[55]:


data = pd.concat([train, test], sort=False)


# In[56]:


print(len(train), len(test), len(data))


# In[57]:


data.isnull().sum()


# In[58]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# In[59]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[60]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['fare_value']=data['Fare']/50


# In[61]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['age_value']=data['Age']/50


# In[62]:


data['family'] = (data['SibSp'] + data['Parch'])/5 


# In[63]:


data['isAlone'] = 0
data.loc[data['family'] > 0, 'isAlone'] = 1


# In[64]:


delete_columns = ['Name','PassengerId','SibSp','Parch','Ticket','Cabin','Age','Fare']
data.drop(delete_columns, axis=1, inplace=True)


# In[65]:


train = data[:len(train)]
test = data[len(train):]


# In[66]:


y_train0 = train['Survived']
X_train0 = train.drop('Survived', axis = 1)
X_test0 = test.drop('Survived', axis = 1)


# In[67]:


X = np.array(X_train0)
y = np.array(y_train0)


# In[68]:


X.shape


# In[69]:


y.shape


# In[70]:


import lightgbm as lgbm
random_state = 42

clf = lgbm.LGBMClassifier(random_state=random_state, silent=True, metric='None', n_jobs=4)


# In[71]:


from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
    
ss = ShuffleSplit(n_splits=5,
                  train_size=0.8,
                  test_size =0.2,
                  random_state=0)

for train_index, test_index in ss.split(X): 

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    clf.fit(X_train, Y_train) 
    print(clf.score(X_test, Y_test))


# In[72]:


type(X_train)


# In[73]:


type(X_test0)


# In[74]:


y_pred = clf.predict(np.array(X_test0))


# In[75]:


y_pred[:20]


# In[76]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[77]:


sub

