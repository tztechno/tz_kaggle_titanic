#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


get_ipython().system('ls ../input/titanic')


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


gender_submission.head()


# In[ ]:


data = pd.concat([train, test], sort=False)


# In[ ]:


print(len(train), len(test), len(data))


# In[ ]:


data.isnull().sum()


# In[ ]:


data['Sex'].replace(['male','female'], [0,1], inplace=True)


# In[ ]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['fare_value']=data['Fare']/50


# In[ ]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['age_value']=data['Age']/50


# In[ ]:


data['family'] = (data['SibSp']+data['Parch'])/5 


# In[ ]:


data['isAlone'] = 0
data.loc[data['family'] > 0, 'isAlone'] = 1


# In[ ]:


delete_columns = ['Name','PassengerId','SibSp','Parch','Ticket','Cabin','Age','Fare']
data.drop(delete_columns, axis=1, inplace=True)


# In[ ]:


train = data[:len(train)]
test = data[len(train):]


# In[ ]:


y_train0 = train['Survived']
X_train0 = train.drop('Survived', axis = 1)
X_test0 = test.drop('Survived', axis = 1)


# In[ ]:


X = np.array(X_train0)
y = np.array(y_train0)


# In[ ]:


from catboost import Pool, CatBoostClassifier, cv
clf = CatBoostClassifier()


# In[ ]:


from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
    
ss = ShuffleSplit(n_splits = 1,
                  train_size = 0.5,
                  test_size = 0.5,
                  random_state = 0)

for train_index, test_index in ss.split(X):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    clf.fit(X_train, Y_train) 
    


# In[ ]:


y_pred = clf.predict(np.array(X_test0))
y_pred = y_pred.astype(np.int)


# In[ ]:


y_pred[:20]


# In[ ]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[ ]:


sub

