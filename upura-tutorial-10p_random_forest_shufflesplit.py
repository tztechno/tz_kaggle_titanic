#!/usr/bin/env python
# coding: utf-8

# In[507]:


import numpy as np
import pandas as pd


# In[508]:


get_ipython().system('ls ../input/titanic')


# In[509]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[510]:


gender_submission.head()


# In[511]:


train.head()


# In[512]:


test.head()


# In[513]:


data = pd.concat([train, test], sort=False)


# In[514]:


data.head()


# In[515]:


print(len(train), len(test), len(data))


# In[516]:


data.isnull().sum()


# In[517]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# In[518]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[519]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['fare_value']=data['Fare']/50


# In[520]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['age_value']=data['Age']/50


# In[521]:


data['family'] = (data['SibSp'] + data['Parch'])/5 


# In[522]:


data['isAlone'] = 0
data.loc[data['family'] > 0, 'isAlone'] = 1


# In[523]:


delete_columns = ['Name','PassengerId','SibSp','Parch','Ticket','Cabin','Age','Fare']
data.drop(delete_columns, axis=1, inplace=True)


# In[524]:


train = data[:len(train)]
test = data[len(train):]


# In[525]:


y_train0 = train['Survived']
X_train0 = train.drop('Survived', axis = 1)
X_test0 = test.drop('Survived', axis = 1)


# In[526]:


from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=5, 
                  train_size=0.9,  
                  test_size =0.1,
                  random_state=1)  


# In[527]:


X = np.array(X_train0)
y = np.array(y_train0)


# In[528]:


X.shape


# In[529]:


y.shape


# In[530]:


###xxx
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

for train_index, test_index in ss.split(X, y):
    x_train, x_test = X[train_index], X[test_index] # 学習データ，テストデータ
    y_train, y_test = y[train_index], y[test_index] # 学習データのラベル，テストデータのラベル 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    pred_accuracy = round(clf.score(x_test, y_test)*100, 2)
    print(pred_accuracy)


# In[531]:


y_pred = clf.predict(X_test0)


# In[532]:


y_pred[:20]


# In[533]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


# In[534]:


sub

