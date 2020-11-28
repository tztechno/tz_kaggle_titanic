#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd


# In[62]:


get_ipython().system('ls ../input/titanic')


# In[63]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[64]:


gender_submission.head()


# In[65]:


train.head()


# In[66]:


test.head()


# In[67]:


data = pd.concat([train, test], sort=False)


# In[68]:


data.head()


# In[69]:


print(len(train), len(test), len(data))


# In[70]:


data.isnull().sum()


# In[71]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# In[72]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[73]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['fare_value']=data['Fare']/50


# In[74]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['age_value']=data['Age']/50


# In[75]:


data['family'] = (data['SibSp'] + data['Parch'])/5 


# In[76]:


data['isAlone'] = 0
data.loc[data['family'] > 0, 'isAlone'] = 1


# In[77]:


delete_columns = ['Name','PassengerId','SibSp','Parch','Ticket','Cabin','Age','Fare']
data.drop(delete_columns, axis=1, inplace=True)


# In[78]:


train = data[:len(train)]
test = data[len(train):]


# In[79]:


y_train0 = train['Survived']
X_train0 = train.drop('Survived', axis = 1)
X_test0 = test.drop('Survived', axis = 1)


# In[80]:


from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=5,    # 分割を1個生成
                  train_size=0.9,  # 学習
                  test_size =0.1,  # テスト
                  random_state=1)  # 乱数種（再現用）


# In[81]:


X = np.array(X_train0)
y = np.array(y_train0)


# In[82]:


X.shape


# In[83]:


y.shape


# In[84]:


from __future__ import print_function
import numpy as np
from keras.datasets import mnist

from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Dropout, Flatten

from keras.optimizers import SGD
from keras.utils import np_utils
from keras import metrics


# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')
print(__doc__)


# In[86]:


NB_EPOCH = 100   #CHAMPION400
VERBOSE = 1
OPTIMIZER = SGD() 
BATCH_SIZE = 32
NB_CLASSES = 2

N_HIDDEN0 = 256
N_HIDDEN1 = 128
N_HIDDEN2 = 64

DROPOUT = 0.5  #CHAMPION0
VALIDATION_SPLIT = 0.5  #CHAMPION0.5
RESHAPED = 7


# In[87]:


model = Sequential()

model.add (Dense(N_HIDDEN0, input_shape = (RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))


model.add (Dense(N_HIDDEN1))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add (Dense(N_HIDDEN2))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))


model.add (Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()


# In[88]:


from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
    
ss = ShuffleSplit(n_splits=5,    # 分割を1個生成
                  train_size=0.8,  # 学習
                  test_size =0.2,  # テスト
                  random_state=0)  # 乱数種（再現用）

for train_index, test_index in ss.split(X): 

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                 optimizer = OPTIMIZER,
                 metrics = ['accuracy'])    
    
    history = model.fit(X_train, Y_train, 
                        validation_data = (X_test, Y_test), 
                        epochs=NB_EPOCH, 
                        batch_size=BATCH_SIZE) 
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])

    plt.show() 
    
    score = model.evaluate (X_test, Y_test, verbose = VERBOSE )   
    
    print("Test score:", score[0])
    print('Test accuracy:', score[1])  


# In[89]:


y_pred = model.predict(X_test0)


# In[90]:


y_pred[:20]


# In[91]:


y_pred_life=[]
for i in range(len(y_pred)):
    if y_pred[i][0]>y_pred[i][1]:
        y_pred_life.append(0)
    else:
        y_pred_life.append(1)  
y_pred_life


# In[92]:


sub = gender_submission
sub['Survived'] = list(map(int, y_pred_life))
sub.to_csv("submission.csv", index=False)


# In[93]:


sub

