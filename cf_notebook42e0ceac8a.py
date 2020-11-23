########################################################


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_train = pd.read_csv('../input/titanic/train.csv')
data_train = np.array(df_train)
df_test = pd.read_csv('../input/titanic/test.csv')
data_test = np.array(df_test)


# In[3]:


data_train


# In[4]:


len(data_train)


# In[5]:


len(data_test)


# In[6]:


tg = []
for i in range(0, len(data_train)):
    tg.append(data_train[i][1])
target = np.array(tg)
target


# In[7]:


target2=pd.DataFrame(target)
target2.to_csv('target4.csv', index=False)


# In[8]:


data_train.shape


# In[9]:


data_test.shape


# In[10]:


td=[]
for i in range(0, len(data_train)):
    td.append(data_train[i][2:])
for i in range(0, len(data_test)):
    td.append(data_test[i][1:])
traintestdata = np.array(td)
traintestdata


# In[11]:


td=[]
for i in range(0, len(traintestdata)):
    td.append(traintestdata[i][0])
max(td)


# In[12]:


#Pcalss
td=[]
for i in range(0, len(traintestdata)):
    td.append((traintestdata[i][0]-1)/2)
traintestdata_Pclass = np.array(td)
traintestdata_Pclass


# In[13]:


#sex
td=[]
for i in range(0, len(traintestdata)):
    if traintestdata[i][2] == 'male':
        td.append(0)
    else:
        td.append(1)   
traintestdata_sex = np.array(td)
traintestdata_sex
#male0,female1


# In[14]:


td=[]
for i in range(0, len(traintestdata)):
    if str(traintestdata[i][3]) == 'nan':
        td.append(29)
    else:
        td.append(traintestdata[i][3])
        
np.mean(td)


# In[15]:


#age
td=[]
for i in range(0, len(traintestdata)):
    if str(traintestdata[i][3]) == 'nan':
        td.append(29/80)
    else:
        td.append(traintestdata[i][3]/80)

traintestdata_age = np.array(td)
traintestdata_age


# In[16]:


#family
td=[]
for i in range(0, len(traintestdata)):
        td.append((1+traintestdata[i][4]+traintestdata[i][5])/11)  
traintestdata_family = np.array(td)
traintestdata_family


# In[17]:


#fare
td=[]
for i in range(0, len(traintestdata)): 
    if str(traintestdata[i][7]) == 'nan':
        td.append(0.065)           
    else:
        td.append(traintestdata[i][7]/512.3292)     
        
traintestdata_fare = np.array(td)
traintestdata_fare


# In[18]:


#embarked
td=[]
for i in range(0, len(traintestdata)):
    if traintestdata[i][9]=='S':
        td.append(0/3)          
    elif traintestdata[i][9]=='Q':
        td.append(1/3)          
    elif traintestdata[i][9]=='C':
        td.append(2/3)          
    else:
        td.append(3/3)  
        
traintestdata_embarked = np.array(td)
traintestdata_embarked


# In[19]:


traintestdata2=[]

traintestdata２.append(traintestdata_Pclass)
traintestdata２.append(traintestdata_sex)
traintestdata２.append(traintestdata_age)
traintestdata２.append(traintestdata_family)
traintestdata２.append(traintestdata_fare)
traintestdata２.append(traintestdata_embarked)

traintestdata3=np.transpose(traintestdata2)
traintestdata3


# In[20]:


traintestdata4=pd.DataFrame(traintestdata3)
traintestdata4.to_csv('traintestdata4.csv', index=False)

########################################################



# In[21]:


from __future__ import print_function
import numpy as np
from keras.datasets import mnist

from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Dropout, Flatten

from keras.optimizers import SGD
from keras.utils import np_utils
from keras import metrics


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
print(__doc__)
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('traintestdata4.csv')
data=np.array(df)

dt=pd.read_csv('target4.csv')
target=np.array(dt)


# In[23]:


X = np.array(data[0:891])
y = np.array(target)


# In[24]:


from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1,
                  train_size=0.9,
                  test_size =0.1,
                  random_state=0)

train_index, test_index = next(ss.split(X)) 
list(train_index), list(test_index) 

X_train, X_test = X[train_index], X[test_index]
Y_train, Y_test = y[train_index], y[test_index]

########################################################



# In[25]:


NB_EPOCH = 600
VERBOSE = 1
OPTIMIZER = SGD() 
BATCH_SIZE = 32
NB_CLASSES = 2

N_HIDDEN0 = 512
N_HIDDEN1 = 256
N_HIDDEN2 = 128
N_HIDDEN3 = 128
N_HIDDEN4 = 64
N_HIDDEN5 = 64

DROPOUT = 0.5
VALIDATION_SPLIT = 0.5
RESHAPED = 6


# In[26]:


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

model.add (Dense(N_HIDDEN3))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add (Dense(N_HIDDEN4))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add (Dense(N_HIDDEN5))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))


model.add (Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()


# In[27]:


model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = OPTIMIZER,
             metrics = ['accuracy'])


# In[28]:


history = model.fit(X_train, Y_train, 
                    validation_data = (X_test, Y_test), 
                    epochs=NB_EPOCH, 
                    batch_size=BATCH_SIZE)


# In[29]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()


# In[30]:


score = model.evaluate (X_test, Y_test, verbose = VERBOSE )


# In[31]:


print("Test score:", score[0])
print('Test accuracy:', score[1])

########################################################



# In[32]:


x_test_honban = np.array(data[891:])
x_test_honban.shape


# In[33]:


y_pred = model.predict(x_test_honban)
y_pred


# In[34]:


y_pred_life=[]
for i in range(len(y_pred)):
    if y_pred[i][0]>y_pred[i][1]:
        y_pred_life.append(0)
    else:
        y_pred_life.append(1)        


# In[35]:


my_submit0 = ['PassengerId']
my_submit1 = ['Survived']

for i in range(0, len(y_pred)):
    my_submit0.append(str(892+i))
    my_submit1.append(str(y_pred_life[i]))

    
my_submit=[]

my_submit.append(my_submit0)
my_submit.append(my_submit1)

my_submit2=np.transpose(my_submit)
my_submit2


# In[36]:


my_submit3 = pd.DataFrame(my_submit2)
my_submit3.to_csv('titanic_submission.csv', index=False, header=False)


########################################################




