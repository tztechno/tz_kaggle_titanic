#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import random
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from tqdm.notebook import tqdm
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import math


# In[ ]:





# In[3]:


@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    msg = f"[{name}] done in {time.time() - t0:.0f} s"
    print(msg)
    
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# In[4]:


from pandas_profiling import ProfileReport
from matplotlib_venn import venn2
sns.set_style('darkgrid')


# In[5]:


INPUT_DIR = '../input/titanic'
OUTPUT_DIR = './'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[6]:


train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")


# In[7]:


whole_df=pd.concat([train_df, test_df], ignore_index=True)


# In[8]:


#train_df.dtypes


# In[9]:


#report = ProfileReport(train_df)
#report.to_file(os.path.join(OUTPUT_DIR, 'train_report.html'))


# In[10]:


#report = ProfileReport(test_df)
#report.to_file(os.path.join(OUTPUT_DIR, 'test_report.html'))


# In[11]:


#report = ProfileReport(whole_df)
#report.to_file(os.path.join(OUTPUT_DIR, 'whole_report.html'))


# In[12]:


#!pip install sweetviz
#import sweetviz as sv


# In[13]:


#compare_report = sv.compare([train_df, 'Training Data'],
#                            [test_df, 'Test Data'])
#compare_report.show_html('./compare.html')


# In[ ]:





# In[14]:



def create_sex_value(input_df):
    use_columns = ['Sex']  
    
    td=[]
    for i in range(0, len(input_df)):
        if input_df['Sex'][i] == 'male':
            td.append(0)
        else:
            td.append(1)   
    sex_value = pd.DataFrame(td, columns=['sex_value'])
              
    return sex_value.copy()


# In[15]:



def create_family_size(input_df):
    use_columns = ['SibSp','Parch']  
    
    td=[]
    for i in range(0, len(input_df)):

        td.append((input_df['SibSp'][i]+input_df['Parch'][i]+1)/10)          
        
    family_size = pd.DataFrame(td, columns=['family_size'])
              
    return family_size.copy()


# In[16]:


def create_sibsp_size(input_df):
    use_columns = ['SibSp']  
    
    td=[]
    for i in range(0, len(input_df)):

        td.append(input_df['SibSp'][i]/4)          
        
    sibsp_size = pd.DataFrame(td, columns=['sibsp_size'])
              
    return sibsp_size.copy()


# In[17]:


def create_parch_size(input_df):
    use_columns = ['Parch']  
    
    td=[]
    for i in range(0, len(input_df)):

        td.append(input_df['Parch'][i]/4)          
        
    parch_size = pd.DataFrame(td, columns=['parch_size'])
              
    return parch_size.copy()


# In[18]:



def create_embarked_value(input_df):
    use_columns = ['Embarked']  
    
    td=[]
    for i in range(0, len(input_df)):

        if input_df['Embarked'][i]=='S':
            td.append(1)          
        elif input_df['Embarked'][i]=='Q':
            td.append(0.6)          
        elif input_df['Embarked'][i]=='C':
            td.append(0.2)          
        else:
            td.append(1)           
        
    embarked_value = pd.DataFrame(td, columns=['embarked_value'])
              
    return embarked_value.copy()


# In[19]:



def create_fare_value(input_df):
    use_columns = ['Fare']  
    
    td=[]
    for i in range(0, len(input_df)):

        td.append(input_df['Fare'][i]/100)         
             
    fare_value = pd.DataFrame(td, columns=['fare_value'])
              
    return fare_value.copy()


# In[20]:



def create_age_value(input_df):
    use_columns = ['Age']  
    
    td=[]
    for i in range(0, len(input_df)):

        td.append((input_df['Age'][i])/50)         
             
    age_value = pd.DataFrame(td, columns=['age_value'])
              
    return age_value.copy()


# In[21]:



def create_pclass_value(input_df):
    use_columns = ['Pclass']  
    
    td=[]
    for i in range(0, len(input_df)):

        if input_df['Pclass'][i]==3:
            td.append(0)          
        elif input_df['Pclass'][i]==2:
            td.append(0.4)          
        elif input_df['Pclass'][i]==1:
            td.append(0.8)          
        else:
            td.append(0)       
             
    pclass_value = pd.DataFrame(td, columns=['pclass_value'])
              
    return pclass_value.copy()


# In[22]:


def create_cabin_value(input_df):
    use_columns = ['Cabin']  
    
    td=[]
    for i in range(0, len(input_df)):
        first = str(input_df['Cabin'][i])[0]
        if first=='B':
            td.append(0)          
        elif first=='C':
            td.append(0.2)        
        elif first=='G':
            td.append(0.4)          
        elif first=='D':
            td.append(0.6)        
        elif first=='E':
            td.append(0.8)  
        else:
            td.append(1)       
             
    cabin_value = pd.DataFrame(td, columns=['cabin_value'])
              
    return cabin_value.copy()


# In[ ]:





# In[23]:


from contextlib import contextmanager
from time import time

@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


# In[24]:


from tqdm import tqdm

def to_feature(input_df):
    processors = [
        create_sex_value,
        create_family_size,
        create_sibsp_size,
        create_parch_size,           
        create_embarked_value,
        create_fare_value,
        create_age_value,
        create_pclass_value,
        create_cabin_value,
    ]
    
    out_df = pd.DataFrame()
    
    for func in tqdm(processors, total=len(processors)):
        with timer(prefix='create ' + func.__name__ + ' '):
            _df = func(input_df)
        
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df


# In[25]:


train_feat_df = to_feature(train_df)
test_feat_df = to_feature(test_df)


# In[26]:


train_feat_df2 = pd.DataFrame(train_feat_df)
train_feat_df2.to_csv('train_feat_df.csv')
#train_feat_df2.to_csv('train_feat_df.csv', index=False, header=False)
test_feat_df2 = pd.DataFrame(test_feat_df)
test_feat_df2.to_csv('test_feat_df.csv')
#test_feat_df2.to_csv('test_feat_df.csv', index=False, header=False)


# In[27]:


train_feat_df


# In[28]:


from sklearn.metrics import average_precision_score
import lightgbm as lgbm

def pr_auc(y_true, y_pred):

    score = average_precision_score(y_true, y_pred)
    return "pr_auc", score, True

def fit_lgbm(X, y, cv, params: dict=None, verbose=100):

    if params is None:
        params = {}

    models = []

    oof_pred = np.zeros_like(y, dtype=np.float)

    for i, (idx_train, idx_valid) in enumerate(cv): 

        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMClassifier(**params)
        
        with timer(prefix='fit fold={} '.format(i + 1)):
            clf.fit(x_train, y_train, 
                    eval_set=[(x_valid, y_valid)],  
                    early_stopping_rounds=verbose, 
                    eval_metric=pr_auc,
                    verbose=verbose)

        pred_i = clf.predict_proba(x_valid)[:, 1]
        oof_pred[idx_valid] = pred_i
        models.append(clf)

        print(f'Fold {i} PR-AUC: {average_precision_score(y_valid, pred_i):.4f}')

    score = average_precision_score(y, oof_pred)
    print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred, models        


# In[29]:


params = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'max_depth': 8,
    'n_estimators': 10000000,
    'colsample_bytree': .5,
}

y = train_df['Survived'].values


# In[30]:


train_df['Survived']


# In[31]:


y


# In[32]:


from sklearn.model_selection import StratifiedKFold

fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=71)
cv = list(fold.split(train_feat_df, y))


# In[33]:


train_feat_df


# In[34]:


oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params)


# In[35]:


def visualize_importance(models, feat_train_df):

    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')        .sum()[['feature_importance']]        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
    sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    fig.tight_layout()
    return fig, ax


# In[36]:


pred = np.array([model.predict_proba(test_feat_df.values)[:, 1] for model in models])
pred = np.mean(pred, axis=0)

#sub_df = pd.DataFrame({ 'target': pred })
#sub_df.to_csv(os.path.join(OUTPUT_DIR, 'titanic_submission.csv'), index=False)


# In[37]:


pred


# In[38]:


pred_life=[]

for i in range(len(pred)):
    if pred[i]<0.5:
        pred_life.append(0)
    else:
        pred_life.append(1)        

my_submit0 = ['PassengerId']
my_submit1 = ['Survived']


for i in range(0, len(pred)):
    my_submit0.append(str(892+i))
    my_submit1.append(str(pred_life[i]))

    
my_submit=[]

my_submit.append(my_submit0)
my_submit.append(my_submit1)

my_submit2=np.transpose(my_submit)

my_submit3 = pd.DataFrame(my_submit2)
my_submit3.to_csv('titanic_lgb6_submission.csv', index=False, header=False)


# In[ ]:




