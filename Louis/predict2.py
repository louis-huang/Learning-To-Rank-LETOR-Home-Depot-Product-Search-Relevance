#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import gensim 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer 
import string
import spacy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet,Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import math

train_df =  pd.read_csv('/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/data/new features/features_training.csv')
test_df = pd.read_csv('/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/data/new features/features_testing.csv')

train_df.drop(["cos_sim"],axis = 1, inplace=True)
test_df.drop(["cos_sim"],axis = 1, inplace=True)


x_train = train_df.iloc[:,:-1].values
y_train = train_df.iloc[:,-1:].values
x_test = test_df[:].values

train_matrix = xgb.DMatrix(data=x_train,label=y_train)
test_matrix = xgb.DMatrix(data = x_test)

random_st = 1024

x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=0.3, random_state=random_st)


models = [xgb.XGBRegressor(colsample_bytree = 1, gamma = 0.4, max_depth=4, subsample=0.9,n_estimators= 200,random_state=random_st),
            GradientBoostingRegressor(learning_rate=0.1,min_samples_split=3, subsample=0.8,random_state = random_st,n_estimators=200)
          ]
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def make_result(model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    sub = pd.read_csv("/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/data/sample_submission_new.csv")
    sub["relevance"] = y_pred
    sub.to_csv('/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/result/submission.csv', index=False)

loss = make_scorer(rmse,greater_is_better=False)

for model in models:
    model_name = type(model).__name__
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring=loss)

    print("{} has been trained. Average RMSE:{}".format(model_name, np.mean(scores)))



model_gb = GradientBoostingRegressor(learning_rate=0.1,min_samples_split=3, subsample=0.8,\
                                  random_state = random_st,n_estimators=200)
model_gb.fit(x_tr, y_tr)


print(sqrt(mean_squared_error(y_te, model_gb.predict(x_te))))

#y_pred_2 = model_gb.predict(x_test)

print(model_gb.feature_importances_)



make_result(xgb.XGBRegressor(colsample_bytree = 1, gamma = 0.4, max_depth=4, subsample=0.9,n_estimators= 200,random_state=random_st))
# In[149]:


y_tr_1 = model_xgb.predict(x_train)
y_tr_2 = model_gb.predict(x_train)


# In[150]:


x_tr = np.stack((y_tr_1,y_tr_2)).T


# In[166]:


model = LinearRegression()
model.fit(x_tr, y_train)
print(sqrt(mean_squared_error(y_train, model.predict(x_tr))))


# In[167]:


model.coef_


# In[168]:


y_te_1 = model_xgb.predict(x_test)
y_te_2 = model_gb.predict(x_test)
x_te = np.stack((y_te_1,y_te_2)).T
y_pred_3 = model.predict(x_te)


# In[169]:


y_pred_3.shape


# In[171]:


y_pred_3[:20]


# In[159]:


sub  = pd.read_csv("/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/data/sample_submission_new.csv")

sub["relevance"] = y_pred_1
sub.to_csv('submission_linear.csv',index=False)



params = {'gamma':[i/10.0 for i in range(2,8)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [3,4,5],'n_estimators':[100,200,300]}

xgb_model = xgb.XGBRegressor(nthread = -1,)
clf = GridSearchCV(xgb_model, params)
clf.fit(x_train,y_train)
print(clf.best_score_)
print(clf.best_params_)

