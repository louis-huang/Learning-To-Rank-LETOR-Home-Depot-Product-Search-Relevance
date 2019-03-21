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

random_st = 1024


x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=0.3, random_state=random_st)



m1  = GradientBoostingRegressor(max_features=0.8, min_samples_split = 5,subsample= 0.9,
    learning_rate = 0.02, max_depth = 7, min_samples_leaf = 9, n_estimators = 300)

m2 = xgb.XGBRegressor(subsample=1.0, max_depth=5, colsample_bytree=0.8,n_estimators=300)

y_tr_1, y_tr_2 = m1.fit(x_train, y_train).predict(x_train), m2.fit(x_train, y_train).predict(x_train)

m3 = RandomForestRegressor(max_depth=5, n_estimators=300,max_features=15)

y_tr_3 = m3.fit(x_train, y_train).predict(x_train)

x_tr_2 = np.stack((y_tr_1,y_tr_2,y_tr_3)).T


model = ElasticNet(l1_ratio=0.2, alpha=0.01)

model.fit(x_tr_2, y_train)

y_pred_1, y_pred_2 = m1.predict(x_test), m2.predict(x_test)

y_pred_3 = m3.predict(x_test)

x_te_2 = np.stack((y_pred_1,y_pred_2,y_pred_3)).T

y_pred = model.predict(x_te_2)

sub  = pd.read_csv("/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/data/sample_submission_new.csv")
sub["relevance"] = y_pred
sub.to_csv('/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/result/submission_stacking.csv',index=False)