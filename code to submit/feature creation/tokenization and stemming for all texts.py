#This code did tokenization and stemming for all texts
#Names: Gaoyuan Huang, Jiaxi Peng, Zheheng Mao

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
from collections import defaultdict

#read data
train_df =  pd.read_csv('../input/new-575-train-test/new_train.csv', sep = "\t")
train_df = train_df.fillna('')
test_df = pd.read_csv('../input/new-575-train-test/new_test.csv', sep = "\t")
test_df = test_df.fillna('')
#merge trianing and testing
df_all = pd.concat((train_df, test_df), axis=0, ignore_index=True)
#prepare for tokenization
table = str.maketrans(dict.fromkeys(string.punctuation))
porter = nltk.PorterStemmer()
#nlp = spacy.load('en_core_web_lg')
num_train = train_df.shape[0]
random_st = 1024
#define function to tokenize
def process_3(text):
    text = text.lower().translate(table)
    tokens = word_tokenize(text)
    return " ".join(tokens)
#tokenization without stemming
df_all["product_description_tokens"] = df_all["product_description"].apply(lambda x : process_3(x))
df_all["product_title_tokens"] = df_all["product_title"].apply(lambda x : process_3(x))
df_all["search_term_tokens"] = df_all["search_term"].apply(lambda x : process_3(x))
df_all["attribute_tokens"] = df_all["attribute"].apply(lambda x : process_3(x))
#tokenization with stemming
df_all["product_description_stemmed"] = df_all["product_description_tokens"].apply(lambda x : " ".join([porter.stem(w) for w in x.split(" ") if w.isalpha()]))
df_all["product_title_stemmed"] = df_all["product_title_tokens"].apply(lambda x : " ".join([porter.stem(w) for w in x.split(" ") if w.isalpha()]))
df_all["search_term_stemmed"] = df_all["search_term_tokens"].apply(lambda x : " ".join([porter.stem(w) for w in x.split(" ") if w.isalpha()]))
df_all["attribute_stemmed"] = df_all["attribute_tokens"].apply(lambda x : " ".join([porter.stem(w) for w in x.split(" ") if w.isalpha()]))
#save output
df_all.to_csv("df_all_toeknized.csv",index = None, sep = "\t")
