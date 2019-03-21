#This code create similarity features using API from Spacy which uses wrod2vec to calculate similarity
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
#read data 
train_df =  pd.read_csv('../input/new_train.csv', sep = "\t")
train_df = train_df.fillna('')
test_df = pd.read_csv('../input/new_test.csv', sep = "\t")
test_df = test_df.fillna('')
#combine training and testing
df_all = pd.concat((train_df, test_df), axis=0, ignore_index=True)
#define function to do calculation
def spacy_sim(str1, str2):
    return nlp(str1).similarity(nlp(str2))
def crearte_features(samples):
    samples['product_info'] = samples['search_term']+"\t"+samples['product_title']+"\t"+samples['product_description']
    samples['sim_in_title'] = samples['product_info'].map(lambda x:spacy_sim(x.split('\t')[0],x.split('\t')[1]))
    samples['sim_in_description'] = samples['product_info'].map(lambda x:spacy_sim(x.split('\t')[0],x.split('\t')[2]))
    samples['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    return samples
#run calculation and save
df_all = crearte_features(df_all)
df_all.to_csv("df_all_spacy_sim.csv",sep = "\t",index = False)
train_df = df_all.iloc[:num_train]
train_df[['sim_in_title', 'sim_in_description',  'len_of_query','relevance']].to_csv("feature_train_spacy.csv",index = None, sep = "\t")
test_df = df_all.iloc[num_train:]
test_df[['sim_in_title', 'sim_in_description', 'len_of_query']].to_csv("feature_test_spacy.csv",index = None, sep = "\t")
