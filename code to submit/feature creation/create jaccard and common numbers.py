#This code create jaccard similarity and common numbers.
#Names: Gaoyuan Huang, Jiaxi Peng, Zheheng Mao
import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
print(os.listdir("../input/"))
#read data
df= pd.read_csv('../input/575-cosine-similarity/df_all_stemmed_cos_sim.csv')
#define jaccard calculation
def get_jaccard_sim(str1, str2): 
    try:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return np.nan

# if search term doesnt contain numbers ==> -1
# if any number in search term matches decriptions ==> 1
# if search term contains numbers, but numbers dont match descriptions ==>0
def common_num(str1,str2):
    try:
        count_common = np.nan
        for word in str1.split():
            if bool(re.match('[0-9]', word)) == True or bool(re.match('[a-z][0-9]', word)) == True or word.isdigit():
                count_common = 0
                if str2.find(word)>=0:
                    count_common+=1
                else:
                    count_common+= 0

        if count_common == 0:
            return ('no common number')
        elif count_common>0:
            return ('common number')
    except:
        return np.nan
#calculate common numbers
df['num_in_title'] = df[['search_term_tokens','product_title_tokens']].apply(lambda x: common_num(*x),axis = 1)
df['num_in_description'] = df[['search_term_tokens','product_description_tokens']].apply(lambda x: common_num(*x),axis = 1)
df['num_in_attribute'] = df[['search_term_tokens','attribute_tokens']].apply(lambda x: common_num(*x),axis = 1)
#calculate jaccard similarity
df['jc_title'] = df[['search_term_tokens','product_title_tokens']].apply(lambda x: get_jaccard_sim(*x),axis = 1)
df['jc_description'] = df[['search_term_tokens','product_description_tokens']].apply(lambda x: get_jaccard_sim(*x),axis = 1)
df['jc_attribute'] = df[['search_term_tokens','attribute_tokens']].apply(lambda x: get_jaccard_sim(*x),axis = 1)
df.to_csv('df_all_stemmed_cos_sim_num_jc.csv',index = False)
