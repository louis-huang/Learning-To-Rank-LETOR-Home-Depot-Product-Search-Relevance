#this code calculate cosine similarity using raw tf and cosine similarity using tf-idf
#Names: Gaoyuan Huang, Jiaxi Peng, Zheheng Mao

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import numba
import os
print(os.listdir("../input/"))
import numba
#read all tokenized data and fill na with ""
df_all = pd.read_csv('../input/df_all_toeknized.csv',sep = "\t")
df_all.search_term_stemmed.fillna(df_all.search_term_tokens, inplace=True)
df_all.fillna("",inplace=True)
#save this file
df_all.to_csv("df_all_toeknized_filled_na_search_term.csv",index = False)
#prepare calculation
df_all['cos_description'] = 0
df_all['cos_title'] = 0
df_all['cos_attribute'] = 0
df_all['cos_description_tfidf'] = 0
df_all['cos_title_tfidf'] = 0
df_all['cos_attribute_tfidf'] = 0
#define helper functions to calculate similarity
def get_cosine_sim(*strs): 
    try:
        vectors = [t for t in get_vectors(*strs)]
        return cosine_similarity(vectors)[0][1]
    except:
        return 0
    
def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_vectors_2(text):
    cv = CountVectorizer(max_features=5000)
    cv.fit(text)
    return cv.transform(text)


def get_tfidf(text, q):
    cv = TfidfVectorizer()
    cv.fit(text)
    return cv.transform(text),cv.transform(q)

#calculate similarity using stemmed texts and their raw term frequency matrix
df_all["cos_description_stemmed"] = df_all[["product_description_stemmed","search_term_stemmed"]].apply(lambda x:get_cosine_sim(*x),axis = 1) 
df_all["cos_title_stemmed"] = df_all[["product_title_stemmed","search_term_stemmed"]].apply(lambda x:get_cosine_sim(*x),axis = 1) 
df_all["cos_attribute_stemmed"] = df_all[["attribute_stemmed","search_term_stemmed"]].apply(lambda x:get_cosine_sim(*x),axis = 1) 
#save file
df_all.to_csv("df_all_stemmed_cos_sim.csv",index = False)
#get all unique product information
product = df_all.drop_duplicates("product_uid")
#get all texts
total_des = product.product_description_stemmed.values
total_search = df_all.search_term_stemmed.values
total_title = product.product_title_stemmed.values
total_attr = product.attribute_stemmed.values
#make tf-idf matrix
term_description_matrix, des_search_mat = get_tfidf(total_des,total_search)
term_attr_matrix, attr_search_mat = get_tfidf(total_attr,total_search)
term_title_matrix, title_search_mat = get_tfidf(total_title,total_search)
#save matrix as features
product["des_mat"] = [i for i in term_description_matrix]
product["attr_mat"] = [i for i in term_attr_matrix]
product["title_mat"] = [i for i in term_title_matrix]
df_all["search_des_mat"] = [i for i in des_search_mat]
df_all["search_attr_mat"] = [i for i in attr_search_mat]
df_all["search_title_mat"] = [i for i in title_search_mat]

#prepare data for merging with original data set
product_merge = product[["product_uid","des_mat","attr_mat","title_mat"]]
#merge
new_df_all = df_all.merge(product_merge,on = "product_uid",how = "outer")
#define function to calculate cosine similarity using tf-idf
@numba.jit
def cos(*x):
    return cosine_similarity(x[0].reshape(1,-1), x[1].reshape(1,-1))[0][0]
#do the calculation
new_df_all["cos_description_tfidf"] = new_df_all[["search_des_mat","des_mat"]].apply(lambda x:cos(*x),axis = 1) 
new_df_all["cos_title_tfidf"] = new_df_all[["search_title_mat","title_mat"]].apply(lambda x:cos(*x),axis = 1) 
new_df_all["cos_attribute_tfidf"] = new_df_all[["search_attr_mat","attr_mat"]].apply(lambda x:cos(*x),axis = 1)
#save files
new_df_all[["cos_description_tfidf","cos_title_tfidf","cos_attribute_tfidf"]].to_csv("df_all_tfidf.csv",index = False)
#new_df_all.to_csv("df_all_tfidf_with_matrix.csv",index = False)
