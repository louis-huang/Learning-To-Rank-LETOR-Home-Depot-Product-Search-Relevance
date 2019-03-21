#this code combine all features together for building models later.
#Names: Gaoyuan Huang, Jiaxi Peng, Zheheng Mao
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
#read tfidf similarity
df = pd.read_csv("../input/predict-jiaxi-tfidf/df_all_tfidf.csv")
#combine brand
brand = pd.read_csv("../input/create-brank-features/feature_brand.csv")
data = pd.concat([df, brand], axis=1)
#combine spacy similarity
spacy = pd.read_csv("../input/575-spacy-similarity-and-common-words/df_all_spacy_sim.csv",sep = "\t")
spacy = spacy[['sim_in_title', 'sim_in_description', 'len_of_query']]
data = pd.concat([data, spacy], axis=1)
#combine common number
com_num = pd.read_csv("../input/csc-575-final-project-common-numbers/df_all_stemmed_cos_sim_num.csv")
com_num = com_num[['num_in_description','num_in_title', 'num_in_attribute']]
com_num.fillna("No Number appear",inplace=True)
com_num = pd.get_dummies(com_num,drop_first=True)
#combine common words
com_words = pd.read_csv("../input/fork-of-575-preprocess-2/feature_common_words.csv")
#combine cosine similarity created from raw term frequency matrxi
raw_cos = pd.read_csv("../input/575rawcosinesim/raw_cosine_similarity.csv")
final_data = pd.concat([data, raw_cos, com_num,com_words], axis=1)
#read training and testing
train = pd.read_csv("../input/csc-575-final-project-kaggle-winter-2019/train_new.csv",sep="\t")
test = pd.read_csv("../input/csc-575-final-project-kaggle-winter-2019/test_new.csv",sep="\t")
sub = pd.read_csv("../input/csc-575-final-project-kaggle-winter-2019/sample_submission_new.csv",sep="\t")
train = train[['search_term', 'relevance']]
#merge with traning and testing to create final datasets
train_df = pd.concat([final_data[:num_train], train],axis = 1)
test_df = pd.concat([final_data[num_train:].reset_index(drop = True), test],axis = 1, join_axes=[test.index])
train_df.to_csv("training.csv", index = None)
test_df.to_csv("testing.csv", index = None)
train_df.drop(["search_term"],axis = 1, inplace=True)
test_df.drop(["search_term","id","product_uid","product_title"], axis = 1, inplace=True)
train_df.to_csv("features_training.csv", index = None)
test_df.to_csv("features_testing.csv", index = None)

