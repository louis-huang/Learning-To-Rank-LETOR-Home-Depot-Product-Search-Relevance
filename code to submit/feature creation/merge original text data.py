#This code merge all text files togeter.
#Names: Gaoyuan Huang, Jiaxi Peng, Zheheng Mao
import pandas as pd
import numpy as np
#read cleaned attributes
attr = pd.read_csv("/Users/louis/Desktop/MSPA/575Information retrieval system/575Final/data/attributes_new_modified.csv",
                sep = "\t", error_bad_lines=False)
#drop na
attr.dropna(axis = 0, inplace = True)
#read training data, description data and test data
train = pd.read_csv("train_new.csv",sep = "\t")
description = pd.read_csv("product_descriptions_new.csv",sep = "\t")
test = pd.read_csv("test_new.csv",sep = "\t")
#get all products id
ids = set(attr.product_uid.unique())
#for each product, we concatenate texts in attributes to save as a string
all_rows = []
for uid in ids:
    cur_row = []
    cur_dt = attr.loc[attr.product_uid == uid,:]
    attribute = ". ".join(cur_dt.value.values)
    cur_row = [int(uid),attribute]
    all_rows.append(cur_row)
new_attr = pd.DataFrame(all_rows, columns=["product_uid","attribute"])
new_attr.sort_values(by = ["product_uid"],inplace = True)
new_attr.reset_index(drop = True, inplace=True)
#merge description and attributes
new_description = description.merge(new_attr, on = "product_uid", how = "outer")
new_description.reset_index(drop = True, inplace=True)
#save product descriptions
new_description.to_csv("new_description.csv", sep = "\t", index = None)
#merge product description with training data
new_train = train.merge(new_description, on = "product_uid", how = "left")
#merge product description with testing data
new_test = test.merge(new_description, on = "product_uid", how = "left")
new_train.reset_index(inplace = True, drop = True)
new_test.reset_index(inplace = True, drop = True)
#reorder columns
new_train = new_train[['id', 'product_uid', 'product_title', 'product_description', 'attribute', 'search_term', 'relevance']]
new_test = new_test[['id', 'product_uid', 'product_title','product_description', 'attribute', 'search_term']]
#save files
new_train.to_csv("new_train.csv", index = None, sep = "\t")
new_test.to_csv("new_test.csv", index = None, sep = "\t")
