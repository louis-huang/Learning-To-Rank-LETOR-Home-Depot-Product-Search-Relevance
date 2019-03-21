#This code corrects some error rows of attribute dataset.
#Names: Gaoyuan Huang, Jiaxi Peng, Zheheng Mao
#I found there were two rows having extra delimiters. So I first skipped them and then added them to the end of the csv file. Not sure how to add them back to where they should be.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import os
print(os.listdir("../input"))
#read data and skip error rows
attribute = pd.read_csv("../input/attributes_new.csv",sep = "\n", error_bad_lines=False)
#get values in the data
r1 = []
r2 = []
r3 = []
for idx, row in attribute.iterrows():
    vals = row.values[0].split("\t")
    r1.append(vals[0])
    r2.append(vals[1])
    r3.append(vals[2])
dt = pd.DataFrame({"product_uid":r1, "name":r2, "value":r3})
#save
dt.to_csv("attributes_new_modified.csv",sep = "\t",index = None)

