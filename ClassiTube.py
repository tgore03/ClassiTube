import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


#data = np.loadtxt('USvideos_modified.csv', delimiter=",", dtype="str")
data = pd.read_csv('USvideos_modified.csv')
type_str_list = ['channel_title', 'tag_appeared_in_title', 'title', 'tags', 'description']
for column in type_str_list:
    data[column] = data[column].astype(str)

data['category_id'] = data['category_id'].astype(int)

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

#print(data[['channel_title', 'tag_appeared_in_title', 'tags', 'description']])
l = np.empty(shape=[len(data),], dtype='str')
j=0
for i,row in data.iterrows():
    row['title'] = row['title'].translate(non_bmp_map)

print(data['title'])

##print(data[['channel_title', 'tag_appeared_in_title', 'title', 'tags', 'description']])
##
##X = data.iloc[:, [5, 14, 15, 16]]
##Y = data[:,4]
##
##print(X)
##print(Y)
