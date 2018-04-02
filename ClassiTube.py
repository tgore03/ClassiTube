import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


#Read Data

#data = np.loadtxt('USvideos_modified.csv', dtype=delimiter=',', usecols=(15), skiprows=1)

data = pd.read_csv('USvideos_modified.csv')
X_names = ['tags']
Y_names = ['category_id'] 
for column in X_names:
    data[column] = data[column].astype(str)

X = data[X_names]
Y = data[Y_names]

#Preprocess the data
for i, row in X.iterrows():
    row['tags'] = row['tags'].replace('|', ' ')

#TFIDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(X['tags'])
print(tfidf.shape)











##print(data[['channel_title', 'tag_appeared_in_title', 'title', 'tags', 'description']])
##
##X = data.iloc[:, [5, 14, 15, 16]]
##Y = data[:,4]
##
##print(X)
##print(Y)

##non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
##l = np.empty(shape=[len(data),], dtype='str')
##j=0
##for i,row in data.iterrows():
##    row['title'] = row['title'].translate(non_bmp_map)
