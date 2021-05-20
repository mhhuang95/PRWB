#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import io
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from collections import Counter

from keras.preprocessing.text import Tokenizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def read_rawdata():

    classes = ['athletics', 'cricket', 'football', 'rugby', 'tennis']
    
    texts = []
    labels = []

    for label, news_class in enumerate(classes):
        filepath = './Dataset/bbcsport/'+ news_class + '/'
        
        idx = 0
        for file in os.listdir(filepath):
            
#             print(news_class, file, idx)
            text = []
            
            with open(filepath + file, 'r') as f:
                text = f.read().replace('\n', ' ').split('.')[0]
          
            texts.append(text)
            labels.append(label)

            idx += 1
            
#             if idx == 2:
#                 break
                
#     print(texts, labels)
#     print(len(texts), len(labels))
    return texts, labels

def remove_punc(texts):
    punc = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    new_texts = []
    for text in texts:
        for c in text:
            if c in punc:
                text = text.replace(c,' ')
        new_texts.append(text)
    return new_texts


def remove_stopwords(texts,stop_list):
    texts_nswds = []
    for text in texts:
        word_list = text.split() # turn each text to a list of words
        new_text = []
        for word in word_list:
            if word.lower() not in stop_list:
                new_text.append(word.lower())
        texts_nswds.append(new_text)
    return texts_nswds

def load_vectors(fname, size=None):
    fin = io.open(fname, 'r', newline='\n', errors='ignore')
#     n, d = map(str, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        if size and i >= size:
            break
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype='f8')
        i += 1
    print("Vectors loaded!")
    return data

def lemmatize(text_list):
    lmtzer = WordNetLemmatizer()
    
    new_texts = [] # list of list of words
     
    for list in text_list: # text is a list of words
        new_text = [] # list of words
        for word in list:
            ld_wrd = lmtzer.lemmatize(word)
            new_text.append(ld_wrd)
        new_texts.append(new_text)
        
    return new_texts

def write_matrix_to_textfile(a_matrix, f):

    def compile_row_string(a_row):
        return str(a_row).strip(']').strip('[').replace('\n', '')

    for row in a_matrix:
        f.write(compile_row_string(row)+'\n')
    return True

def solve_dis_opt(a, X):
    n = X.shape[1]
    minval = a[0]*a[1]*np.linalg.norm(X[:, 0] - X[:, 1])**2 /(a[0]+a[1])
    res_i, res_j = 0, 1
    for i in range(n-1):
        for j in range(i+1, n):
            val = a[i]*a[j]*np.linalg.norm(X[:, i] - X[:, j])**2 /(a[i]+a[j])
            if minval > val:
                minval = val
                res_i, res_j = i,j
    return res_i, res_j
            

def merge(X, a, target_n):
    
    assert target_n >= 2
    
    d = X.shape[0]
    n = X.shape[1]
    if n <= target_n:
        return X, a, False
    else:
        for k in range(n - target_n):
            i,j = solve_dis_opt(a, X)
            new_a_ele, new_x_ele = a[i] + a[j],  (a[i] * X[:,i] + a[j] * X[:,j] )/(a[i] + a[j])
            a = np.concatenate((a[:i], a[i+1:j], a[j+1:], np.array([new_a_ele])))
            X = np.concatenate((X[:,:i], X[:, i+1:j], X[:, j+1:], np.array(new_x_ele).reshape([d, 1])), axis=1)
        return X, a, True
    

def preprocessingBBCnews_tfidf(texts, num_file=None, target_n=None):
   
    lens_raw = np.array([len(text.split()) for text in texts])        
    if num_file:
        texts = texts[:num_file]
    
    num_texts = len(texts)
    
    texts_rc = remove_punc(texts)
         
    f = open("Dataset/stoplist2.txt", "r")
    stop_list = f.read()
    step_list = list(stop_list.split())

    texts_rs = remove_stopwords(texts_rc,stop_list)
    
    texts_rs = lemmatize(texts_rs)
    
    dct = Dictionary(texts_rs)
    print(dct[10])
    corpus = [dct.doc2bow(line) for line in texts_rs]
#     print(corpus)
    
    tfidf = TfidfModel(corpus)
    
#     print(tfidf[corpus[4]])
    
    filename_300d = 'glove.6B.300d.txt' 
    
    dictionary = load_vectors(filename_300d)
    d = len(dictionary['the'])
    
    with open("BBCsportdata_d300_n16_tfidf.d2s",'w') as f:
        for i in range(num_texts):
            a = []
            X = []
            for idx, prob in tfidf[corpus[i]]:
                if dct[idx] in dictionary.keys():
                    a.append(prob)
                    X.append(dictionary[dct[idx]])
            
            a = np.array(a)
            a = a / np.sum(a)
            X = np.array(X).T
            print(i, a.shape, X.shape)
            if target_n:
                X, a,flag = merge(X, a, target_n)
                if flag:
                    print(X.shape, a.shape)
            
            f.write(str(len(a))) 
            f.write('\n')
            write_matrix_to_textfile(a.reshape([1, len(a)]), f)
            write_matrix_to_textfile(X, f)
                
    print("Data preprocessing done!")
    return

if __name__ == "__main__":
    
    texts, labels = read_rawdata()
    
    temp = list(zip(texts, labels)) 
    random.shuffle(temp) 
    texts, labels = zip(*temp) 
    
    with open("BBCsportdata_d300_n16_tfidf_labels.d2s",'w') as f:
        for i in range(len(labels)):
            f.write(str(labels[i]))
            f.write('\n')
    
    
    preprocessingBBCnews_tfidf(texts, target_n=16)
    
