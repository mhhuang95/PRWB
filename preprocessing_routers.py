#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
from gensim.models import KeyedVectors, word2vec, LdaModel

import _pickle as cPickle

import codecs, sys, glob


def extract_features(in_file='story_clusters.txt', features_file='feature_words.txt', filtered_category_list=[]):
    f = codecs.open(in_file, 'r', encoding='utf-8', errors='ignore')
    clusters = f.read().strip('%%%').split('\n%%%')
    docs = []
    labels = []
    word_length = 0
    document_count = 0
    for i, cluster in enumerate(clusters):
        if (not filtered_category_list) or (cluster.strip().split('\n')[0] in filtered_category_list):
            for ln in cluster.strip().split('\n')[1:]:
                docs.append(ln)
                labels.append(i)
                document_count = document_count + 1
                word_length = word_length + len(ln.split(' '))

    print("raw categories: %d" % len(clusters))
    print("document count: %d" % document_count)
    print("average words: %d" % (word_length / document_count))

    vectorizer = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b")
    vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names()
    fw = open(features_file, 'wb')
    fw.write(u'\n'.join(features).encode('utf-8'))
    fw.close()


def build_word2vec_index(feature_file='feature_words.txt', word_vecs_file='GoogleNews-vectors-negative300.bin',
                         out_dic='word_vecs.pkl'):
    words = open(feature_file).read().strip().split('\n')
    dic = dict()
    model = KeyedVectors.load_word2vec_format(word_vecs_file, binary=True)
    for w in words:
        try:
            v = model[w]
            dic[w] = v
        except:
            continue  # whatever
            try:
                v = model[correct(w)]
                dic[w] = v
            except:
                print(w + ' not existed in word2vec model')
                continue
    fw = open(out_dic, 'wb')
    cPickle.dump(dic, fw)
    fw.close()


def convert_d2_format(in_file='story_clusters.txt', embedding_dic='word_vecs.pkl', embedding_dim_size=300,
                      weighting_type='tf', d2_vocab='story_cluster.d2s.vocab0', d2s_file='story_cluster.d2s',
                      filtered_category_list=[]):
    word2vec_dic = cPickle.load(open(embedding_dic, 'rb'))
    vocab = word2vec_dic.keys()
    # clusters = open(in_file).read().strip('%%%').split('\n%%%')
    f = codecs.open(in_file, 'r', encoding='utf-8', errors='ignore')
    clusters = f.read().strip('%%%').split('\n%%%')
    docs = []
    labels = []
    idx = 0
    for i, cluster in enumerate(clusters):
        if (not filtered_category_list) or (cluster.strip().split('\n')[0] in filtered_category_list):
            for ln in cluster.strip().split('\n')[1:]:
                docs.append(ln)
                labels.append(idx)
            idx += 1
            print(cluster.strip().split('\n')[0])

    temp = list(zip(docs, labels))
    random.shuffle(temp)
    docs, labels = zip(*temp)

    fw1 = open('reuters_cluster.d2s.labels', 'w')
    fw1.write('\n'.join(str(j) for j in labels))
    fw1.close()

    for i in range(8):
        print(i, np.sum(np.array(labels) == i))

    if weighting_type == 'tf':
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', vocabulary=vocab)
    if weighting_type == 'tfidf':
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', vocabulary=vocab)
    X = vectorizer.fit_transform(docs)
    print(X.shape)
    fw = open(d2s_file, 'w')
    for i in range(X.shape[0]):
        fw.write(str(embedding_dim_size) + '\n')
        nonzero_ids = X[i].nonzero()
        if (len(nonzero_ids[0]) > 0):
            fw.write(str(len(nonzero_ids[0])) + '\n')
            fw.write(' '.join(str(X[i][0, j]) for j in nonzero_ids[1]) + '\n')
            fw.write(' '.join(str(j + 1) for j in nonzero_ids[1]) + '\n')
        else:
            print >> sys.stderr, "empty document found!"
            fw.write('1\n')
            fw.write('1\n')
            fw.write('0\n')
    fw.close()
    fw = open(d2_vocab, 'w')
    fw.write(str(embedding_dim_size) + ' ' + str(len(vocab)) + '\n')
    fw.write('\n'.join(' '.join(str(v) for v in word2vec_dic[w]) for w in vocab))
    fw.close()


if __name__ == '__main__':
    dataset = 'reuters'

    vec_dim = 300
    word_vecs='glove.6B.300d.bin'

    cluster_file = 'Dataset/' + dataset + '_clusters.txt'
    vec_dic = dataset + '_word_vecs.pkl'

    reuters_r10_categories = ['crude',   'interest', 'money-fx',  'ship', 'trade']
    category_list = reuters_r10_categories


    extract_features(in_file=cluster_file, features_file='reuters.terms',
                     filtered_category_list=reuters_r10_categories)
    build_word2vec_index(feature_file='reuters.terms', word_vecs_file=word_vecs, out_dic=vec_dic)
    convert_d2_format(in_file=cluster_file, embedding_dic=vec_dic, embedding_dim_size=vec_dim,
                      weighting_type='tfidf', d2_vocab=dataset + '_cluster.d2s.vocab0',
                      d2s_file=dataset + '_cluster.d2s', filtered_category_list=reuters_r10_categories)