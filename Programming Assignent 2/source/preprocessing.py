# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np  
from string import punctuation
import re
mydir = 'source'

def data_preprocessing(doc_path, word_filter):
    #################################################
    '''read stopword'''
    #################################################
    stopword = []
    path = mydir + '/data/stop.txt'
    with open(path) as f:
        data = f.readlines()
        for line in data:
            stopword.append(line.rstrip())
            
    regex = re.compile('[%s]' % re.escape(punctuation))

    #################################################
    '''read document & group'''
    #################################################  
    docs = pd.read_csv(doc_path)       

    #################################################
    '''document preprocessing'''
    ################################################# 
    docs_token_list = []
    for doc in docs.content:
        removed_newline = doc.replace("\\n", " ").lower()
        removed_pun = regex.sub('', removed_newline)
        doc_token_list = [c for c in removed_pun.split(' ') if c not in stopword]
        docs_token_list.append(list(filter(None, doc_token_list)))

    #################################################
    '''build word count(in collection && build word count(in per doc))'''
    ################################################# 
    word_count = {}
    word_count_per_doc_list = []
    for doc in docs_token_list: 
        word_count_per_doc = {}
        for token in doc:
            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1
                
            if token in word_count_per_doc:
                word_count_per_doc[token] += 1
            else:
                word_count_per_doc[token] = 1
        word_count_per_doc_list.append(word_count_per_doc)

    #################################################
    '''
    build dictionary 
    word2idx => word2idx["worse"] = 41
    idx2word =>  idx2word[41] = 'worse'
    '''
    ################################################# 
    word2idx = {}
    idx2word = {}
    
    index = 0;
    for word in word_count.keys():
        if word_count[word] > word_filter:
            word2idx[word] = index
            idx2word[index] = word
            index += 1
    
    num_voc = len(word2idx)   #3465
    num_doc = len(docs)        
    #################################################
    '''build document-word_count matrix'''
    #################################################
    C_wi_di = np.zeros([num_doc, num_voc])
    for word in word2idx.keys():
        j = word2idx[word]
        for i in range(0, num_doc):
            if word in word_count_per_doc_list[i]:
                C_wi_di[i, j] = word_count_per_doc_list[i][word]
    path =  mydir + '/data/C_wi_di.npy'           
    np.save(path, C_wi_di)
    return num_doc, num_voc, C_wi_di, word2idx
