# -*- coding: utf-8 -*-
import pickle
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import csv
from sklearn.metrics.pairwise import cosine_similarity
import time
            
#################################################
#轉成稀疏矩陣的型式
#################################################
def sparse_matrix(tfidf_matrix):
    tfidf_matrix = np.array(tfidf_matrix).T #(doc_id, token_index, tf-idf weight)
    row = tfidf_matrix[0]   
    col = tfidf_matrix[1]
    freq = tfidf_matrix[2]   #tf-idf weight
    return csr_matrix((freq, (row, col)))

#################################################
#計算tf-idf
#################################################
def tfidf(doc_i, tok_j, freq, doclen, avgdoclen, idf):
    k1 = 1.6
    b = 0.9
    normalizer = 1 - b + b * doclen[doc_i] / avgdoclen               #web ir VSM_ppt 
    freq = freq * (k1 + 1) / (freq + k1 * normalizer) * idf[tok_j]   #bm25
    return (doc_i, tok_j, freq)

#################################################
#計算tf、df
#################################################
def tf_idf_best(corpus, stopwords):
    raw_tf = []
    df = [0] * 1000000
    N = len(corpus)
    doclen = [0] * N
             
    with open('dict/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    vocabulary_size = len(vocabulary)
    
    for doc_i, doc in enumerate(corpus):
        if doc_i % 5000 == 0:   
            print(str(doc_i) + '/' + str(N))
        doclen[doc_i] = len(doc)
        tokens = [tok for tok in doc.split() if tok not in stopwords]
    
        doc_term = defaultdict(int)
    
        for tok in tokens:
            if tok not in vocabulary:
                tok_ind = vocabulary.setdefault(tok, vocabulary_size)
                vocabulary_size = vocabulary_size + 1
                
            tok_ind = vocabulary[tok]
            doc_term[tok_ind] = doc_term[tok_ind] + 1
    
        for tok_ind in doc_term.keys():
            df[tok_ind] += 1
    
        raw_tf.extend([(doc_i, tok_j, freq) for tok_j, freq in doc_term.items()])
    

    #################################################
    #計算idf、tf-idf
    #################################################
    tfidf_matrix = [0] * len(raw_tf)

    avgdoclen = sum(doclen) / N
    df = np.array(df)
    idf = np.log((N - df + 0.5) / (df + 0.5))
    for i, (doc_i, tok_j, freq) in enumerate(raw_tf):
        tfidf_matrix[i] = tfidf(doc_i, tok_j, freq, doclen, avgdoclen, idf)
        
    return sparse_matrix(tfidf_matrix)


def write_tf_idf_best(vectors, docDict, queryDict, do_revelance, output_file_path):
    #################################################
    #relevent feedback parameters seeting
    #################################################
    treshold = 0.2
    i_N = 1
    beta = 0.55
    a = 1
    maxN = 10
    n_query = len(queryDict)
    #################################################
    #write data
    #################################################
    with open(output_file_path, 'w',  newline='') as csvfile:
        fieldnames = ['query_id', 'retrieved_docs']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for qidx in range(n_query):
            query_vector = vectors[-n_query + qidx]
            dists = cosine_similarity(vectors, query_vector).T[0]
            top100 = dists.argsort()[::-1][0:100]
    
            #################################################
        		#relevent feedback
        	 #################################################
            if do_revelance == 1:
                print('doing revelance feedback...')
                for i in range(i_N):
                    topN = min(len(dists[dists > treshold]), maxN)
                    query_vector = a * query_vector + beta / topN * np.sum(vectors[top100[:topN]], axis=0)
                    dists = cosine_similarity(vectors, query_vector).T[0]
                    top100 = dists.argsort()[::-1][0:100]
    
            top100 = list(docDict.id[top100].fillna(' ').as_matrix())
            writer.writerow([queryDict.number[qidx][-3:]," ".join(top100)])
            print('write querID:' + str(qidx) + 'to csv ')
    
      
        