# -*- coding: utf-8 -*-
import pickle
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import csv
          
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
    k1 = 1.0
    b = 0.75
    normalizer = 1 - b + b * doclen[doc_i] / avgdoclen               #web ir VSM_ppt 
    freq = freq * (k1 + 1) / (freq + k1 * normalizer) * idf[tok_j]   #bm25
    return (doc_i, tok_j, freq)

#################################################
#計算tf、df
#################################################
def tf_idf(corpus, stopwords):
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



def write_tf_idf(vectors, docDict, queryDict, do_revelance, output_file_path):
    #################################################
    #relevent feedback parameters seeting
    #################################################
    treshold = 0.2
    i_N = 1
    beta = 0.55
    maxN = 10
    n_query = len(queryDict)
    n_doc = len(docDict)
    #################################################
    #write data
    #################################################
    with open(output_file_path, 'w',  newline='') as csvfile:
        fieldnames = ['query_id', 'retrieved_docs']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        
        for qidx in range(n_query):
            query_vector = np.array(vectors[-n_query + qidx].toarray())
            
            score_list = []
            for didx in range(n_doc):
                score_list.append(np.dot(query_vector[0], np.array(vectors[didx].toarray())[0]))
    
            doc_idx_list = []
            doc_score_list = []
            for idx, val in enumerate(score_list):
                doc_idx_list.append(idx)
                doc_score_list.append(val)
            doc_score_list, doc_idx_list = (list(t) for t in zip(*sorted(zip(doc_score_list, doc_idx_list))))
            top100 = doc_idx_list[::-1][0:100]
            retrieved_docs = ''
            for idx in range(100):
                if idx == 0:
                    retrieved_docs =  docDict.id[top100[idx]]
                else:
                    retrieved_docs = retrieved_docs + ' ' + docDict.id[top100[idx]]
            #################################################
        		#relevent feedback
        	 #################################################      
            if do_revelance == 1:
                print('doing revelance feedback...')
                for _ in range(i_N):
                    count_score = 0
                    for score in score_list:
                        if score > treshold:
                            count_score = count_score + 1
                        
                    topN = min(count_score, maxN)
    
                    query_vector = query_vector + beta / topN * np.sum(vectors[top100[:topN]], axis=0)
                    score_list = []
                    for didx in range(n_doc):
                        score_list.append(np.dot(query_vector[0], np.array(vectors[didx].toarray())[0]))
            
                    doc_idx_list = []
                    doc_score_list = []
                    for idx, val in enumerate(score_list):
                        doc_idx_list.append(idx)
                        doc_score_list.append(val)
                    doc_score_list, doc_idx_list = (list(t) for t in zip(*sorted(zip(doc_score_list, doc_idx_list))))
                    top100 = doc_idx_list[::-1][0:100]
                    retrieved_docs = ''
                    for idx in range(100):
                        if idx == 0:
                            retrieved_docs =  docDict.id[top100[idx]]
                        else:
                            retrieved_docs = retrieved_docs + ' ' + docDict.id[top100[idx]]
             
            writer.writerow([queryDict.number[qidx][-3:],retrieved_docs])
            print('write querID:' + str(qidx) + 'to csv ')
