# -*- coding: utf-8 -*-
from readData import * 
import pickle
import pandas as pd 


def generate_doc_query_dict(query_file_path): 

    with open('dict/doc_file_dict.pkl', 'rb') as f:
        doc_file_dict = pickle.load(f)
    
    #condition setting 
    producedoc_file_dict = 1
    produceQueryDict = 1
    
    readTrain = 0
    if readTrain == 1:
        read_query_fileName = 'data/query-train.xml'
        save_query_fileName = 'data/queryDict_train.csv'
    else:
        #read_query_fileName = 'data/query-test.xml'
        read_query_fileName = query_file_path
        save_query_fileName = 'data/queryDict_test.csv'
    
    
    '''===============Generate document dict============================================================================'''
    if producedoc_file_dict == 1:   
        docDict = []
        count = 1
        for key in doc_file_dict:
            #print(str(count) + '/' + str(len(doc_file_dict)))
            docFileName = 'data/' + doc_file_dict[key]['docFileName'] 
            docDict.append(doc2dict(docFileName))
            count = count + 1    
        docDict = pd.DataFrame(docDict)
        docDict.to_csv('data/docDict.csv', index=False, encoding='utf-8')
    else:
        docDict = pd.read_csv('data/docDict.csv')
    print('--------------------------------')
    print('Generate docDict done')
    
    '''===================Generate query============================================================================'''
    if produceQueryDict == 1:
        queryFileName = read_query_fileName
        queryDict = pd.DataFrame(query2dicts(queryFileName))
        queryDict.to_csv(save_query_fileName, index=False, encoding='utf-8')
    else:
        queryDict = pd.read_csv(save_query_fileName)
    numQuery = len(queryDict)
    print('--------------------------------')
    print('Generate queryDict done')



