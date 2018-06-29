# -*- coding: utf-8 -*-
from tf_idf_best import * 
from tf_idf import * 
from readData import *
from generate_docDict_queryDict import *
import argparse
import pandas as pd
parser = argparse.ArgumentParser(description='Process argument.')
parser.add_argument('-r', dest='use_feedback',action='store_true', help='use feedback', default=True)
parser.add_argument('-b', dest='use_best',action='store_true', help='use best version', default=True)
parser.add_argument('-i', dest='query_file', default='data/query-test.xml', help='The input query file.')
parser.add_argument('-o', dest='output_file', default='submit.csv', help='The output ranked list file.')
parser.add_argument('-m', dest='model_dir', default='data/model', help='The input model directory.')
parser.add_argument('-d', dest='ntcir_dir', default='data/CIRB010', help='The directiry of NTCIR documents.')
args = parser.parse_args()

#################################################
#data preprocessing
#################################################
print('------------------------------------')
print('dara preprocessing processing.....')
#read vocab.all
vocab_path = args.model_dir + '/vocab.all'
read_vocab_list(vocab_path) 

#read file-list
file_list_path = args.model_dir  + '/file-list'
read_file_list(file_list_path) 

#read query file
query_file_path = args.query_file
generate_doc_query_dict(query_file_path) 
           
#output file path
output_file_path = args.output_file

#feedback
do_revelance = args.use_feedback
print('dara preprocessing done.....')
print('------------------------------------')

#################################################
#rad data
#################################################
print('------------------------------------')
print('read data processing.....')

docDict = pd.read_csv('data/docDict.csv')
queryDict = pd.read_csv('data/queryDict_test.csv')
corpus = make_corpus(docDict, queryDict)
stopwords = []
with open('stopwords_zh.txt', 'r', encoding='utf-8') as f:
    for item in f.readlines():
        stopwords.append(item.strip('\n'))
        
print('read data done.....')
print('------------------------------------')  

     
#################################################
#main
#################################################

if args.use_best:
    print('------------------------------------')
    print('tf-idf best processing.....')
    vectors = tf_idf_best(corpus, stopwords)
    write_tf_idf_best(vectors, docDict, queryDict, do_revelance, output_file_path)
    print('------------------------------------')
    print('tf-idf best done.....')
else:
    print('------------------------------------')
    print('tf-idf processing.....')
    vectors = tf_idf(corpus, stopwords)
    write_tf_idf(vectors, docDict, queryDict, do_revelance, output_file_path)
    print('------------------------------------')
    print('tf-idf done.....')
    