# coding: utf-8
import pickle as pk
from argparse import ArgumentParser
from PLSA import *
from preprocessing import *
from classification import *
import numpy as np 
mydir = 'source'
parser = ArgumentParser()
parser.add_argument('-e', dest='use_dict',action='store_true', help='use dictionary', default=True)
parser.add_argument('-b', dest='use_best',action='store_true', help='use best version', default=True)
parser.add_argument('-d', dest='documents', default='source/data/doc.csv', help='The doc.csv')
parser.add_argument('-g', dest='group', default='source/data/groups.csv', help='The group.csv')
parser.add_argument('-o', dest='output_file', default='source/result/predict.csv', help='The output path of your classification')
args = parser.parse_args()

def main():
    total_iteration = 110
    word_filter = 160
    K = 50
    training_flag = 0
    
    print('data_preprocessing.............')
    doc_path = args.documents
    num_doc, num_voc, C_wi_di, word2idx = data_preprocessing(doc_path, word_filter)
    print('----data_preprocessing done----')
    print('num_doc:' + str(num_doc))
    print('num_voc:' + str(num_voc))
    print('#########################################################')
       
    if training_flag == 0:
        print('PLSA training.............')
        doc_gen_topic, topic_gen_word, p = initial_prob(num_doc, num_voc, K, C_wi_di)    
        PLSA_training(num_doc, num_voc, K, total_iteration, doc_gen_topic, topic_gen_word, p, C_wi_di)
        print('------PLSA trining done-------')
        print('#########################################################')
    else: 
        #################################################
        '''Reload doc_gen_topic and topic_gen_word kaggle private score: acc 0.57823'''
        ################################################# 
        path = mydir + '/data/doc_gen_topic.pk'
        with open(path, 'rb') as f:
            doc_gen_topic = pk.load(f)
            
        path = mydir + '/data/topic_gen_word.pk'
        with open(path, 'rb') as f:
            topic_gen_word = pk.load(f)
            
        path = mydir + '/data/C_wi_di.npy'   
        C_wi_di = np.load(path)
               
    do_using_dict = args.use_dict
    do_using_best = args.use_best          
    if do_using_dict or do_using_best:
        print('building dictionary.............')
        group_path = args.group
        class_word_list, class_name_list = load_group(group_path)
        class_name_word_list = build_best_dictionary(class_word_list, class_name_list, word2idx)
        print('----building dictionary done----')
        print('#########################################################')
             
    else:
        print('using group csv only.............')
        group_path = args.group
        class_word_list, class_name_list = load_group(group_path)
        class_name_word_list = class_word_list
        print('----building dictionary done----')
        print('#########################################################')


    print('documents classification.........')      
    output_path = args.output_file
    ans = []
    for doc_idx in range(num_doc):
        ans.append([str(doc_idx)])
        if do_using_dict or do_using_best:
            ans[doc_idx].append(documents_classify(doc_idx, word2idx, class_name_word_list, topic_gen_word, doc_gen_topic))
        else:
            ans[doc_idx].append(documents_classify_using_groupcsv(doc_idx, word2idx, class_name_word_list, topic_gen_word, doc_gen_topic))
    write_csv(output_path, ans)
    print('----documents classification done----')  

if __name__ == '__main__':
    main()






