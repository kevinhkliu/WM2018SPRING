# -*- coding: utf-8 -*-
import pandas as pd
import csv
import numpy as np  
mydir = 'source'
#################################################
'''Reload class'''
################################################# 
def load_group(group_path):
    df = pd.read_csv(group_path)  
    class_word_list = df['relevant_words'].values.tolist()
    class_name_list = df['class_name'].values.tolist()
    return class_word_list, class_name_list


#################################################
'''dictionary: combine relevant_words and class_name'''
################################################# 
def build_best_dictionary(class_word_list, class_name_list, word2idx):
    class_name_word_list = []
    for idx in range(len(class_name_list)):
        tmp = []
        for token in class_name_list[idx].split('.'):
            if token in word2idx and token not in class_word_list:
                tmp.append(token)
        class_name_word_list.append(tmp)
        if class_word_list[idx] not in class_name_word_list[idx]:
            class_name_word_list[idx].append(class_word_list[idx])
    path = mydir + '/data/class_name_word_list.npy'
    np.save(path, class_name_word_list)
    return class_name_word_list


#################################################
'''cal p(w | d)'''
################################################# 
def cal_p_w_d(word_idx, doc_idx, topic_gen_word, doc_gen_topic):
    return np.dot(topic_gen_word[: ,word_idx], doc_gen_topic[doc_idx, :])

#################################################
'''get topic index'''
################################################# 
def documents_classify(doc_idx, word2idx, class_name_word_list, topic_gen_word, doc_gen_topic):
    topic_score_list = []
    for topic_word in class_name_word_list:
        total_topic_score = 0
        for word in topic_word:
            if word in word2idx:
                word_idx = word2idx[word]
                total_topic_score = total_topic_score + cal_p_w_d(word_idx, doc_idx, topic_gen_word, doc_gen_topic)
        topic_score = total_topic_score / len(topic_word)
        topic_score_list.append(topic_score)
    topic_index = np.argsort(np.array(topic_score_list))[-1]
    return topic_index

def documents_classify_using_groupcsv(doc_idx, word2idx, class_word_list, topic_gen_word, doc_gen_topic):
    topic_score_list = []
    for word in class_word_list:
        if word in word2idx:
            word_idx = word2idx[word]
            topic_score_list.append(cal_p_w_d(word_idx, doc_idx, topic_gen_word, doc_gen_topic))
    topic_index = np.argsort(np.array(topic_score_list))[-1]
    return topic_index

#################################################
'''write csv'''
################################################# 
def write_csv(output_path, ans):
    filename = output_path
    text = open(filename, "w")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["doc_id","class_id"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()    
    print("-----DONE-----")