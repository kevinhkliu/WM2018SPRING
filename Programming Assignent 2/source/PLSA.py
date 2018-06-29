# -*- coding: utf-8 -*-
import numpy as np  
from numpy import inf
import time
import pickle as pk
mydir = 'source' 
#################################################
'''initial  p(tk|dj), p(wi|tk) and p[i, j, k]'''
################################################# 
def initial_prob(num_doc, num_voc, K, C_wi_di):

    # doc_gen_topic[j, k] :  p(tk|dj) lamda 
    doc_gen_topic = np.random.uniform(0,1,[num_doc, K])
    for j in range(0, num_doc):
        normalization = np.sum(doc_gen_topic[j, :])
        for k in range(0, K):
            doc_gen_topic[j, k] /= normalization

    # topic_gen_word[k, i] : p(wi|tk) theta
    topic_gen_word = np.random.uniform(0,1,[K, num_voc])
    for k in range(0, K):
        normalization = np.sum(topic_gen_word[k, :])
        for i in range(0, num_voc):
            topic_gen_word[k, i] /= normalization
    
    #p[i, j, k] : p(tk|wi,dj)
    p = np.zeros([num_doc, num_voc, K])
    
    return doc_gen_topic, topic_gen_word, p

#################################################
'''E step:    p(wi|tk) * p(tk|dj) /  summation(p(wi|tk) * p(tk|dj))'''
################################################# 
def E_step(num_doc, num_voc, K, doc_gen_topic, topic_gen_word, p):
    for i in range(0, num_doc):
        for j in range(0, num_voc):
            p[i,j] = np.multiply(topic_gen_word[:,j].reshape(1,K),doc_gen_topic[i,:].reshape(1,K))
            summation = np.sum(p[i,j,:])
            if summation == 0:
                p[i, j, :] = 0;
            else:
                p[i, j, :] /= summation;
    return p


#################################################
'''M step:   
doc_gen_topicÔºöp(tk|dj) 
p(tk|dj) = summation(c(wi,dj) * p(tk|wi,dj)) / summation(c(wi,dj))

ttopic_gen_wordÔºöp(wi|tk)
p(tk|dj) = summation(c(wi,dj) * p(tk|wi,dj)) / summation(c(wi,dj)* p(tk|wi,dj))
'''
################################################# 
def M_step(num_doc, num_voc, K, doc_gen_topic, topic_gen_word, p, C_wi_di):
    # update doc_gen_topic p(tk|dj)
    for i in range(0, num_doc):
        for k in range(0, K):
            doc_gen_topic[i, k] = np.sum(np.multiply(C_wi_di[i,:].reshape(1,num_voc), p[i,:,k].reshape(1,num_voc)))
        summation = np.sum(C_wi_di[i, :])
        if summation == 0:
            doc_gen_topic[i, :] = 1.0 / K
        else:
            doc_gen_topic[i, :] /= summation
                    
    for k in range(0, K):
        for j in range(0, num_voc):
            topic_gen_word[k, j] = np.dot(C_wi_di[:,j], p[:,j,k])
        summation = np.sum(topic_gen_word[k, :])
        if summation == 0:
            for j in range(0, num_voc):
                topic_gen_word[k, j] = 1.0 / num_voc
        else:
            for j in range(0, num_voc):
                topic_gen_word[k, j] /= summation
    return doc_gen_topic, topic_gen_word


#################################################
'''Maximize the log-likelihood of the training  
L = summation(c(wi,dj)* p(wi|tk) * p(tk|dj))
'''
#################################################    
def LogLikelihood(doc_gen_topic, topic_gen_word, C_wi_di):
    tmp = np.dot(doc_gen_topic, topic_gen_word)
    np.seterr(divide='ignore')
    log_tmp = np.log(tmp)
    log_tmp[log_tmp == -inf] = 0
    loglikelihood_matrix = np.multiply(C_wi_di,log_tmp)
    #print('loglikelihood_matrix : ',loglikelihood_matrix.shape)
    loglikelihood = np.sum(loglikelihood_matrix)
    print('loglikelihood : ', loglikelihood)
    
    
#################################################
'''training: E -> M -> E -> M -> ....'''
################################################# 
def PLSA_training(num_doc, num_voc, K, total_iteration, doc_gen_topic, topic_gen_word, p, C_wi_di):
    LogLikelihood(doc_gen_topic, topic_gen_word, C_wi_di)
    for i in range(0, total_iteration):
        tStart = time.time() #Ë®àÊ??ãÂ?
        print("interation: " + str(i) + '/' + str(total_iteration))
        p = E_step(num_doc, num_voc, K, doc_gen_topic, topic_gen_word, p)
        doc_gen_topic, topic_gen_word = M_step(num_doc, num_voc, K, doc_gen_topic, topic_gen_word, p, C_wi_di)
        LogLikelihood(doc_gen_topic, topic_gen_word, C_wi_di)
        tEnd = time.time() #Ë®àÊ?ÁµêÊ?
        print("It cost %f sec" % (tEnd - tStart))
        print('-----------------------------------------')

    #################################################
    '''Save doc_gen_topic and topic_gen_word'''
    ################################################# 
    path = mydir + '/data/doc_gen_topic.pk'
    pk.dump(doc_gen_topic, open(path, 'wb'))
    path = mydir + '/data/topic_gen_word.pk'
    pk.dump(topic_gen_word, open(path, 'wb'))