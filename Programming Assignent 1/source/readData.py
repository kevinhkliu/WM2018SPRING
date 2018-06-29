# -*- coding: utf-8 -*-
import jieba
import logging
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
jieba.setLogLevel(logging.INFO)
from collections import defaultdict

#https://github.com/fxsjy/jiebademo/blob/master/jiebademo/jieba/dict.txt.big
jieba.load_userdict('dict.txt.big')

def cut(x):
    if not x:
        x = ' '
    return ' '.join(jieba.cut(x)) + ' '


def doc2dict_withoutCut(file):
    e = ET.parse(file).getroot()
    text = ''
    for x in e.find('doc/text').iter():
        text += x.text

    d = {
        'id': e.find('doc/id').text,
        'title': e.find('doc/title').text,
        'date': e.find('doc/date').text,
        'text': text.strip('\n')}

    return d

def query2dicts_withoutCut(file):
    e = ET.parse(file).getroot()

    d_list = []
    for x in e.findall('topic'):
        d_list.append({
            'number': x.find('number').text,
            'title': x.find('title').text.strip('\n'),
            'question': x.find('question').text.strip('\n'),
            'concepts': x.find('concepts').text.strip('\n'),
            'narrative': x.find('narrative').text.strip('\n')})

    return d_list

def make_corpus_without_cut(doc, query):
    doc_content = doc.title + doc.text
    query_content = query.concepts + query.title

    corpus = pd.concat((doc_content, query_content)).as_matrix()

    return corpus

def make_doc_corpus(doc):
    docCorpus = doc.title.apply(lambda x: x + ' ') + doc.text
    #docCorpus = doc.title + doc.text
    return docCorpus

def make_query_corpus(query):
    queryCorpus = query.concepts.apply(lambda x: x + ' ') + query.title
    #queryCorpus = query.concepts + query.title
    return queryCorpus

def doc2dict(file):
    e = ET.parse(file).getroot()
    text = ''
    for x in e.find('doc/text').iter():
        text += x.text

    d = {
        'id': e.find('doc/id').text,
        'title': cut(e.find('doc/title').text),
        'date': e.find('doc/date').text,
        'text': cut(text).strip('\n')}

    return d

def query2dicts(file):
    e = ET.parse(file).getroot()

    d_list = []
    for x in e.findall('topic'):
        d_list.append({
            'number': x.find('number').text,
            'title': cut(x.find('title').text.strip('\n')),
            'question': cut(x.find('question').text.strip('\n')),
            'concepts': cut(x.find('concepts').text.strip('\n')),
            'narrative': cut(x.find('narrative').text.strip('\n'))})

    return d_list

def make_corpus(doc, query):
    doc_content = doc.title.apply(lambda x: x + ' ') + doc.text
    query_content = query.concepts.apply(lambda x: x + ' ') + query.title

    corpus = pd.concat((doc_content, query_content)).as_matrix()

    return corpus


def read_file_list(file):
    with open(file) as f:
        data = f.readlines()
        docID = 0
        doc_file_dict = {}
        for line in data:
            docFileName = line.strip('\n')
            docName = line.strip('\n').split('/')[3]
            doc_file_dict[docID] = {'docName': docName, 'docFileName': docFileName}
            docID = docID + 1
            
    with open('dict/doc_file_dict.pkl', 'wb') as f:
        pickle.dump(doc_file_dict, f)    
        
def read_vocab_list(file):
    vocabulary = defaultdict()
    with open(file, encoding='utf-8') as f:
        data = f.readlines()
        vocID = 0
        vocabulary = {}
        for line in data:
            vocab = line.strip('\n')
            vocabulary[vocab] = vocID
            vocID = vocID + 1
            
    with open('dict/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)  
   
'''     
def read_vocab_list(file):
    with open(file, encoding='utf-8') as f:
        data = f.readlines()
        vocID = 0
        vocDict = {}
        for line in data:
            vocab = line.strip('\n')
            vocDict[vocID] = vocab
            vocID = vocID + 1
    with open('dict/vocDict.pkl', 'wb') as f:
        pickle.dump(vocDict, f)  
'''        
        
'''       
def read_inverted_file(file):
    with open('dict/vocDict.pkl', 'rb') as f:
        vocDict = pickle.load(f)
    
    with open(file) as f:
        lines = f.readlines()
        invertedDict = {}
        lineIdx = 0
        print(len(lines))
        while lineIdx < len(lines):
            lineList = lines[lineIdx].strip('\n').split(' ')
            print(str(lineIdx) + '/' + str(len(lines)))
            vocID = int(lineList[0])
            vocID2 = int(lineList[1])
            docFreq = int(lineList[2]) #出現在幾個文章
            postingList = []
            postingCountList = []
            collFreq = 0
            
            for idx in range(docFreq):
                lineIdx = lineIdx + 1
                posting = lines[lineIdx].strip('\n').split(' ')
                docID = posting[0]      
                collFreq = collFreq + int(posting[1])  #voc在corpus中出現次數
                postingList.append(docID)
                postingCountList.append(posting[1])
                idx = idx + 1
            lineIdx = lineIdx + 1
            if vocID2 == -1: #voc is unigram
                invertedDict[str(vocDict[vocID])] = {'docFreq': docFreq, 'collFreq': collFreq, 'postingList': postingList, 'postingCountList': postingCountList}
            else:  # voc is bigram
                vocab = str(vocDict[vocID]) + str(vocDict[vocID2])
                invertedDict[vocab] = {'docFreq': docFreq, 'collFreq': collFreq, 'postingList': postingList, 'postingCountList': postingCountList}

    with open('dict/invertedDict.pkl', 'wb') as f:
        pickle.dump(invertedDict, f)  
'''            


