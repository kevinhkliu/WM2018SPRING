# Mayor Ko's Trial Balloon

## Team member
Da-Yo Tseng, Hung-Kuo Liu, Chin-Hua Hu and Ying-Ting Lin

## Introduction
Trial balloon, is the most direct way to observe the reaction of an audience. We can easily know the most popular topic that community really care about. In addition, if we have different times of the data on social network, it can be used for visualization that show the evolution of events/topics.  
Nowadays, there are lots of Facebook users post their comments for some topics or events on Mayor’s fan page, but we don’t know what topics or events that users really care about at the different times. So we want to provide a system that can collect public opinion on social network.  
We are going to collect posts and comments from Mayor Ko's Facebook fan page. Then, we will use this dataset to analyze public opinion trend. Furthermore, we wants to visualize the results by different kinds of graph.

## Data
By using Facebook graph api, we crawl almost 1500 posts and 74705 comments in Ko's Facebook fan page, 977 posts and 15497 comments in Yao's Facebook fan page.

## Dependency
`Python3` `numpy` `copy` `pickle` `re` `gensim` `itertools` `scikit-learn` `matplotlib` `matplotlib` `pyLDAvis`

## Requirement
* Download facebook pre-trained word vector [[link]](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.zip).
* Download jieba package [[link]](https://github.com/fxsjy/jieba).
* Download ANTUSD dictionary [[link]](https://docs.google.com/forms/d/e/1FAIpQLSe2Bx1CYqLajfthIL8Q_32HXHqWrxkJMc6f9AnsVuxTD4BdGg/viewform?c=0&w=1).
```
pip install textrank4zh --user
```
## Demo website  
1. [WRM2018 final](https://boiling-ravine-49392.herokuapp.com/post.html)

## Reference
1. [Facebook crawler code link](https://github.com/b02902131/FBcrawler)
2. [TextRank4ZH](https://github.com/letiantian/TextRank4ZH)
3. [如何用Python从海量文本抽取主题](https://www.jianshu.com/p/fdde9fc03f94)
4. [topic-model-tutorial](https://github.com/derekgreene/topic-model-tutorial)
5. [LDAvis](https://github.com/cpsievert/LDAvis)
6. [ANTUSD: A Large Chinese Sentiment Dictionary](http://www.lrec-conf.org/proceedings/lrec2016/pdf/450_Paper.pdf)
