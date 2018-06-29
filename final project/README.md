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

## Posts topic top words example
* Ko's Facebook fan page

| Topic index   | topic top words  |
| :------------: | :--------------- |
| Topic 01      | 公民, 參與, 顧問, 報名, 開放, 全民, 市民, 政府 |
| Topic 02      | 世大運, 比賽, 臺灣, 回家, 選手, 世界, 這次, 加油 |
| Topic 03      | 台北, 專輯, 片花, 一分鐘, 限量, 募款, 獲贈, 專案 | 
| Topic 04      | 城市, 臺北, 設計, 文化, 臺灣, 一個, 發展, 成為 |
| Topic 05      | 教育, 孩子, 老師, 學生, 學校, 實驗, 學習, 國小 |
| Topic 06      | 防災, 颱風, 臺北市, 市民, 政府, 資訊, 演習, 災害 |
| Topic 07      | 新政, 柯p, 捐款, 柯語錄, 支持, 柯文哲, 逐字, 智慧 |
| Topic 08      | 台灣, 改變, 選舉, 政治, 柯文哲, 文化, 相信, 市民 |
| Topic 09      | 問題, 解決, 大巨蛋, 安全, 面對, 社子島, 一個, 市府 |
| Topic 10      | 公車, 幹線, 捷運, 交通, 運輸, 路線, 路網, 吃到飽 |
| Topic 11      | 市場, 改建, 傳統, 環南, 東門, 美食, 攤商, 開工 |
| Topic 12      | 燈節, 台北, 北門, 幸福, 時間, 旅遊網, 中華路, 中山堂 |
| Topic 13      | 老人, 服務, 社區, 照顧, 長者, 共餐, 居家, 照護 |
| Topic 14      | 住宅, 公共, 政策, 公宅, 居住, 首爾, 租金, 智慧 |
| Topic 16      | 計畫, 市政, 報告, 會議, 行動, 南港, 改變, 門戶 |
| Topic 16      | 醫院, 醫療, 台大, 醫師, 病人, 聯合, 病房, 工作 |

* Yao's Facebook fan page

| Topic index   | topic top words  |
| :------------: | :--------------- |
| Topic 01      | 民調, 支持, 電話, 台北市長, 唯一, 民進黨, 姚文智, 階段 |
| Topic 02      | 機場, 松山, 桃園, 廢除, 松機, 跑道, 遷移, 桃機 |
| Topic 03      | 城市, 台北, 論壇, 國際, 未來, 改造, 願景, 翻轉 | 
| Topic 04      | 都市, 條例, 再生, 住宅, 更新, 公辦, 推動, 政府 |
| Topic 05      | 台灣, 價值, 遊行, 首都, 一起, 民主, 國家, 史明 |
| Topic 06      | 蔡英文, 小英, 總部, 競選, 姚文智, 成立, 過半, 大會 |
| Topic 07      | 公園, 中央, 活動, 創意, 台北, 公頃, 松機, 紐約 |
| Topic 08      | 內湖, 交通, 創意, 報名, 問題, 競賽, 朋友, 活動 |
| Topic 09      | 故宮, 漫畫, 國寶, 鍾孟舜, 紀念堂, 中正, 計畫, 藝術 |
| Topic 10      | 姚文智, 台北, 市政, 鎖定, 翻轉, 一起, 民視, 分享 |
| Topic 11      | 食安, 食品, 安全, 食農, 飲食, 教育法, 環境, 食物 |
| Topic 12      | 文化, 古蹟, 菊元, 百貨, 資產, 俞大維, 保存, 文資 |

## Demo website  
1. [WRM2018 final](https://boiling-ravine-49392.herokuapp.com/post.html)

## Reference
0. [DoctorKoWJ Facebook fan page](https://www.facebook.com/DoctorKoWJ/)
0. [Yao Facebook fan page](https://www.facebook.com/YaoTurningTaipei/)
1. [Facebook crawler code link](https://github.com/b02902131/FBcrawler)
2. [TextRank4ZH](https://github.com/letiantian/TextRank4ZH)
3. [如何用Python从海量文本抽取主题](https://www.jianshu.com/p/fdde9fc03f94)
4. [topic-model-tutorial](https://github.com/derekgreene/topic-model-tutorial)
5. [LDAvis](https://github.com/cpsievert/LDAvis)
6. [ANTUSD: A Large Chinese Sentiment Dictionary](http://www.lrec-conf.org/proceedings/lrec2016/pdf/450_Paper.pdf)
