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

## posts topic top words example
* posts in Ko's Facebook fan page
| Topic index   | topic top words  |
| :------------ |:---------------:|
| Topic 01      | 公民, 參與, 顧問, 報名, 開放, 全民, 市民, 政府, 委員會, 顧問團, 台北市, 咖啡館, 預算, 希望, 公開, 廣場, 討論, 網路, 歡迎, 透明 |
| Topic 02      | 世大運, 比賽, 臺灣, 回家, 選手, 世界, 這次, 加油, 臺北, 一起, 倒數, 賽事, 台灣, 場館, 運動, 看見, 志工, 開幕, 國際, 主場 |
| Topic 03      | 台北, 專輯, 片花, 一分鐘, 限量, 募款, 獲贈, 專案, 即可, 一張, 線上, 捐款, 入場, 簽唱會, 音樂, 樂團, 乙張, 柯p, 香港, 天晴 | 
| Topic 04      | 城市, 臺北, 設計, 文化, 臺灣, 一個, 發展, 成為, 世界, 產業, 價值, 藝術, 台北, 光榮, 國際, 公園, 多元, 進步, 臺北市, 創意 |
| Topic 05      | 教育, 孩子, 老師, 學生, 學校, 實驗, 學習, 國小, 平等, 機會, 動物, 田園, 家長, 課程, 小朋友, 台北市, 高中, 應該, 教育局, 酷課 |
| Topic 06      | 防災, 颱風, 臺北市, 市民, 政府, 資訊, 演習, 災害, 朋友, 停止, 台北市, 安全, 民眾, 中央, 上班, 中心, 相關, 上課, 應變, 市府 |
| Topic 07      | 新政, 柯p, 捐款, 柯語錄, 支持, 柯文哲, 逐字, 智慧, 辯論, 創造, 一定, 打敗, 號碼, 交換, 速度, 遠大, 申論, 答案, 不一定, 團結 |
| Topic 08      | 台灣, 改變, 選舉, 政治, 柯文哲, 文化, 相信, 市民, 社會, 民主, 一個, 台北, 一場, 歷史, 自由, 首都, 擁抱, 朋友, 支持, 運動 |
| Topic 09      | 問題, 解決, 大巨蛋, 安全, 面對, 社子島, 一個, 市府, 政府, 公安, 遠雄, 處理, 食安, 北市府, 提出, 改善, 已經, 堅持, 第一步, 過去 |
| Topic 10      | 公車, 幹線, 捷運, 交通, 運輸, 路線, 路網, 吃到飽, 定期票, 公共, 綠色, 調整, 內湖, 八橫八縱, 改善, 搭乘, 大眾, 增加, 班次, 上路 |
| Topic 11      | 市場, 改建, 傳統, 環南, 東門, 美食, 攤商, 開工, 城市, 果菜, 生活, 縮影, 大龍, 老舊, 好壞, 廁所, 批發, 改建案, 職人, 工程 |
| Topic 12      | 燈節, 台北, 北門, 幸福, 時間, 旅遊網, 中華路, 中山堂, 展演, 西門町, 分鐘, 小奇, 西城, 遊行, 一起, 西區, 一場, 廣場, 地點, 一次 |
| Topic 13      | 老人, 服務, 社區, 照顧, 長者, 共餐, 居家, 照護, 福利, 健康, 據點, 社會, 長輩, 疫苗, 長照, 敬老卡, 台北市, 活得, 石頭湯, 重陽 |
| Topic 14      | 住宅, 公共, 政策, 公宅, 居住, 首爾, 租金, 智慧, 社區, 補貼, 房子, 政府, 台北市, 正義, 城市, 不賣, 申請, 青年, 推動, 萬戶 |
| Topic 16      | 計畫, 市政, 報告, 會議, 行動, 南港, 改變, 門戶, 東區, 直播, 簡報, 施政, 建設, 開會, 市府, 一起, 早安, 市議會, 區政, 文山區 |
| Topic 16      | 醫院, 醫療, 台大, 醫師, 病人, 聯合, 病房, 工作, 病患, 器官, 體系, 團隊, 市立, 外科, 醫學, 轉診, 醫生, 加護, 葉克膜, 捐贈 |

* posts in Yao's Facebook fan page
| Topic index   | topic top words  |
| :------------ |:---------------:|
| Topic 01      | 民調, 支持, 電話, 台北市長, 唯一, 民進黨, 姚文智, 階段, 好友, 提名, 市長, 爭取, 參選, 候選人, 第一, 政見, 市政, 顧立雄, 人選, 感謝 |
| Topic 02      | 機場, 松山, 桃園, 廢除, 松機, 跑道, 遷移, 桃機, 發展, 航廈, 離島, 捷運, 第三, 問題, 空域, 公頃, 台北, 起降, 居民, 完工 |
| Topic 03      | 城市, 台北, 論壇, 國際, 未來, 改造, 願景, 翻轉, 張基義, 邀請, 發展, 美學, 創新, 錄影, 矽谷, 開放, 月場, 邀請到, 需要, 地殼 | 
| Topic 04      | 都市, 條例, 再生, 住宅, 更新, 公辦, 推動, 政府, 老舊, 特別, 社會, 加速, 建築物, 公共, 修法, 改造, 重建, 萬戶, 提出, 發展 |
| Topic 05      | 台灣, 價值, 遊行, 首都, 一起, 民主, 國家, 史明, 朋友, 市長, 夢想, 站出來, 民進黨, 守護, 今天, 自由, 相信, 東京, 進行式, 一個 |
| Topic 06      | 蔡英文, 小英, 總部, 競選, 姚文智, 成立, 過半, 大會, 國會, 聯合, 承德路, 點亮, 下午, 最後, 地點, 邀請, 時間, 到場, 星期日, 一起 |
| Topic 07      | 公園, 中央, 活動, 創意, 台北, 公頃, 松機, 紐約, 想像, 報名, 官網, 河濱, 改變, 一座, 綠地, 未來, 城市, 截止, 這一, 擁有 |
| Topic 08      | 內湖, 交通, 創意, 報名, 問題, 競賽, 朋友, 活動, 狀況, 論壇, 右岸, 解決, 解開, 柯市長, 通勤, 公車, 智慧, 紓解, 提供, 更多 |
| Topic 09      | 故宮, 漫畫, 國寶, 鍾孟舜, 紀念堂, 中正, 計畫, 藝術, 鄭問, 轉型, 雙溪, 銅像, 鄭植羽, 正義, 院長, 展出, 漫畫家, 工會, 蔣介石, 反對 |
| Topic 10      | 姚文智, 台北, 市政, 鎖定, 翻轉, 一起, 民視, 分享, 下午, 政見, 辯論會, 加入, 城區, 市長, 西區, 藍圖, 廢除, 快樂, 公辦, 好友 |
| Topic 11      | 食安, 食品, 安全, 食農, 飲食, 教育法, 環境, 食物, 建構, 提出, 健康, 教育, 回收法, 三法, 姚文智, 出發, 管理法, 安心, 管理, 蔡英文 |
| Topic 12      | 文化, 古蹟, 菊元, 百貨, 資產, 俞大維, 保存, 文資, 建築, 歷史, 故居, 市定, 北市府, 保留, 審議, 大稻埕, 指定, 紀念, 文化局, 柯市府 |

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
