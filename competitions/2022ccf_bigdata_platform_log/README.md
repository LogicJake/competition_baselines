# 2022 CCF 大数据平台安全事件检测与分类识别 无监督学习赛道 baseline 分享

赛道链接：https://www.datafountain.cn/competitions/595

注意下，这个比赛没有提供标签，是无监督学习，初赛是二分类，复赛是七分类，复赛依然是无监督学习。

第一次做非监督学习，分数不高，本 baseline 只有 0.549 左右，写文时能排在 21 名 (top50% 左右)，基本没啥技术含量，纯当抛砖引玉之用。

## baseline 思路

1. 做 count 特征；
2. 利用 message 简单做了点 TFIDF 特征；
3. 跑孤立森林模型，预测 outlier；
4. 调整下阈值。
