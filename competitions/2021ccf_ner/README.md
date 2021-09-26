# Another CCF Baseline! 产品评论观点提取 0.646 baseline

赛题：产品评论观点提取
赛道链接: https://www.datafountain.cn/competitions/529

## 赛题介绍

官网介绍：「观点提取旨在从非结构化的评论文本中提取标准化、结构化的信息，如产品名、评论维度、评论观点等。此处希望大家能够通过自然语言处理的语义情感分析技术判断出一段银行产品评论文本的情感倾向，并能进一步通过语义分析和实体识别，标识出评论所讨论的产品名，评价指标和评价关键词。」

实体标注采用 BIO 格式，即 Begin, In, Out 格式

- B-BANK 代表银行实体的开始
- I-BANK 代表银行实体的内部
- B-PRODUCT 代表产品实体的开始
- I-PRODUCT 代表产品实体的内部
- O 代表不属于标注的范围
- B-COMMENTS_N 代表用户评论（名词）
- I-COMMENTS_N 代表用户评论（名词）实体的内部
- B-COMMENTS_ADJ 代表用户评论（形容词）
- I-COMMENTS_ADJ 代表用户评论（形容词）实体的内部

另外，赛题还需要选手对文本内容进行情感分类任务。

总结: 这是一个 NER + classification 比赛，线上评测指标为两者指标相加，其中 NER 为 strict-F1，classification 采用 Kappa

## baseline 思路

还是请出我们的老熟人 NLP baseline 小能手 simpletransformers

- https://simpletransformers.ai/docs/ner-model/
- https://simpletransformers.ai/docs/classification-models/

### NER

- 文本序列最大长度设置为 400
- 将数据集整理为 CoNLL 格式
- 训练集验证集大致划分为 9:1
- 预训练模型 hfl/chinese-bert-wwm-ext
- 训练 3 epochs，线下 f1 0.875

### Classification

- 文本序列最大长度设置为 400
- 三分类，数据集极度不平均，中立态度占了 96%
- 使用了全量数据
- 预训练模型 hfl/chinese-bert-wwm-ext
- 训练 3 epochs

## 提交结果

合并两项提交，线上得分 0.64668491406，提交时能排在第 17 位

具体细节见代码
