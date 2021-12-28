# 山东赛 - 网格事件智能分类 baseline

赛道链接：http://data.sd.gov.cn/cmpt/cmptDetail.html?id=67

## 赛题介绍

基于网格事件数据，对网格中的事件内容进行提取分析，对事件的类别进行划分，具体为根据提供的事件描述，对事件所属政务类型进行划分。 

## baseline

这是个典型的文本多分类问题，之前总是出 simpletransformers 的 baseline，这次正规点出一个 transformers 的 baseline 啦~~

### 文本预处理

简单拼接文本，用 [SEP] 进行分割

```
def concat_text(row):
    return f'事件简述:{row["name"]}[SEP]'\
           f'事件内容:{row["content"]}'

train['text'] = train.apply(lambda row: concat_text(row), axis=1)
```

### 预训练模型

hfl/chinese-roberta-wwm-ext

### 训练

- 单折
- 只训练了 2 epochs，val acc 为 0.667 左右

### 线上分数

0.68018

### TODO

- 多折
- 融合
- 等等