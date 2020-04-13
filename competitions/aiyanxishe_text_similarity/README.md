AI 研习社 英文文本语义相似度 比赛 Baseline

## 比赛链接

https://god.yanxishe.com/53
 
## 比赛介绍

语义相似度是 NLP 的核心问题之一，对问答、翻译、检索等任务具有非常重要的意义。
 
该比赛给出两段短文本，要求判断文本的相似度评分(0-5)。

```
text_a: 'It depends on what you want to do next, and where you want to do it.'
text_b: 'It's up to you what you want to do next.'
score: 4.00
```

比赛的数据量不大，训练集 2300+ 对文本，测试集 500+ 对文本。

## 一个快速且有效的 baseline

这里我使用 simpletransformers 来构建 baseline。simpletransformers 是 transformers 的更高层封装，对于 NLP 的各种任务均提供了快速的实现方式 (通常只需要 build, train, predict 三行代码)

具体参考的例子如下：

https://github.com/ThilinaRajapakse/simpletransformers#regression

代码如下：

```
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from tqdm import tqdm

from simpletransformers.classification import ClassificationModel

# 读取数据
train = pd.read_csv('raw_data/train.csv')
test = pd.read_csv('raw_data/test.csv')
train.columns = ['text_a', 'text_b', 'labels']

# 配置训练参数
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,
    'regression': True,
}

# build model
model = ClassificationModel('roberta', 
                            'roberta-base', 
                            num_labels=1, 
                            use_cuda=True, 
                            cuda_device=0, 
                            args=train_args)
                            
# train model
model.train_model(train, eval_df=test)

# predict
preds = list()
for i, row in tqdm(test.iterrows()):
    text_a = row['text_a']
    text_b = row['text_b']
    pred, _ = model.predict([[text_a, text_b]])
    preds.append(pred)
    

sub = pd.DataFrame()
sub['ID'] = test.index
sub['score'] = [i.tolist() for i in preds]

# 后处理, 发现有超过 5 的情况
sub.loc[sub.score < 0.08, 'score'] = 0
sub.loc[sub.score > 5, 'score'] = 5

# submit
sub.to_csv('roberta_baseline.csv', index=False, header=False)
```

线上分数：87.8744，当前能排在前十。
