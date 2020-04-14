#coding: utf-8

__author__ = "zhengheng"

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
text_list = list()
for i, row in tqdm(test.iterrows()):
    text_list.append([row['text_a'], row['text_b']])
    
pred, _ = model.predict(text_list)

sub = pd.DataFrame()
sub['ID'] = test.index
sub['score'] = pred

# 后处理, 发现有超过 5 的情况
sub.loc[sub.score < 0.08, 'score'] = 0
sub.loc[sub.score > 5, 'score'] = 5

# submit
sub.to_csv('roberta_baseline.csv', index=False, header=False)
