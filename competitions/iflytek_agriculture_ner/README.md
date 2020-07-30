# 赛道

讯飞开放平台 农业问答数据处理挑战赛

# 赛道链接

http://challenge.xfyun.cn/topic/info?type=agriculture

# 赛事概要

农业生产中，由于过程的主体是生物，存在多样性和变异性、个体与群体差异性，农业大数据中存在许多的专业名词，如农作物、病虫害、土壤修复、施肥方案、生理胁迫、种苗、疑难杂症、缺素、天气条件、地理信息等，尤其是非结构化的数据快速增长，如何挖掘数据价值、提高数据分析应用能力、减少数据冗余和数据垃圾，是农业大数据面临的重要问题。

数据处理的首要任务是标记命名实体，本次大赛提供了互联网平台上的专家与农民的问答数据作为训练样本，参赛选手需基于提供的样本构建模型，对问答数据进行标记切词。

# Baseline

因为本人也是第一次接触 NER 赛题，可能有很多不正确的地方，欢迎各路大佬指正。

### 标记方式

据我的了解，有好几种比较常见的标记方式，如 BIO, BIEO, BIEOS 等。

本 baseline 采用的是 BIEO，

- B 表示词语的开始(Begin)
- I 表示在词语的中间(Intermediate)
- E 表示词语的结尾 (End)
- O 表示其他情况 (Other)

在本次赛题中，实体有三种类别: 农作物(crop)，病害(disease) 和 药物(medicine)

所以我们标记的情况有 10 种情况

```
labels = [
    'B_crop',
    'I_crop',
    'E_crop',
    'B_disease',
    'I_disease',
    'E_disease',
    'B_medicine',
    'I_medicine',
    'E_medicine',
    'O'
]
```

举个例子来说如：炭疽病危害使用肟菌戊唑醇或苯甲丙环唑防治 这句话，标注后是：

```
炭 B_disease
疽 I_disease
病 E_disease
危 O
害 O
使 O
用 O
肟 B_medicine
菌 I_medicine
戊 I_medicine
唑 I_medicine
醇 E_medicine
或 O
苯 B_medicine
甲 I_medicine
丙 I_medicine
环 I_medicine
唑 E_medicine
防 O
治 O
```

但本赛题的训练集并不是以这种方式给出的，而是以词性分词的方式

```
炭疽病/n_disease 危害/v 使用/v 肟菌戊唑醇/n_medicine 或/c 苯甲丙环唑/n_medicine 防治/vn
```

所以这里要花费比较多的时间来处理成 BIEO 模式，可参见代码

### 模型

因为是初学者，所以我偷懒直接使用了 simpletransformers 这个包

simpletransformers 这个包高度封装了 huggingface 的 transformers

十分贴心地提供了各种 NLP 任务如文本分类、问答系统、NER 等下游任务的封装

简直就是调包侠的福音，只需要这么短:

```
model_args = NERArgs()
model_args.train_batch_size = 8
model_args.num_train_epochs = 5
model_args.fp16 = False
model_args.evaluate_during_training = True

model = NERModel("bert", 
                 "hfl/chinese-bert-wwm-ext",
                 labels=labels,
                 args=model_args)

model.train_model(train_data, eval_data=eval_data)
result, model_outputs, preds_list = model.eval_model(eval_data)
```

PS: 这里预训练模型使用的是哈工大讯飞联合实验室提供的 chinese-bert-wwm 

https://github.com/ymcui/Chinese-BERT-wwm

### 合并答案

因为 test 只提供了原始文本，这里也是需要将原始文本分割成一个个字送进去模型，

最后还需要将结果输出成要求的格式，也比较耗时间，可以参考给出的代码


# 线上分数

单折 0.94，距离前排有点远，但是这么少的代码能到这个分数也挺让我惊讶的


# TODO

- 优化预处理方式，baseline 代码里写得不是太合理；
- 改用 BIO 减少模型输出的类别
- 多个预训练模型融合，目前感觉并不是太好，三个模型融合后只有 0.946
- 抛弃 simpletransformers，尝试 BERT + BiLSTM + CRF 等
