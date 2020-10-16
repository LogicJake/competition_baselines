# CCF 2020 - 房产行业聊天问答匹配

## 赛题介绍

赛题介绍请直接查看比赛链接 https://www.datafountain.cn/competitions/474

该题为经典的 Sentence Pair Classification

|  客户问题 | 经纪人回复 |  标签  |
|  -------  | ---------- | ------ |
| 您好，请问这个户型有什么优缺点呢？| 你是想看看这套房子是吗 |  0  |
|    | 在的 | 0 |
|    | 此房房型方正 得房率高 多层不带电梯4/6楼 | 1 |

## Baseline

- 10 折 RoBERTa，单折榜单分数为 0.749
- 直接执行 run.sh 即可，输出 baseline.tsv 可提交，榜单分数为 0.764，目前可以排在 top20
- 依赖：simpletransformers, 'hfl/chinese-roberta-wwm-ext' 预训练模型 (执行脚本会自动下载)
- 参数微调可修改 run_folds.py 里的 train_args
