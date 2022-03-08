# 第三届阿里云磐久智维算法大赛初赛 baseline 分享

- 赛道：第三届阿里云磐久智维算法大赛（天池平台）
- 赛道链接：https://tianchi.aliyun.com/competition/entrance/531947/introduction




## 问题描述

给定一段时间的系统日志数据，参赛者应提出自己的解决方案，以诊断服务器发生了哪种故障。具体来说，参赛者需要从组委会提供的数据中挖掘出和各类故障相关的特征，并采用合适的机器学习算法予以训练，最终得到可以区分故障类型的最优模型。数据处理方法和算法不限，但选手应该综合考虑算法的效果和复杂度，以构建相对高效的解决方案。

## 数据描述

- SEL 日志数据，提供了每一个 sn 必要时段的 log 日志
- 训练标签数据，提供了一个 sn 在 fault_time 时刻出现的故障 (label，四分类，其中 0 类和 1 类表示CPU相关故障，2 类表示内存相关故障，3 类表示其他类型故障)
- 提交样例，给定 sn、fault_time 两个字段信息，选手需要根据 SEL 日志信息给出最终的 label。

## 评测函数

多分类加权 Macro F1-score (类别 0 权重最大)

```
def  macro_f1(target_df: pd.DataFrame,  submit_df: pd.DataFrame)  -> float:

    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """

    weights =  [3/7,  2/7,  1/7,  1/7]

    overall_df = target_df.merge(submit_df, how='left', on=['sn', 'fault_time'], suffixes=['_gt', '_pr'])
    overall_df.fillna(-1)

    macro_F1 =  0.
    for i in  range(len(weights)):
        TP =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] == i)])
        FP =  len(overall_df[(overall_df['label_gt'] != i) & (overall_df['label_pr'] == i)])
        FN =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
    return macro_F1
```

## baseline (线下 0.59 线上 0.65)

本赛题有明显的文本特征，也有时间序列特征，可以考虑结合两者做特征，也可以尝试直接拼接日志当成文本多分类。

本 baseline 简单做了 tfidf, w2v, nunique, 时间特征等，具体代码点击阅读原文(或者访问我们的 github 仓库)

```
def make_dataset(dataset, data_type='train'):
    ret = list()

    for idx, row in tqdm(dataset.iterrows()):
        sn = row['sn']
        fault_time = row['fault_time']
        ts = row['fault_time_ts']

        if data_type == 'train':
            label = row['label']

        df = sel_data[sel_data['sn'] == sn].copy()

        df = df[df['time_ts'] <= ts].copy()
        df = df.sort_values(by='time_ts').reset_index(drop=True)
        df = df.tail(40).copy()        # TODO: could change last 40 logs here

        # make some features

        logs_count = len(df)

        if logs_count > 0:
            msg_nunique = df['msg'].nunique()
            msg_category_nunique = df['category'].nunique()
            msg_split_0_nunique = df['msg_split_0'].nunique()
            msg_split_1_nunique = df['msg_split_1'].nunique()
            msg_split_2_nunique = df['msg_split_2'].nunique()
            last_category = df['category'].value_counts().index[0]
            last_category = cate_map[last_category] if last_category in cate_map else len(cate_map)

            s = df['time_ts'].values
            if len(s) > 0:
                seconds_span = s[-1] - s[0]
            else:
                seconds_span = 0

            df['time_ts_shift_1'] = df['time_ts'].shift(1)
            df['time_ts_diffs_1'] = df['time_ts'] - df['time_ts_shift_1']
            s = df['time_ts_diffs_1'].values
            if len(s) > 1:
                log_time_diffs_avg = np.mean(s[1:])
                log_time_diffs_max = np.max(s[1:])
                log_time_diffs_min = np.min(s[1:])
                log_time_diffs_std = np.std(s[1:])
            else:
                try:
                    log_time_diffs_avg = log_time_diffs_max = log_time_diffs_min = s[0]
                    log_time_diffs_std = 0
                except:
                    log_time_diffs_avg = log_time_diffs_max = log_time_diffs_min = log_time_diffs_std = 0

            all_msg = "\n".join(df['msg'].values.tolist()).lower()
            w2v_emb = get_w2v_mean(all_msg)[0]
            tfv_emb = get_tfidf_svd([s.lower() for s in df['msg'].values.tolist()])

        else:
            logs_count = 0
            msg_nunique = 0
            msg_category_nunique = 0
            msg_split_0_nunique = 0
            msg_split_1_nunique = 0
            msg_split_2_nunique = 0
            last_category = 0
            seconds_span = 0
            log_time_diffs_avg = 0
            log_time_diffs_max = 0
            log_time_diffs_min = 0
            log_time_diffs_std = 0
            w2v_emb = [0] * 32
            tfv_emb = [0] * 16


        # format dataset
        data = {
            'sn': sn,
            'fault_time': fault_time,
            'logs_count': logs_count,
            'msg_nunique': msg_nunique,
            'msg_category_nunique': msg_category_nunique,
            'msg_split_0_nunique': msg_split_0_nunique,
            'msg_split_1_nunique': msg_split_1_nunique,
            'msg_split_2_nunique': msg_split_2_nunique,
            'last_category': last_category,
            'seconds_span': seconds_span,
            'log_time_diffs_avg': log_time_diffs_avg,
            'log_time_diffs_max': log_time_diffs_max,
            'log_time_diffs_min': log_time_diffs_min,
            'log_time_diffs_std': log_time_diffs_std,
        }

        for i in range(32):
            data[f'msg_w2v_{i}'] = w2v_emb[i]
        for i in range(16):
            data[f'msg_tfv_{i}'] = tfv_emb[i]

        if data_type == 'train':
            data['label'] = label

        ret.append(data)

    return ret
```

## TODO:

1. 构造更多的文本特征
2. 构造更多的统计特征
3. 使用 BERT 等更先进的模型提取文本特征
4. 其他我不知道的构建数据集方式
5. 文本清洗等
6. 融合或者 stacking
