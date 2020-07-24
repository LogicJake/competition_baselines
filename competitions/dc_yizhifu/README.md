线下 0.7154，线上 0.675 左右，完整代码见文末阅读全文。

## 比赛链接

https://www.dcjingsai.com/v2/cmptDetail.html?id=410

## 比赛背景
央行发布《金融科技FinTech》报告，强调金融科技成为推动金融转型升级的新引擎，成为促进普惠金融发展的新机遇。运用大数据、人工智能等技术建立金融风控模型，有效甄别高风险交易，智能感知异常交易，实现风险早识别、早预警、早处置，提升金融风险技防能力，是 “金融+科技”成果的显著体现。

翼支付积极研究探索“金融科技FinTech”技术并努力应用到实际业务中，挖掘更多金融科技在实际普惠金融业务的应用方案。本次竞赛将为校园新生力量提供才华施展的舞台和交流学习的通道，在实践中磨炼数据挖掘的专业能力，帮助学生完成从校园到社会的角色转变。

## 数据分析
赛方给出了三张数据表：基础信息表，交易信息表和操作信息表。数据字段较多，且匿名化程度很高，很多连续特征均被分桶成离散量。

## baseline
该 baseline 主要给大家建立起一个完整的特征工程，模型训练，sub 提交的流程。特征工程建立在基础信息表和交易信息表之上。基础信息表中有大量的类别特征，有些类别特征保留了大小关系，所以可以直接从变量中提取数字做处理：

```
for f in [
        'balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2',
        'balance2_avg', 'product1_amount', 'product2_amount',
        'product3_amount', 'product4_amount', 'product5_amount', 'product6_amount'
]:
    df_feature[f] = df_feature[f].apply(lambda x: int(x.split(' ')[1]) if type(x) != float else np.NaN)
```

效仿点击率特征，对欺诈问题还可以进行类似的欺诈率特征构造，即某某类别下欺诈的可能性为多大。由于该类特征设计到标签信息，要特别小心标签泄露问题，所以采用五折构造的方式，对训练集，每次使用其中4折数据做统计，给另外一折做特征。

```
# 欺诈率
def stat(df, df_merge, group_by, agg):
    group = df.groupby(group_by).agg(agg)

    columns = []
    for on, methods in agg.items():
        for method in methods:
            columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
    group.columns = columns
    group.reset_index(inplace=True)
    df_merge = df_merge.merge(group, on=group_by, how='left')

    del (group)
    gc.collect()
    return df_merge


def statis_feat(df_know, df_unknow):
    df_unknow = stat(df_know, df_unknow, ['province'], {'label': ['mean']})
    df_unknow = stat(df_know, df_unknow, ['city'], {'label': ['mean']})

    return df_unknow


df_train = df_feature[~df_feature['label'].isnull()]
df_train = df_train.reset_index(drop=True)
df_test = df_feature[df_feature['label'].isnull()]

df_stas_feat = None
kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, val_index in kf.split(df_train, df_train['label']):
    df_fold_train = df_train.iloc[train_index]
    df_fold_val = df_train.iloc[val_index]

    df_fold_val = statis_feat(df_fold_train, df_fold_val)
    df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

    del (df_fold_train)
    del (df_fold_val)
    gc.collect()

df_test = statis_feat(df_train, df_test)
df_feature = pd.concat([df_stas_feat, df_test], axis=0)
df_feature = df_feature.reset_index(drop=True)

del (df_stas_feat)
del (df_train)
del (df_test)
gc.collect()
```

从行为上更能识别欺诈行为，从交易表可以挖掘很多有用的特征。所以这里给出了交易金额的统计数据。

```
df_temp = df_trans.groupby(['user'
                            ])['amount'].agg(amount_mean='mean',
                                             amount_std='std',
                                             amount_sum='sum',
                                             amount_max='max',
                                             amount_min='min').reset_index()
df_feature = df_feature.merge(df_temp, how='left')
```

交易时间采用的是距离某起始时间点的时间间隔，例如： 9 days 09:02:45.000000000，表示距离某起始时间点9天9小时2分钟45秒。为了方便后面提取时间特征，可以自己设置一个起始日期转成正常日期。

```
def parse_time(tm):
    days, _, time = tm.split(' ')
    time = time.split('.')[0]

    time = '2020-1-1 ' + time
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time = (time + timedelta(days=int(days)))

    return time


df_trans['date'] = df_trans['tm_diff'].apply(parse_time)
df_trans['day'] = df_trans['date'].dt.day
df_trans['hour'] = df_trans['date'].dt.hour
```

## 进阶思路
* 对类别特征做 embedding
* 深度挖掘交易表和操作表
* 时间信息的利用
* 对于表中大量类别特征如何处理
