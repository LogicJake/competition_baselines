# CCF 2020 - 大数据时代的Serverless工作负载预测 Baseline 分享 (第一个版本)

本次比赛将以系列形式进行分享，本次为第一个可提交的版本，A 榜分数为 0.08459508000。  

EDIT1: 2020/10/21 开源第二个版本 A 榜分数为 0.20860830。

EDIT2: 2020/11/10 开源第三个版本 A 榜分数为 0.26515555。

### 背景：

云计算时代，Serverless软件架构可根据业务工作负载进行弹性资源调整，这种方式可以有效减少资源在空闲期的浪费以及在繁忙期的业务过载，同时给用户带来极致的性价比服务。在弹性资源调度的背后，对工作负载的预测是一个重要环节。如何快速感知业务的坡峰波谷，是一个实用的Serverless服务应该考虑的问题。

### 任务：

传统的资源控制系统以阈值为决策依据，只关注当前监控点的取值，缺少对历史数据以及工作负载趋势的把控，不能提前做好资源的调整，具有很长的滞后性。近年来，随着企业不断上云，云环境的工作负载预测成为一个经典且极具挑战的难题。

在此任务中，我们将提供给参赛者一系列真实场景下的性能监控数据，参赛者可针对训练数据做特征工程及建模，预测未来一段时间的工作负载情况。为了简化任务，本赛题挑选两个在生产环境较为重要的指标作为评测标准：CPU的利用率和队列中的Job数。

### BaseLine V1

本版本的 baseline 直接使用原始数据，未做特征工作，仅按照赛题要求理清建模和提交流程。

结构化时序问题，

先导入数据，按 qid 和时间进行排序

```
train = pd.read_csv('raw_data/train.csv')
train = train.sort_values(by=['QUEUE_ID', 'DOTTING_TIME']).reset_index(drop=True)

test = pd.read_csv('raw_data/evaluation_public.csv')
test = test.sort_values(by=['ID', 'DOTTING_TIME']).reset_index(drop=True)
```

这些 columns 在 test 只有单一值, 所以可以直接去掉

```
del train['STATUS']
del train['PLATFORM']
del train['RESOURCE_TYPE']

del test['STATUS']
del test['PLATFORM']
del test['RESOURCE_TYPE']
```

时间排序好后应该也没什么用了

```
del train['DOTTING_TIME']
del test['DOTTING_TIME']
```

Label Encoding

```
le = LabelEncoder()
train['QUEUE_TYPE'] = le.fit_transform(train['QUEUE_TYPE'].astype(str))
test['QUEUE_TYPE'] = le.transform(test['QUEUE_TYPE'].astype(str))
```

生成 target 列，

根据赛题要求：

对于每行测试数据，赛题会给定该队列在某时段的性能监控数据（比如9:35 – 10:00），希望参赛者可以预测该点之后的未来五个点的指标（10:00 – 10:25）

所以我们 shift(-5) 取值就可以了

```
df_train = pd.DataFrame()

for id_ in tqdm(train.QUEUE_ID.unique()):
    tmp = train[train.QUEUE_ID == id_]
    tmp['CPU_USAGE_next25mins'] = tmp['CPU_USAGE'].shift(-5)
    tmp['LAUNCHING_JOB_NUMS_next25mins'] = tmp['LAUNCHING_JOB_NUMS'].shift(-5)
    df_train = df_train.append(tmp)

df_train = df_train[df_train.CPU_USAGE_next25mins.notna()]
```

接下来是建模

因为我们有两个目标，采用 GroupKFold 五折 LGBM 回归，分别进行建模训练

代码较长，请直接参考源码

两个模型分别训练好后，进行合并以及后续处理和生成提交结果

```
prediction = prediction1.copy()
prediction = pd.merge(prediction, prediction2[['myid', 'LAUNCHING_JOB_NUMS_next25mins']], on='myid')

# 注意: 提交要求预测结果需为非负整数

prediction['CPU_USAGE_next25mins'] = prediction['CPU_USAGE_next25mins'].apply(np.floor)
prediction['CPU_USAGE_next25mins'] = prediction['CPU_USAGE_next25mins'].apply(lambda x: 0 if x<0 else x)
prediction['CPU_USAGE_next25mins'] = prediction['CPU_USAGE_next25mins'].astype(int)
prediction['LAUNCHING_JOB_NUMS_next25mins'] = prediction['LAUNCHING_JOB_NUMS_next25mins'].apply(np.floor)
prediction['LAUNCHING_JOB_NUMS_next25mins'] = prediction['LAUNCHING_JOB_NUMS_next25mins'].apply(lambda x: 0 if x<0 else x)
prediction['LAUNCHING_JOB_NUMS_next25mins'] = prediction['LAUNCHING_JOB_NUMS_next25mins'].astype(int)

preds = []

for id_ in tqdm(prediction.ID.unique()):
    items = [id_]
    tmp = prediction[prediction.ID == id_].sort_values(by='myid').reset_index(drop=True)
    for i, row in tmp.iterrows():
        items.append(row['CPU_USAGE_next25mins'])
        items.append(row['LAUNCHING_JOB_NUMS_next25mins'])
    preds.append(items)

sub = pd.DataFrame(preds)
sub.columns = sub_sample.columns

sub.to_csv('baseline.csv', index=False)
```

### BaseLine V2

PS: 我这种构造 label 的方式是有问题的，因为 DOTTING_TIME 并不是连续的，暂时还没有时间去做数据清洗

PS: V2 的改进是基于 qid 训练多个模型，即一个 qid 一个模型，因为所有 test 的 qid 都是在训练集中出现过的，所以可以这么搞，如果切榜后有新的 qid 就只能抓瞎了 :(  (QQ群里官方已答疑，切榜后的 qid 应该没有新的)

PS: 另外随便加了点 diff 特征，原始特征是 0.17 左右  
  
参考 baseline_v2.ipynb  

### BaseLine V3

与上个版本不同的地方主要在于：

根据官网里的这句话：“对于每行测试数据，赛题会给定该队列在某时段的性能监控数据（比如9: 35– 10:00），希望参赛者可以预测该点之后的未来五个点的指标（10:00 – 10:25）” 来重新构造 label：

使用 sliding window 滑窗 groupby qid 依次划取 10 个时间点的数据，前 5 个为特征，后 5 个为 label

```
t0 t1 t2 t3 t4 --> t5 t6 t7 t8 t9
t1 t2 t3 t4 t5 --> t6 t7 t8 t9 t10
``` 

另外，直接去掉了 job 的建模和预测，全置 0；增加了一点行内统计特征。
