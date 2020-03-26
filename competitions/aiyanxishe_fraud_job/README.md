## 比赛链接

https://god.yanxishe.com/46

## 比赛背景

通过互联网寻找职位信息已经成为当前重要求职手段，通过 AI 技术自动判断真假求职信息是一种有趣的应用场景。

## 数据分析

数据字段有：

- benefits: 文本类型，描述本岗位求职者可以收获到什么；
- company_profile: 文本类型，招聘公司的简介；
- department: 文本类型，招聘部门；
- description：文本类型，岗位描述；
- employment_type：文本类型，全职还是兼职；
- fraudulent: 标签，0=真，1=假；
- function: 文本类型，岗位名称；
- `has_company_logo`: 数值类型，招聘公司是否有 logo；
- industry: 文本类型，招聘公司的行业；
- location: 文本类型，招聘公司所在地；
- required_education: 文本类型，该岗位所需学历；
- required_experience: 文本类型，该岗位所需经验；
- requirements: 文本类型，该岗位所需技能；
- salary_range: 文本类型，该岗位薪资范围；
- telecommuting: 数值类型，招聘公司是否提供了联系电话；
- title: 文本类型，岗位名称。

可以看到大部分特征都是文本类型，其中的几个可以转成类别特征外，长文本我们可以用 TFIDF 来对文本进行降维转换操作。

## baseline

该赛题非常坑的一点是测试集只有 200 条 (训练集有 18000 条左右)，并且测试集和训练集的正负样本分布不同，正样本在训练集中不到 5%，而测试集据我提交的结果来看起码有 40% 以上，所以如果我们把 train 和 test 拼接起来做 TFIDF 或者热点编码，会导致结果非常糟糕，线上只有 50 分左右。

注意到这一点后，其他的都是常规套路。

单词个数特征

```
def process(x):
    if x == 'nan':
        return 0
    else:
        return len(x.split())


for col in ['benefits', 'title', 'company_profile', 'description', 'requirements']:
    train[f'{col}_wordsLen'] = train[col].astype('str').apply(lambda x: process(x))
    test[f'{col}_wordsLen'] = test[col].astype('str').apply(lambda x: process(x))
```

Label Encoding

```
df = pd.concat([train, test])
del train, test

for f in tqdm(['department', 'employment_type', 'function', 'industry',
               'location', 'required_education', 'required_experience', 'title']):
    lbl = LabelEncoder()
    df[f] = lbl.fit_transform(df[f].astype(str))

train = df[df['fraudulent'].notnull()].copy()
test = df[df['fraudulent'].isnull()].copy()

del df
gc.collect()
```

TFIDF，这里可以尝试不同的 max_features 效果

```
def get_tfidf(train, test, colname, max_features):

    text = list(train[colname].fillna('nan').values)
    tf = TfidfVectorizer(min_df=0, 
                         ngram_range=(1,2), 
                         stop_words='english', 
                         max_features=max_features)
    tf.fit(text)
    X = tf.transform(text)
    X_test = tf.transform(list(test[colname].fillna('nan').values))

    df_tfidf = pd.DataFrame(X.todense())
    df_tfidf_test = pd.DataFrame(X_test.todense())
    df_tfidf.columns = [f'{colname}_tfidf{i}' for i in range(max_features)]
    df_tfidf_test.columns = [f'{colname}_tfidf{i}' for i in range(max_features)]
    for col in df_tfidf.columns:
        train[col] = df_tfidf[col]
        test[col] = df_tfidf_test[col]
        
    return train, test


train, test = get_tfidf(train, test, 'benefits', 12)
train, test = get_tfidf(train, test, 'company_profile', 24)
train, test = get_tfidf(train, test, 'description', 48)
train, test = get_tfidf(train, test, 'requirements', 20)
```

采用 LGB 五折训练，线下 auc: 0.9766968325791855，线上 86 分，目前能排在前五名。
