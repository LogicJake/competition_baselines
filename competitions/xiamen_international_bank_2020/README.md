# 2020厦门国际银行数创金融杯建模大赛baseline分享

本 Baseline 来自 「酒心巧克力」的分享

成绩：0.34

比赛地址：https://www.dcjingsai.com/v2/cmptDetail.html?id=439&=76f6724e6fa9455a9b5ef44402c08653&ssoLoginpToken=&sso_global_session=e44c4d57-cd19-4ada-a1d3-a5250252bf86&sso_session=irjO90jPA0%205ytlVRkI1fA%3D%3D

## 赛题背景

在数字金融时代，大数据、人工智能技术在银行业内的发展日新月异，业内各机构都在加速数字化转型发展。厦门国际银行作为有特色的科技领先型中小银行，多年来始终坚持发挥数字金融科技力量，践行“数字赋能”理念，持续推进智慧风控、智慧营销、智慧运营、智慧管理，运用人工智能和大数据分析技术建立智能化客户服务模式和金融智慧营销服务体系，提升营销过程的智慧化、精准化水平，在为客户提供更贴心更具可用性的金融服务。

厦门国际银行联合厦门大学数据挖掘研究中心，为搭建一个行业交流平台，与社会各界精英共同探索机器学习和人工智能等热门技术问题，携手DataCastle数据城堡共同举办“2020第二届厦门国际银行“数创金融杯”建模大赛“。本届大赛以“金融+科技”为理念，着力于金融营销中的真实场景，总奖金达31万元。

## 任务

随着科技发展，银行陆续打造了线上线下、丰富多样的客户触点，来满足客户日常业务办理、渠道交易等客户需求。面对着大量的客户，银行需要更全面、准确地洞察客户需求。在实际业务开展过程中，需要发掘客户流失情况，对客户的资金变动情况预判；提前/及时针对客户进行营销，减少银行资金流失。本次竞赛提供实际业务场景中的客户行为和资产信息为建模对象，一方面希望能借此展现各参赛选手的数据挖掘实战能力，另一方面需要选手在复赛中结合建模的结果提出相应的营销解决方案，充分体现数据分析的价值。

## Label说明

label -1 下降

label 0 维稳

label 1 提升

## 官方说明

客户贡献度主要和客户的aum值有关

## 评价函数KAPPA

Kappa系数是一个用于一致性检验的指标，也可以用于衡量分类的效果。因为对于分类问题，所谓一致性就是模型预测结果和实际分类结果是否一致。kappa系数的计算是基于混淆矩阵的，取值为-1到1之间,通常大于0。

基于混淆矩阵的kappa系数计算公式如下：

![[公式]](https://www.zhihu.com/equation?tex=kappa+%3D+%5Cfrac%7Bp_o-p_e%7D%7B1-p_e%7D+)

其中：

![[公式]](https://www.zhihu.com/equation?tex=p_o+%3D+%5Cfrac+%7B%E5%AF%B9%E8%A7%92%E7%BA%BF%E5%85%83%E7%B4%A0%E4%B9%8B%E5%92%8C%7D%7B%E6%95%B4%E4%B8%AA%E7%9F%A9%E9%98%B5%E5%85%83%E7%B4%A0%E4%B9%8B%E5%92%8C%7D) ，**其实就是acc**。

![[公式]](https://www.zhihu.com/equation?tex=p_e+%3D+%5Cfrac%7B%5Csum_%7Bi%7D%7B%E7%AC%ACi%E8%A1%8C%E5%85%83%E7%B4%A0%E4%B9%8B%E5%92%8C+%2A+%E7%AC%ACi%E5%88%97%E5%85%83%E7%B4%A0%E4%B9%8B%E5%92%8C%7D%7D%7B%28%5Csum%7B%E7%9F%A9%E9%98%B5%E6%89%80%E6%9C%89%E5%85%83%E7%B4%A0%7D%29%5E2%7D+) ，即所有类别分别对应的“实际与预测数量的乘积”，之总和，除以“样本总数的平方”。

具体可以参考这个网址：

https://zhuanlan.zhihu.com/p/67844308

## 数据介绍

参考比赛网址，任务与数据

## 方案

在观察数据后发现，每个季度的最后一个月都有B6，B7，B8。而对于测试集，他的最后一个月的cust_no包含了绝大部分cust_no,要预测客户未来的情况走向。采用的是最后一个季度的情况进行预测，其中以下三个cust_no经过合并特征后丢失了，前两个月的合并后也一样丢失了：['0xb2d0afb2', '0xb2d2ed87', '0xb2d2d9d2']。这里先把他们的label都设为0.丢掉了含有NAN的列。

```python
     #没找到这三个cust_no。摸奖。
    low=pd.DataFrame()
    low['cust_no']=['0xb2d0afb2', '0xb2d2ed87', '0xb2d2d9d2']
    low['label']=[0,0,0]

```

### 基础特征

目前并没有做什么特征，只是简单的对I3，I8，I12进行了encoder。然后以I3进行分组训练（客户等级），其它的就是全梭哈。

```python
 le = LabelEncoder()
    train_B7['I3'] = le.fit_transform(train_B7['I3'].astype(str))
    test_B7['I3'] = le.transform(test_B7['I3'].astype(str))
    le = LabelEncoder()
    train_B7['I8'] = le.fit_transform(train_B7['I8'].astype(str))
    test_B7['I8'] = le.transform(test_B7['I8'].astype(str))
    le = LabelEncoder()
    train_B7['I12'] = le.fit_transform(train_B7['I12'].astype(str))
    test_B7['I12'] = le.transform(test_B7['I12'].astype(str))

    predictionsB4 = pd.DataFrame()


    predictionsB7 = pd.DataFrame()
    scoresB7 = list()

    for eve_id in tqdm(test_B7.I3.unique()):
        prediction,score= run_lgb_id(train_B7, test_B7, target='label', eve_id=eve_id)
        predictionsB7=predictionsB7.append(prediction)
        scoresB7.append(score)
```



### 采用的数据

对训练集，只采用了9月份，12月份的数据，测试集也采用的是3月份数据

```python
    # 1.读取文件：
    train_label_3=pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_label\y_Q3_3.csv')
    train_label_4 = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_label\y_Q4_3.csv')

    train_3 = pd.DataFrame()
    train_4 = pd.DataFrame()

    id3_data = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\cust_avli_Q3.csv')
    id4_data = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\cust_avli_Q4.csv')

    #合并有效客户的label
    train_label_3 = pd.merge(left=id3_data, right=train_label_3, how='inner', on='cust_no')
    train_label_4 = pd.merge(left=id4_data, right=train_label_4, how='inner', on='cust_no')
    #合并个人信息
    inf3_data = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\cust_info_q3.csv')
    inf4_data = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\cust_info_q4.csv')
    train_label_3 = pd.merge(left=inf3_data, right=train_label_3, how='inner', on='cust_no')
    train_label_4 = pd.merge(left=inf4_data, right=train_label_4, how='inner', on='cust_no')

    #第3季度信息提取
    for i in range(9,10):
        aum_3=pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\aum_m'+str(i)+'.csv')
        be_3 = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\behavior_m' + str(i) + '.csv')
        cun_3 = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\cunkuan_m' + str(i) + '.csv')
        fre_3=pd.merge(left=aum_3,right=be_3,how='inner', on='cust_no')
        fre_3=pd.merge(left=fre_3,right=cun_3,how='inner', on='cust_no')
        train_3=train_3.append(fre_3)

    train_fe3=pd.merge(left=fre_3,right=train_label_3,how='inner', on='cust_no')
    train_fe3.to_csv(r'E:\For_test2-10\data\厦门_data\train_feature\train3_fe_B7.csv',index=None)

    #第4季度信息提取
    for i in range(12,13):
        aum_4=pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\aum_m'+str(i)+'.csv')
        be_4 = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\behavior_m' + str(i) + '.csv')
        cun_4 = pd.read_csv(r'E:\For_test2-10\data\厦门_data\train_feature\cunkuan_m' + str(i) + '.csv')
        fre_4=pd.merge(left=aum_4,right=be_4,how='inner', on='cust_no')
        fre_4=pd.merge(left=fre_4,right=cun_4,how='inner', on='cust_no')
        train_3=train_3.append(fre_4)

    train_fe4=pd.merge(left=fre_4,right=train_label_4,how='inner', on='cust_no')
    train_fe4.to_csv(r'E:\For_test2-10\data\厦门_data\train_feature\train4_fe_B7.csv',index=None)

    train_B7=[train_fe3,train_fe4]
    train_B7=pd.concat(train_B7)

    test = pd.DataFrame()
    idtest_data = pd.read_csv(r'E:\For_test2-10\data\厦门_data\test_feature\cust_avli_Q1.csv')
    inftest_data = pd.read_csv(r'E:\For_test2-10\data\厦门_data\test_feature\cust_info_q1.csv')
    test_inf = pd.merge(left=inftest_data, right=idtest_data, how='inner', on='cust_no')
    # 第3季度信息提取
    for i in range(3, 4):
        aum = pd.read_csv(r'E:\For_test2-10\data\厦门_data\test_feature\aum_m' + str(i) + '.csv')
        be = pd.read_csv(r'E:\For_test2-10\data\厦门_data\test_feature\behavior_m' + str(i) + '.csv')
        cun = pd.read_csv(r'E:\For_test2-10\data\厦门_data\test_feature\cunkuan_m' + str(i) + '.csv')
        fre = pd.merge(left=aum, right=be, how='inner', on='cust_no')
        fre = pd.merge(left=fre, right=cun, how='inner', on='cust_no')
        test = test.append(fre)

    test_fe = pd.merge(left=test, right=test_inf, how='inner', on='cust_no')
    test_fe.to_csv(r'E:\For_test2-10\data\厦门_data\train_feature\test_fe_B7.csv', index=None)

    test_B7=test_fe.dropna(axis=1, how='any')
    train_B7=train_B7.dropna(axis=1, how='any')
```



## 模型

采用的是LGB模型，5折交叉验证

```python
def run_lgb_id(df_train, df_test, target, eve_id):
    feature_names = list(
        filter(lambda x: x not in ['label','cust_no'], df_train.columns))


    # 提取 eve_ID 对应的数据集
    df_train = df_train[df_train.I3 == eve_id]
    df_test = df_test[df_test.I3 == eve_id]

    model = lgb.LGBMRegressor(num_leaves=32,
                              max_depth=6,
                              learning_rate=0.08,
                              n_estimators=10000,
                              subsample=0.9,
                              feature_fraction=0.8,
                              reg_alpha=0.5,
                              reg_lambda=0.8,
                              random_state=2020)
    oof = []
    prediction = df_test[['cust_no']]
    prediction[target] = 0

    kfold = KFold(n_splits=5, random_state=2020)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric='mse',
                              early_stopping_rounds=20,
                             )

        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = df_train.iloc[val_idx][[target, 'cust_no']].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict(df_test[feature_names], num_iteration=lgb_model.best_iteration_)

        prediction[target] += pred_test / kfold.n_splits


        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    df_oof = pd.concat(oof)
    score = mean_squared_error(df_oof[target], df_oof['pred'])
    print('MSE:', score)

    return prediction,score
  
```

最后采用的是MSE作为线下评价指标

大佬们可以修改一下

可能合并特征的时候把cust_no merge掉了哈哈

线上：0.34左右

代码很大程度上借鉴了恒佬分享的baseline

第一次分享baseline 不喜勿喷

谢谢大家
