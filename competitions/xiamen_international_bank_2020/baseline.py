import warnings
warnings.simplefilter('ignore')

import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb



def run_lgb_id(df_train, df_test, target, eve_id):
    feature_names = list(
        filter(lambda x: x not in ['label','cust_no'], df_train.columns))


    # 提取 QUEUE_ID 对应的数据集
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


if __name__ == "__main__":
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
    print(test_B7)
    print(train_B7)

    # Label Encoding
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
    print(np.mean(scoresB7))
    predictionsB7['label'] = predictionsB7['label'].apply(np.round)
    predictionsB7['label'] = predictionsB7['label'].apply(lambda x: -1 if x<-1 else x)
    predictionsB7['label'] = predictionsB7['label'].apply(lambda x: 1 if x>1 else x)
    predictionsB7['label'] = predictionsB7['label'].astype(int)


    #没找到这三个cust_no。摸奖。
    low=pd.DataFrame()
    low['cust_no']=['0xb2d0afb2', '0xb2d2ed87', '0xb2d2d9d2']
    low['label']=[0,0,0]

    print(low)
    predictionsB7=predictionsB7.append(low)

    prediction=[predictionsB4,predictionsB7]
    prediction=pd.concat(prediction)
    prediction.to_csv('sub_10_30.csv',index=None)

