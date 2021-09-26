import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm.sklearn import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time


train = pd.read_csv('data/train_data.csv', encoding='gbk')
test = pd.read_csv('data/A_test_data.csv', encoding='gbk')

df = pd.concat([train, test], axis=0, ignore_index=True)

df['timestamp'] = df['time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('timestamp').reset_index(drop=True)

df['day'] = df['time'].dt.day
df['dayofweek'] = df['time'].dt.dayofweek
df['hour'] = df['time'].dt.hour

df['IP_PORT'] = df['IP'] + ':' + df['port'].astype('str')
for f in tqdm(['account', 'group', 'IP', 'url', 'switchIP', 'IP_PORT']):
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique()))))
    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1 in tqdm(['account', 'group']):
    for f2 in ['IP', 'url', 'IP_PORT']:
        df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['id'].transform('count')
        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / df[f1 + '_count']
        df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')
        df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

for f in tqdm(['account', 'group']):
    df[f + '_next10_time_gap'] = df.groupby(f)['timestamp'].shift(-10) - df['timestamp']


X = df[~df['ret'].isna()].sort_values('id').reset_index(drop=True)
X_test = df[df['ret'].isna()].sort_values('id').reset_index(drop=True)
cols = [f for f in X.columns if f not in ['id', 'time', 'timestamp', 'ret']]


def eval_score(y_true, y_pred):
    return 'eval_score', 1 / (np.sin(np.arctan(np.sqrt(mean_squared_error(y_true, y_pred)))) + 1), True


X['score'] = 0
X_test['ret'] = 0
feat_imp_df = pd.DataFrame({'feats': cols, 'imp': 0})
skf = KFold(n_splits=5, shuffle=True, random_state=2021)
clf = LGBMRegressor(
    learning_rate=0.1,
    n_estimators=30000,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2021
)
for i, (trn_idx, val_idx) in enumerate(skf.split(X)):
    print('--------------------- {} fold ---------------------'.format(i))
    t = time.time()
    trn_x, trn_y = X[cols].iloc[trn_idx].reset_index(drop=True), X['ret'].values[trn_idx]
    val_x, val_y = X[cols].iloc[val_idx].reset_index(drop=True), X['ret'].values[val_idx]
    clf.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        eval_metric=eval_score,
        early_stopping_rounds=200,
        verbose=200
    )
    X.loc[val_idx, 'score'] = clf.predict(val_x)
    X_test['ret'] += clf.predict(X_test[cols]) / skf.n_splits
    feat_imp_df['imp'] += clf.feature_importances_
    print('runtime: {}\n'.format(time.time() - t))

cv_score = eval_score(X['ret'], X['score'])[1]
X_test[['id', 'ret']].to_csv('sub_{}.csv'.format(cv_score), index=False)


