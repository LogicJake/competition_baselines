import os

import numpy as np
import pandas as pd

np.random.seed(2021)

launch = pd.read_csv('raw_data/wsdm_train_data/app_launch_logs.csv')
test = pd.read_csv('raw_data/test-a.csv')

launch.date.min(), launch.date.max()

launch_grp = launch.groupby('user_id').agg(launch_date=('date', list),
                                           launch_type=('launch_type',
                                                        list)).reset_index()


def choose_end_date(launch_date):
    n1, n2 = min(launch_date), max(launch_date)
    if n1 < n2 - 7:
        end_date = np.random.randint(n1, n2 - 7)
    else:
        end_date = np.random.randint(100, 222 - 7)
    return end_date


def get_label(row):
    launch_list = row.launch_date
    end = row.end_date
    label = sum([1 for x in set(launch_list) if end < x < end + 8])
    return label


launch_grp['end_date'] = launch_grp.launch_date.apply(choose_end_date)
launch_grp['label'] = launch_grp.apply(get_label, axis=1)

train = launch_grp[['user_id', 'end_date', 'label']]
train

test['label'] = -1
test

data = pd.concat([train, test], ignore_index=True)
data

data = data.merge(launch_grp[['user_id', 'launch_type', 'launch_date']],
                  how='left',
                  on='user_id')
data


# get latest 32 days([end_date-31, end_date]) launch type sequence
# 0 for not launch, 1 for launch_type=0, and 2 for launch_type=1
def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date),
                      key=lambda x: x[1])
    seq_map = {d: t + 1 for t, d in seq_sort}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end - 31, end + 1)]
    return seq


data['launch_seq'] = data.apply(gen_launch_seq, axis=1)
data

data.head()

data.drop(columns=['launch_date', 'launch_type'], inplace=True)

os.makedirs('data', exist_ok=True)
data.to_pickle('data/all_data.pkl')
