import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from model import AQYModel
from model_tools import AQYDataset, fit, predict, validate

warnings.filterwarnings('ignore')


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(2021)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

data = pd.read_pickle('data/all_data.pkl')
data.head()

user_lbe = LabelEncoder()
data['user_id'] = user_lbe.fit_transform(data['user_id'])
data['user_id'] = data['user_id'] + 1

train = data[data['label'] != -1]
test = data[data['label'] == -1]

train = train.sample(frac=1, random_state=2021).reset_index(drop=True)

train_shape = int(train.shape[0] * 0.9)

valid = train.iloc[train_shape:]
train = train.iloc[:train_shape]

print(train.shape, valid.shape, test.shape)

train_dataset = AQYDataset(train, device)
valid_dataset = AQYDataset(valid, device)
test_dataset = AQYDataset(test, device)

train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)
valid_loader = DataLoader(valid_dataset,
                          batch_size=128,
                          shuffle=False,
                          num_workers=4)
test_loader = DataLoader(test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=4)

model = AQYModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_score = float('-inf')
last_improve = 0
best_model = None

for epoch in range(10):
    train_score = fit(model, train_loader, optimizer, criterion, device)
    val_score = validate(model, valid_loader, device)

    if val_score > best_val_score:
        best_val_score = val_score
        best_model = copy.deepcopy(model)
        last_improve = epoch
        improve = '*'
    else:
        improve = ''

    if epoch - last_improve > 3:
        break

    print(
        f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score} {improve}'
    )

model = best_model

valid['pred'] = predict(model, valid_loader, device)
valid['diff'] = valid['label'] - valid['pred']
valid['diff'] = abs(valid['diff']) / 7
score = 100 * (1 - valid['diff'].mean())
print(f'Valid Score: {score}')

os.makedirs('sub', exist_ok=True)

test['pred'] = predict(model, test_loader, device)
test = test[['user_id', 'pred']]
test['user_id'] = test['user_id'] - 1
test['user_id'] = user_lbe.inverse_transform(test['user_id'])

test.to_csv(f'sub/{score}.csv', index=False, header=False, float_format="%.2f")
