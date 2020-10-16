import sys

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from simpletransformers.classification import ClassificationModel

fold = int(sys.argv[1])

train_query = pd.read_csv('raw_data/train/train.query.tsv', sep='\t', header=None)
train_query.columns = ['qid', 'text_a']
train_reply = pd.read_csv('raw_data/train/train.reply.tsv', sep='\t', header=None)
train_reply.columns = ['qid', 'rid', 'text_b', 'labels']
train = pd.merge(train_reply, train_query, on='qid', how='left')

df = train[['text_a', 'text_b', 'labels']]
df = df.sample(frac=1, random_state=1029)
train_df = df[df.index % 10 != fold]
eval_df = df[df.index % 10 == fold]
print(train_df.shape, eval_df.shape)

train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,
    'fp16': False
}
model = ClassificationModel('bert',
                            'hfl/chinese-roberta-wwm-ext',
                            num_labels=2,
                            use_cuda=True,
                            cuda_device=0,
                            args=train_args)
model.train_model(train_df, eval_df=eval_df)

test_query = pd.read_csv('raw_data/test/test.query.tsv', sep='\t', header=None, encoding="gbk")
test_query.columns = ['qid', 'text_a']
test_reply = pd.read_csv('raw_data/test/test.reply.tsv', sep='\t', header=None, encoding="gbk")
test_reply.columns = ['qid', 'rid', 'text_b']
test = pd.merge(test_reply, test_query, on='qid', how='left')
df_test = test[['text_a', 'text_b']]

submit_sample = pd.read_csv('raw_data/sample_submission.tsv', sep='\t', header=None)
submit_sample.columns =['qid', 'rid', 'label']

data = []
for i, row in df_test.iterrows():
    data.append([row['text_a'], row['text_b']])

predictions, raw_outputs = model.predict(data)
submit_sample['label'] = predictions

np.save(f'prob_{fold}', raw_outputs)
submit_sample.to_csv(f'sub_{fold}.tsv', sep='\t', index=False, header=False)


