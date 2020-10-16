import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

p0 = np.load('prob_0.npy')
p1 = np.load('prob_1.npy')
p2 = np.load('prob_2.npy')
p3 = np.load('prob_3.npy')
p4 = np.load('prob_4.npy')
p5 = np.load('prob_5.npy')
p6 = np.load('prob_6.npy')
p7 = np.load('prob_7.npy')
p8 = np.load('prob_8.npy')
p9 = np.load('prob_9.npy')

p = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 10

submit_sample = pd.read_csv('raw_data/sample_submission.tsv', sep='\t', header=None)
submit_sample.columns =['qid', 'rid', 'label']

submit_sample['label'] = p.argmax(axis=1)

submit_sample.to_csv('baseline.tsv', sep='\t', index=False, header=False)
