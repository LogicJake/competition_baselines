# 天池-零基础入门NLP-新闻文本分类

- pipeline 来自于 https://tianchi.aliyun.com/notebook-ai/detail?postId=118161 (作者 nano-)
- 模型采用了 transformer_encoder + LSTM (验证了下我们在 2020 腾讯广告算法大赛上用的 transformer 模型在普通 NLP 分类任务中的效果, 见 https://github.com/LogicJake/Tencent_Ads_Algo_2020_TOP12/blob/master/src/keras/f7_AGE_m6_transformer_lstm_2inputs_train_fold.py)
- 增加了 ReduceLROnPlateau callback
- 线上单折 0.944，五折估计可以 0.95
