## 比赛链接

https://god.yanxishe.com/44

## 比赛介绍

微博立场检测是判断微博作者对某个话题是持何种立场，立场有三种：FAVOR 支持，AGAINST 反对，NONE 两者都不是。

## EDA

数据集共有 3000 条数据，2400 条训练数据，600 条测试数据。

分为 target, text, stance 三个字段，其中 target 是对应于某个特定话题，text 是微博的正文内容，stance 是立场标签。

数据集中的话题共有五类，分布比较均匀。

- 深圳禁摩限电
- 开放二胎
- 俄罗斯在叙利亚的反恐行动
- IphoneSE
- 春节放鞭炮

立场标签分布上，FAVOR 和 AGAINST 基本持平，NONE 比较少

## Baseline

当前的 NLP 比赛，肯定少不了 BERT 的身影。所以本 Baseline 也是采用了 BERT 预训练模型 chinese-roberta-wwm-ext，简单的 五折交叉验证。

搭建模型主要使用了 keras_bert

```
def build_bert():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(3, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(1e-5),
                  metrics=['accuracy'])
    return model
```

交叉验证：

```
def run_cv(nfold, data, data_label, data_test):
    
    kf = KFold(n_splits=nfold, shuffle=True, random_state=1029).split(data)
    train_model_pred = np.zeros((len(data), 3))
    test_model_pred = np.zeros((len(data_test), 3))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        
        model = build_bert()
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('./' + str(i) + '.hdf5', monitor='val_acc', 
                                         verbose=2, save_best_only=True, mode='max', save_weights_only=True)
        
        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test, shuffle=False)
        
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )
        
        train_model_pred[test_fold, :] =  model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
        
        del model; gc.collect()
        K.clear_session()
        
    return train_model_pred, test_model_pred
```

## 效果

提交后分数为 68.6667，能在当前排 top 3，前两名分别为 69.1667 和 68.8333。
