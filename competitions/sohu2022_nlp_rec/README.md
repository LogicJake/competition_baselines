# 2022搜狐校园 情感分析 × 推荐排序 算法大赛 baseline 分享

- 赛道链接：https://www.biendata.xyz/competition/sohu_2022/
- 本 baseline 预计可以获得 NLP: 0.655, REC: 0.550 左右的榜上分数

## 环境：

- transformers
- deepctr 后续应当用 deepctr-torch 版，这样就不用边折腾 torch 边折腾 TF 了 :(

## 复现步骤

1. 执行 NLP_training 训练 NLP 模型 (409M x 5 < 2G)
2. 执行 NLP_infer 推断生成 NLP 的提交文件，并同时生成 rec 所需要的情感特征
3. 执行 Rec_deepfm 生成 REC 的提交文件
4. 最后把两个文件放在 submission 目录并压缩成 submission.zip 进行提交

## 可能的改进

- NLP 参考其他 trick 进行提升
- 感觉实体情感特征并不能给 rec 很好的支撑，需要同学们继续摸索
