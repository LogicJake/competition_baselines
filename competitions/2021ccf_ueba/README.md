# 发了好像又没发! 2021 CCF 基于UEBA的用户上网异常行为分析 baseline
- 赛道名: 基于UEBA的用户上网异常行为分析
- 赛道链接: https://www.datafountain.cn/competitions/520

## 赛题任务
利用机器学习、深度学习，UEBA等人工智能方法，基于无标签的用户日常上网日志数据，构建用户上网行为基线和上网行为评价模型，依据上网行为与基线的距离确定偏离程度。  
（1）通过用户日常上网数据构建行为基线；  
（2）采用无监督学习模型，基于用户上网行为特征，构建上网行为评价模型，评价上网行为与基线的偏离程度。  

## baseline
baseline 由天才儿童提供，详细代码见阅读原文。该baseline仅供参考，因为存在如下问题：
1. 训练集有label，榜单评价指标为 RMSE，但是主办方要求用无监督算法，所以榜单意义不大，主办方回应主要看方案的实际意义，所以可能变为方案赛
2. 测试集合划分有问题，leak比较严重

