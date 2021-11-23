# 又小又快的爱奇艺用户留存预测pytorch版本baseline，线上84

借鉴官方baseline的标签和特征构造方式，只使用了两个特征，线下得分86，线上得分84。

### 赛道链接

http://challenge.ai.iqiyi.com/detail?raceId=61600f6cef1b65639cd5eaa6

### baseline

本 baseline 做了以下工作：

1. 只使用了user_id和32日launch type序列，因为构造样本的时候每个用户只构造了一条样本，所以user_id作用有限（单user_id线上79），上分主要贡献在32日launch type序列；
2. launch type相较于官方baseline做了一些改进，使用embedding序列，而非单数值序列，有一点提升；
3. 加入了早停；
4. 更多特征构造可以参考官方baseline（基于keras）和水哥的[paddle版本](https://aistudio.baidu.com/aistudio/projectdetail/2715522)。
