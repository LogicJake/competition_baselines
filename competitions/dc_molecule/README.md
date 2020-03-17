# dc_molecule_baseline
AI战疫·小分子成药属性预测大赛 baseline
   
## 主办方：DC竞赛
## 赛道：2020-AI战疫·小分子成药属性预测大赛

**赛道链接**：https://www.dcjingsai.com/common/cmpt/AI%E6%88%98%E7%96%AB%C2%B7%E5%B0%8F%E5%88%86%E5%AD%90%E6%88%90%E8%8D%AF%E5%B1%9E%E6%80%A7%E9%A2%84%E6%B5%8B%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html       
**赛程时间**：*2020.03.05-2020.04.06*      
## 1.数据说明  
**train.csv**

|       字段名       |  类型  |                        说明                         |
| :----------------: | :----: | :-------------------------------------------------: |
|         ID         |  整型  |                      样本编号                       |
| Molecule_max_phase |  整型  |                   分子的最长位相                    |
|  Molecular weight  | 浮点型 |                       分子量                        |
|   RO5_violations   |  整型  |             违反新药5规则（RO5）的数量              |
|       AlogP        | 浮点型 | 由ACD软件计算化合物的脂分配系数（该数据来自ChemBL） |
|      Features      |  向量  |                 小分子的矢量化表示                  |
|       Label        | 枚举/浮点型 |   单位时间内单位机体能将多少容积体液中的药物清除    |

**test.csv**
|       字段名       |    类型     |                        说明                         |
| :----------------: | :---------: | :-------------------------------------------------: |
|         ID         |    整型     |                      样本编号                       |
| Molecule_max_phase |    整型     |                   分子的最长位相                    |
|  Molecular weight  |   浮点型    |                       分子量                        |
|   RO5_violations   |    整型     |             违反新药5规则（RO5）的数量              |
|       AlogP        |   浮点型    | 由ACD软件计算化合物的脂分配系数（该数据来自ChemBL） |
|      Features      |    向量     |                 小分子的矢量化表示                  |



## 2.配置环境与依赖库 
  - python3
  - scikit-learn
  - numpy
  - lightgbm
## 3.运行代码步骤说明  
将数据集下载保存到 raw_data 文件夹, 新建 sub 文件夹保存提交文件。

## 4.特征工程   
 - Molecular weight 和 AlogP log1p
 - 小分子的矢量化表示拆分成具体单一特征(重要)     
 - 小分子的矢量化统计特征     
  
## 5.模型分数   
线上成绩2.096488
