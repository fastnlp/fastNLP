# 支持批并行的LatticeLSTM
+ 原论文：https://arxiv.org/abs/1805.02023
+ 在batch=10时，计算速度已明显超过[原版代码](https://github.com/jiesutd/LatticeLSTM)。
+ 在main.py中添加三个embedding的文件路径以及对应数据集的路径即可运行

## 运行环境：
+ python >= 3.7.3
+ fastNLP >= dev.0.5.0
+ pytorch >= 1.1.0
+ numpy >= 1.16.4
+ fitlog >= 0.2.0
## 支持的数据集：
+ Resume，可以从[这里](https://github.com/jiesutd/LatticeLSTM)下载
+ Ontonote
+ [Weibo](https://github.com/hltcoe/golden-horse)

未包含的数据集可以通过提供增加类似 load_data.py 中 load_ontonotes4ner 这个输出格式的函数来增加对其的支持
## 性能：
|数据集| 目前达到的F1分数(test)|原文中的F1分数(test)|
|:----:|:----:|:----:|
|Weibo|62.73|58.79|
|Resume|95.18|94.46|
|Ontonote|73.62|73.88|

备注：Weibo数据集我用的是V2版本，也就是更新过的版本，根据杨杰博士Github上LatticeLSTM仓库里的某个issue，应该是一致的。

## 如有任何疑问请联系：
+ lixiaonan_xdu@outlook.com
