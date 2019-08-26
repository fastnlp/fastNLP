# 指代消解复现
## 介绍
Coreference resolution是查找文本中指向同一现实实体的所有表达式的任务。
对于涉及自然语言理解的许多更高级别的NLP任务来说，
这是一个重要的步骤，例如文档摘要，问题回答和信息提取。
代码的实现主要基于[ End-to-End Coreference Resolution (Lee et al, 2017)](https://arxiv.org/pdf/1707.07045).


## 数据获取与预处理
论文在[OntoNote5.0](https://allennlp.org/models)数据集上取得了当时的sota结果。
由于版权问题，本文无法提供数据集的下载，请自行下载。
原始数据集的格式为conll格式，详细介绍参考数据集给出的官方介绍页面。

代码实现采用了论文作者Lee的预处理方法，具体细节参见[链接](https://github.com/kentonl/e2e-coref/blob/e2e/setup_training.sh)。
处理之后的数据集为json格式，例子：
```
{
  "clusters": [],
  "doc_key": "nw",
  "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
  "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
}
```

### embedding 数据集下载
[turian emdedding](https://lil.cs.washington.edu/coref/turian.50d.txt)

[glove embedding](https://nlp.stanford.edu/data/glove.840B.300d.zip)



## 运行
```shell
# 训练代码
CUDA_VISIBLE_DEVICES=0 python train.py
# 测试代码
CUDA_VISIBLE_DEVICES=0 python valid.py
```

## 结果
原论文作者在测试集上取得了67.2%的结果，AllenNLP复现的结果为 [63.0%](https://allennlp.org/models)。
其中AllenNLP训练时没有加入speaker信息，没有variational dropout以及只使用了100的antecedents而不是250。

在与AllenNLP使用同样的超参和配置时，本代码复现取得了63.6%的F1值。


## 问题
如果您有什么问题或者反馈，请提issue或者邮件联系我：
yexu_i@qq.com
