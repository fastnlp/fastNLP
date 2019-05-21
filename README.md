# fastNLP

[![Build Status](https://travis-ci.org/fastnlp/fastNLP.svg?branch=master)](https://travis-ci.org/fastnlp/fastNLP)
[![codecov](https://codecov.io/gh/fastnlp/fastNLP/branch/master/graph/badge.svg)](https://codecov.io/gh/fastnlp/fastNLP)
[![PyPI version](https://badge.fury.io/py/fastNLP.svg)](https://badge.fury.io/py/fastNLP)
![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)
[![Documentation Status](https://readthedocs.org/projects/fastnlp/badge/?version=latest)](http://fastnlp.readthedocs.io/?badge=latest)

fastNLP 是一款轻量级的 NLP 处理套件。你既可以使用它快速地完成一个命名实体识别（NER）、中文分词或文本分类任务； 也可以使用他构建许多复杂的网络模型，进行科研。它具有如下的特性：

- 统一的Tabular式数据容器，让数据预处理过程简洁明了。内置多种数据集的DataSet Loader，省去预处理代码。
- 各种方便的NLP工具，例如预处理embedding加载; 中间数据cache等;
- 详尽的中文文档以供查阅；
- 提供诸多高级模块，例如Variational LSTM, Transformer, CRF等;
- 封装CNNText，Biaffine等模型可供直接使用;
- 便捷且具有扩展性的训练器; 提供多种内置callback函数，方便实验记录、异常捕获等。


## 安装指南

fastNLP 依赖如下包:

+ numpy
+ torch>=0.4.0
+ tqdm
+ nltk

其中torch的安装可能与操作系统及 CUDA 的版本相关，请参见 PyTorch 官网 。 
在依赖包安装完成的情况，您可以在命令行执行如下指令完成安装

```shell
pip install fastNLP
```


## 内置组件

大部分用于的 NLP 任务神经网络都可以看做由编码（encoder）、聚合（aggregator）、解码（decoder）三种模块组成。


![](./docs/source/figures/text_classification.png)

fastNLP 在 modules 模块中内置了三种模块的诸多组件，可以帮助用户快速搭建自己所需的网络。 三种模块的功能和常见组件如下:

<table>
<tr>
    <td><b> 类型 </b></td>
    <td><b> 功能 </b></td>
    <td><b> 例子 </b></td>
</tr>
<tr>
    <td> encoder </td>
    <td> 将输入编码为具有具 有表示能力的向量 </td>
    <td> embedding, RNN, CNN, transformer
</tr>
<tr>
    <td> aggregator </td>
    <td> 从多个向量中聚合信息 </td>
    <td> self-attention, max-pooling </td>
</tr>
<tr>
    <td> decoder </td>
    <td> 将具有某种表示意义的 向量解码为需要的输出 形式 </td>
    <td> MLP, CRF </td>
</tr>
</table>


## 完整模型
fastNLP 为不同的 NLP 任务实现了许多完整的模型，它们都经过了训练和测试。

你可以在以下两个地方查看相关信息
- [介绍](reproduction/)
- [源码](fastNLP/models/)

## 项目结构

![](./docs/source/figures/workflow.png)

fastNLP的大致工作流程如上图所示，而项目结构如下：

<table>
<tr>
    <td><b> fastNLP </b></td>
    <td> 开源的自然语言处理库 </td>
</tr>
<tr>
    <td><b> fastNLP.core </b></td>
    <td> 实现了核心功能，包括数据处理组件、训练器、测速器等 </td>
</tr>
<tr>
    <td><b> fastNLP.models </b></td>
    <td> 实现了一些完整的神经网络模型 </td>
</tr>
<tr>
    <td><b> fastNLP.modules </b></td>
    <td> 实现了用于搭建神经网络模型的诸多组件 </td>
</tr>
<tr>
    <td><b> fastNLP.io </b></td>
    <td> 实现了读写功能，包括数据读入，模型读写等 </td>
</tr>
</table>

## 参考资源

- [教程](https://github.com/fastnlp/fastNLP/tree/master/tutorials)
- [文档](https://fastnlp.readthedocs.io/en/latest/)
- [源码](https://github.com/fastnlp/fastNLP)



*In memory of @FengZiYjun.  May his soul rest in peace. We will miss you very very much!*