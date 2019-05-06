fastNLP 中文文档
=====================

fastNLP 是一款轻量级的 NLP 处理套件。你既可以使用它快速地完成一个命名实体识别（NER）、中文分词或文本分类任务；
也可以使用他构建许多复杂的网络模型，进行科研。它具有如下的特性:

- 代码简洁易懂，有着详尽的中文文档以供查阅；
- 深度学习的各个阶段划分明确，配合 fitlog 使用让科研更轻松；
- 内置多种常见模型 （TODO）；
- 基于 PyTorch ，方便从原生 PyTorch 代码迁入，并能使用 PyTorch 中的各种组件；
- 便于 seq2seq；
- 便于 fine-tune

内置的模块
------------

（TODO）


A deep learning NLP model is the composition of three types of modules:

+-----------------------+-----------------------+-----------------------+
| module type           | functionality         | example               |
+=======================+=======================+=======================+
| encoder               | encode the input into | embedding, RNN, CNN,  |
|                       | some abstract         | transformer           |
|                       | representation        |                       |
+-----------------------+-----------------------+-----------------------+
| aggregator            | aggregate and reduce  | self-attention,       |
|                       | information           | max-pooling           |
+-----------------------+-----------------------+-----------------------+
| decoder               | decode the            | MLP, CRF              |
|                       | representation into   |                       |
|                       | the output            |                       |
+-----------------------+-----------------------+-----------------------+


For example:

.. image:: figures/text_classification.png



各个任务上的结果
-----------------------

（TODO）

快速入门
-------------

TODO


用户手册
---------------

.. toctree::
   :maxdepth: 1

    安装指南 <user/installation>
    快速入门 <user/quickstart>
    使用 fastNLP 分类 <user/task1>
    使用 fastNLP 分词 <user/task2>


API 文档
-------------

除了用户手册之外，你还可以通过查阅 API 文档来找到你所需要的工具。

.. toctree::
   :maxdepth: 2
   
   fastNLP


索引与搜索
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
