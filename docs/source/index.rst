fastNLP 中文文档
=====================

`fastNLP <https://github.com/fastnlp/fastNLP/>`_ 是一款轻量级的自然语言处理（NLP）工具包。你既可以用它来快速地完成一个NLP任务，
也可以用它在研究中快速构建更复杂的模型。

fastNLP具有如下的特性：

- 统一的Tabular式数据容器，简化数据预处理过程;
- 内置多种数据集的 :class:`~fastNLP.io.Loader` 和 :class:`~fastNLP.io.Pipe` ，省去预处理代码;
- 各种方便的NLP工具，例如Embedding加载（包括 :class:`~fastNLP.embeddings.ElmoEmbedding` 和 :class:`~fastNLP.embeddings.BertEmbedding` ）、中间数据cache等;
- 部分 `数据集与预训练模型 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_ 的自动下载;
- 提供多种神经网络组件以及复现模型（涵盖中文分词、命名实体识别、句法分析、文本分类、文本匹配、指代消解、摘要等任务）;
- :class:`~fastNLP.Trainer` 提供多种内置 :mod:`~fastNLP.core.callback` 函数，方便实验记录、异常捕获等.


用户手册
----------------

.. toctree::
   :maxdepth: 2

    安装指南 </user/installation>
    快速入门 </user/quickstart>
    详细教程 </user/tutorials>

API 文档
-------------

除了用户手册之外，你还可以通过查阅 API 文档来找到你所需要的工具。

.. toctree::
   :titlesonly:
   :maxdepth: 2
   
   fastNLP

fitlog文档
----------

您可以 `点此 <https://fitlog.readthedocs.io/zh/latest/>`_  查看fitlog的文档。
fitlog 是由我们团队开发的日志记录+代码管理的工具。

索引与搜索
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
