fastNLP 中文文档
=====================

`fastNLP <https://github.com/fastnlp/fastNLP/>`_ 是一款轻量级的 NLP 处理套件。你既可以使用它快速地完成一个序列标注
（NER、POS-Tagging等）、中文分词、文本分类、Matching、指代消解、摘要等任务
（详见 `reproduction <https://github.com/fastnlp/fastNLP/tree/master/reproduction>`_ ）；
也可以使用它构建许多复杂的网络模型，进行科研。它具有如下的特性：

- 统一的Tabular式数据容器，让数据预处理过程简洁明了。内置多种数据集的 :mod:`~fastNLP.io.data_loader` ，省去预处理代码;
- 多种训练、测试组件，例如训练器 :class:`~fastNLP.Trainer` ；测试器 :class:`~fastNLP.Tester` ；以及各种评测 :mod:`~fastNLP.core.metrics` 等等;
- 各种方便的NLP工具，例如预处理 :mod:`embedding<fastNLP.embeddings>` 加载（包括ELMo和BERT）; 中间数据存储 :func:`cache <fastNLP.cache_results>` 等;
- 提供诸多高级模块 :mod:`~fastNLP.modules`，例如 :class:`~fastNLP.modules.VarLSTM` , :class:`Transformer<fastNLP.modules.TransformerEncoder>` , :class:`CRF<fastNLP.modules.ConditionalRandomField>` 等;
- 在序列标注、中文分词、文本分类、Matching、指代消解、摘要等任务上封装了各种 :mod:`~fastNLP.models` 可供直接使用;
- 训练器便捷且具有扩展性，提供多种内置 :mod:`~fastNLP.core.callback` 函数，方便实验记录、异常捕获等。


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
