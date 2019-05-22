===============
快速入门
===============

这是一个简单的分类任务 (数据来源 `kaggle <https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews>`_ )。
给出一段文字，预测它的标签是0~4中的哪一个。

我们可以使用 fastNLP 中 io 模块中的  :class:`~fastNLP.io.CSVLoader` 类，轻松地从 csv 文件读取我们的数据。

.. code-block:: python

    from fastNLP.io import CSVLoader

    loader = CSVLoader(headers=('raw_sentence', 'label'), sep='\t')
    dataset = loader.load("./sample_data/tutorial_sample_dataset.csv")

此时的 `dataset[0]` 的值如下,可以看到，数据集中的每个数据包含 ``raw_sentence`` 和 ``label`` 两个字段，他们的类型都是 ``str``::

    {'raw_sentence': A series of escapades demonstrating the adage that what is good for the
    goose is also good for the gander , some of which occasionally amuses but none of which
    amounts to much of a story . type=str,
    'label': 1 type=str}


我们使用 :class:`~fastNLP.DataSet` 类的 :meth:`~fastNLP.DataSet.apply` 方法将 ``raw_sentence`` 中字母变成小写，并将句子分词。

.. code-block:: python

    dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    dataset.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)

然后我们再用 :class:`~fastNLP.Vocabulary` 类来统计数据中出现的单词，并将单词序列转化为训练可用的数字序列。

.. code-block:: python

    from fastNLP import Vocabulary
    vocab = Vocabulary(min_freq=2).from_dataset(dataset, field_name='words')
    vocab.index_dataset(dataset, field_name='words',new_field_name='words')

同时，我们也将原来 str 类型的标签转化为数字，并设置为训练中的标准答案 ``target``

.. code-block:: python

    dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

现在我们可以导入 fastNLP 内置的文本分类模型 :class:`~fastNLP.models.CNNText` ，


.. code-block:: python

    from fastNLP.models import CNNText
    model = CNNText((len(vocab),50), num_classes=5, padding=2, dropout=0.1)

:class:`~fastNLP.models.CNNText` 的网络结构如下::

    CNNText(
      (embed): Embedding(
        177, 50
        (dropout): Dropout(p=0.0)
      )
      (conv_pool): ConvMaxpool(
        (convs): ModuleList(
          (0): Conv1d(50, 3, kernel_size=(3,), stride=(1,), padding=(2,))
          (1): Conv1d(50, 4, kernel_size=(4,), stride=(1,), padding=(2,))
          (2): Conv1d(50, 5, kernel_size=(5,), stride=(1,), padding=(2,))
        )
      )
      (dropout): Dropout(p=0.1)
      (fc): Linear(in_features=12, out_features=5, bias=True)
    )

下面我们用 :class:`~fastNLP.DataSet` 类的 :meth:`~fastNLP.DataSet.split` 方法将数据集划分为 ``train_data`` 和 ``dev_data``
两个部分，分别用于训练和验证

.. code-block:: python

    train_data, dev_data = dataset.split(0.2)

最后我们用 fastNLP 的 :class:`~fastNLP.Trainer` 进行训练，训练的过程中需要传入模型 ``model`` ，训练数据集 ``train_data`` ，
验证数据集 ``dev_data`` ，损失函数 ``loss`` 和衡量标准 ``metrics`` 。
其中损失函数使用的是 fastNLP 提供的 :class:`~fastNLP.CrossEntropyLoss` 损失函数;
衡量标准使用的是 fastNLP 提供的 :class:`~fastNLP.AccuracyMetric` 正确率指标。

.. code-block:: python

    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric

    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data,
                      loss=CrossEntropyLoss(), metrics=AccuracyMetric())
    trainer.train()

训练过程的输出如下::

    input fields after batch(if batch size is 2):
        words: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 26])
    target fields after batch(if batch size is 2):
        target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2])

    training epochs started 2019-05-09-10-59-39
    Evaluation at Epoch 1/10. Step:2/20. AccuracyMetric: acc=0.333333

    Evaluation at Epoch 2/10. Step:4/20. AccuracyMetric: acc=0.533333

    Evaluation at Epoch 3/10. Step:6/20. AccuracyMetric: acc=0.533333

    Evaluation at Epoch 4/10. Step:8/20. AccuracyMetric: acc=0.533333

    Evaluation at Epoch 5/10. Step:10/20. AccuracyMetric: acc=0.6

    Evaluation at Epoch 6/10. Step:12/20. AccuracyMetric: acc=0.8

    Evaluation at Epoch 7/10. Step:14/20. AccuracyMetric: acc=0.8

    Evaluation at Epoch 8/10. Step:16/20. AccuracyMetric: acc=0.733333

    Evaluation at Epoch 9/10. Step:18/20. AccuracyMetric: acc=0.733333

    Evaluation at Epoch 10/10. Step:20/20. AccuracyMetric: acc=0.733333


    In Epoch:6/Step:12, got best dev performance:AccuracyMetric: acc=0.8
    Reloaded the best model.

这份教程只是简单地介绍了使用 fastNLP 工作的流程，具体的细节分析见 :doc:`/user/tutorial_one`