===============
详细指南
===============

我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。给出一段文字，预测它的标签是0~4中的哪一个
(数据来源 `kaggle <https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews>`_ )。

--------------
数据处理
--------------

数据读入
    我们可以使用 fastNLP  :mod:`fastNLP.io` 模块中的 :class:`~fastNLP.io.CSVLoader` 类，轻松地从 csv 文件读取我们的数据。
    这里的 dataset 是 fastNLP 中 :class:`~fastNLP.DataSet` 类的对象

    .. code-block:: python

        from fastNLP.io import CSVLoader

        loader = CSVLoader(headers=('raw_sentence', 'label'), sep='\t')
        dataset = loader.load("./sample_data/tutorial_sample_dataset.csv")

    除了读取数据外，fastNLP 还提供了读取其它文件类型的 Loader 类、读取 Embedding的 Loader 等。详见 :doc:`/fastNLP.io` 。

Instance 和 DataSet
    fastNLP 中的 :class:`~fastNLP.DataSet` 类对象类似于二维表格，它的每一列是一个 :mod:`~fastNLP.core.field`
    每一行是一个 :mod:`~fastNLP.core.instance` 。我们可以手动向数据集中添加 :class:`~fastNLP.Instance` 类的对象

    .. code-block:: python

        from fastNLP import Instance

        dataset.append(Instance(raw_sentence='fake data', label='0'))

    此时的 ``dataset[-1]`` 的值如下,可以看到，数据集中的每个数据包含 ``raw_sentence`` 和 ``label`` 两个
    :mod:`~fastNLP.core.field` ，他们的类型都是 ``str`` ::

        {'raw_sentence': fake data type=str, 'label': 0 type=str}

field 的修改
    我们使用 :class:`~fastNLP.DataSet` 类的 :meth:`~fastNLP.DataSet.apply` 方法将 ``raw_sentence`` 中字母变成小写，并将句子分词。
    同时也将 ``label`` :mod:`~fastNLP.core.field` 转化为整数并改名为 ``target``

    .. code-block:: python

        dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
        dataset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words')
        dataset.apply(lambda x: int(x['label']), new_field_name='target')

    ``words`` 和 ``target`` 已经足够用于 :class:`~fastNLP.models.CNNText` 的训练了，但我们从其文档
    :class:`~fastNLP.models.CNNText` 中看到，在 :meth:`~fastNLP.models.CNNText.forward` 的时候，还可以传入可选参数 ``seq_len`` 。
    所以，我们再使用 :meth:`~fastNLP.DataSet.apply_field` 方法增加一个名为 ``seq_len`` 的 :mod:`~fastNLP.core.field` 。

    .. code-block:: python

        dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')

    观察可知： :meth:`~fastNLP.DataSet.apply_field` 与 :meth:`~fastNLP.DataSet.apply` 类似，
    但所传入的 `lambda` 函数是针对一个 :class:`~fastNLP.Instance` 中的一个 :mod:`~fastNLP.core.field` 的；
    而 :meth:`~fastNLP.DataSet.apply` 所传入的 `lambda` 函数是针对整个 :class:`~fastNLP.Instance` 的。

    .. note::
         `lambda` 函数即匿名函数，是 Python 的重要特性。 ``lambda x: len(x)``  和下面的这个函数的作用相同::

            def func_lambda(x):
                return len(x)

        你也可以编写复杂的函数做为 :meth:`~fastNLP.DataSet.apply_field` 与 :meth:`~fastNLP.DataSet.apply` 的参数

Vocabulary 的使用
    我们再用 :class:`~fastNLP.Vocabulary` 类来统计数据中出现的单词，并使用 :meth:`~fastNLP.Vocabularyindex_dataset`
    将单词序列转化为训练可用的数字序列。

    .. code-block:: python

        from fastNLP import Vocabulary

        vocab = Vocabulary(min_freq=2).from_dataset(dataset, field_name='words')
        vocab.index_dataset(dataset, field_name='words',new_field_name='words')

数据集分割
    除了修改 :mod:`~fastNLP.core.field` 之外，我们还可以对 :class:`~fastNLP.DataSet` 进行分割，以供训练、开发和测试使用。
    下面这段代码展示了 :meth:`~fastNLP.DataSet.split` 的使用方法（但实际应该放在后面两段改名和设置输入的代码之后）

    .. code-block:: python

        train_dev_data, test_data = dataset.split(0.1)
        train_data, dev_data = train_dev_data.split(0.1)
        len(train_data), len(dev_data), len(test_data)

---------------------
使用内置模型训练
---------------------

内置模型的输入输出命名
    fastNLP内置了一些完整的神经网络模型，详见 :doc:`/fastNLP.models` , 我们使用其中的 :class:`~fastNLP.models.CNNText` 模型进行训练。
    为了使用内置的 :class:`~fastNLP.models.CNNText`，我们必须修改 :class:`~fastNLP.DataSet` 中 :mod:`~fastNLP.core.field` 的名称。
    在这个例子中模型输入 (forward方法的参数) 为 ``words`` 和 ``seq_len`` ; 预测输出为 ``pred`` ;标准答案为 ``target`` 。
    具体的命名规范可以参考 :doc:`/fastNLP.core.const` 。

    如果不想查看文档，您也可以使用 :class:`~fastNLP.Const` 类进行命名。下面的代码展示了给 :class:`~fastNLP.DataSet` 中
    :mod:`~fastNLP.core.field` 改名的 :meth:`~fastNLP.DataSet.rename_field` 方法，以及 :class:`~fastNLP.Const` 类的使用方法。

    .. code-block:: python

        from fastNLP import Const

        dataset.rename_field('words', Const.INPUT)
        dataset.rename_field('seq_len', Const.INPUT_LEN)
        dataset.rename_field('target', Const.TARGET)

    在给 :class:`~fastNLP.DataSet` 中 :mod:`~fastNLP.core.field` 改名后，我们还需要设置训练所需的输入和目标，这里使用的是
    :meth:`~fastNLP.DataSet.set_input` 和 :meth:`~fastNLP.DataSet.set_target` 两个函数。

    .. code-block:: python

        dataset.set_input(Const.INPUT, Const.INPUT_LEN)
        dataset.set_target(Const.TARGET)

快速训练
    现在我们可以导入 fastNLP 内置的文本分类模型 :class:`~fastNLP.models.CNNText` ，并使用 :class:`~fastNLP.Trainer` 进行训练了
    （其中 ``loss`` 和 ``metrics`` 的定义，我们将在后续两段代码中给出）。

    .. code-block:: python

        from fastNLP.models import CNNText
        from fastNLP import Trainer

        model = CNNText((len(vocab),50), num_classes=5, padding=2, dropout=0.1)

        trainer = Trainer(model=model_cnn, train_data=train_data, dev_data=dev_data,
                        loss=loss, metrics=metrics)
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

损失函数
    训练模型需要提供一个损失函数, 下面提供了一个在分类问题中常用的交叉熵损失。注意它的 **初始化参数** 。
    ``pred`` 参数对应的是模型的 forward 方法返回的 dict 中的一个 key 的名字。
    ``target`` 参数对应的是 :class:`~fastNLP.DataSet` 中作为标签的 :mod:`~fastNLP.core.field` 的名字。
    这里我们用 :class:`~fastNLP.Const` 来辅助命名，如果你自己编写模型中 forward 方法的返回值或
    数据集中 :mod:`~fastNLP.core.field` 的名字与本例不同， 你可以把 ``pred`` 参数和 ``target`` 参数设定符合自己代码的值。

    .. code-block:: python

        from fastNLP import CrossEntropyLoss

        # loss = CrossEntropyLoss() 在本例中与下面这行代码等价
        loss = CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET)

评价指标
    训练模型需要提供一个评价指标。这里使用准确率做为评价指标。参数的 `命名规则` 跟上面类似。
    ``pred`` 参数对应的是模型的 forward 方法返回的 dict 中的一个 key 的名字。
    ``target`` 参数对应的是 :class:`~fastNLP.DataSet` 中作为标签的 :mod:`~fastNLP.core.field` 的名字。

    .. code-block:: python

        from fastNLP import AccuracyMetric

        # metrics=AccuracyMetric() 在本例中与下面这行代码等价
        metrics=AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)

快速测试
    与 :class:`~fastNLP.Trainer` 对应，fastNLP 也提供了 :class:`~fastNLP.Tester` 用于快速测试，用法如下

    .. code-block:: python

        from fastNLP import Tester

        tester = Tester(test_data, model_cnn, metrics=AccuracyMetric())
        tester.test()

---------------------
编写自己的模型
---------------------

因为 fastNLP 是基于 `PyTorch <https://pytorch.org/>`_ 开发的框架，所以我们可以基于 PyTorch 模型编写自己的神经网络模型。
与标准的 PyTorch 模型不同，fastNLP 模型中 forward 方法返回的是一个字典，字典中至少需要包含 "pred" 这个字段。
而 forward 方法的参数名称必须与 :class:`~fastNLP.DataSet` 中用 :meth:`~fastNLP.DataSet.set_input` 设定的名称一致。
模型定义的代码如下:

.. code-block:: python

    import torch
    import torch.nn as nn

    class LSTMText(nn.Module):
        def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.5):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, words):
            # (input) words : (batch_size, seq_len)
            words = words.permute(1,0)
            # words : (seq_len, batch_size)

            embedded = self.dropout(self.embedding(words))
            # embedded : (seq_len, batch_size, embedding_dim)
            output, (hidden, cell) = self.lstm(embedded)
            # output: (seq_len, batch_size, hidden_dim * 2)
            # hidden: (num_layers * 2, batch_size, hidden_dim)
            # cell: (num_layers * 2, batch_size, hidden_dim)

            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden = self.dropout(hidden)
            # hidden: (batch_size, hidden_dim * 2)

            pred = self.fc(hidden.squeeze(0))
            # result: (batch_size, output_dim)
            return {"pred":pred}

模型的使用方法与内置模型 :class:`~fastNLP.models.CNNText`  一致

.. code-block:: python

    model_lstm = LSTMText(len(vocab),50,5)

    trainer = Trainer(model=model_lstm, train_data=train_data, dev_data=dev_data,
                    loss=loss, metrics=metrics)
    trainer.train()

    tester = Tester(test_data, model_lstm, metrics=AccuracyMetric())
    tester.test()

.. todo::
    使用 :doc:`/fastNLP.modules` 编写模型

--------------------------
自己编写训练过程
--------------------------

如果你想用类似 PyTorch 的使用方法，自己编写训练过程，你可以参考下面这段代码。其中使用了 fastNLP 提供的 :class:`~fastNLP.Batch`
来获得小批量训练的小批量数据，使用 :class:`~fastNLP.BucketSampler` 做为 :class:`~fastNLP.Batch` 的参数来选择采样的方式。
这段代码中使用了 PyTorch 的 `torch.optim.Adam` 优化器 和 `torch.nn.CrossEntropyLoss` 损失函数，并自己计算了正确率

.. code-block:: python

    from fastNLP import BucketSampler
    from fastNLP import Batch
    import torch
    import time

    model = CNNText((len(vocab),50), num_classes=5, padding=2, dropout=0.1)

    def train(epoch, data):
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        lossfunc = torch.nn.CrossEntropyLoss()
        batch_size = 32

        train_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
        train_batch = Batch(batch_size=batch_size, dataset=data, sampler=train_sampler)

        start_time = time.time()
        for i in range(epoch):
            loss_list = []
            for batch_x, batch_y in train_batch:
                optim.zero_grad()
                output = model(batch_x['words'])
                loss = lossfunc(output['pred'], batch_y['target'])
                loss.backward()
                optim.step()
                loss_list.append(loss.item())
            print('Epoch {:d} Avg Loss: {:.2f}'.format(i, sum(loss_list) / len(loss_list)),end=" ")
            print('{:d}ms'.format(round((time.time()-start_time)*1000)))
            loss_list.clear()

    train(10, train_data)

    tester = Tester(test_data, model, metrics=AccuracyMetric())
    tester.test()

这段代码的输出如下::

    Epoch 0 Avg Loss: 2.76 17ms
    Epoch 1 Avg Loss: 2.55 29ms
    Epoch 2 Avg Loss: 2.37 41ms
    Epoch 3 Avg Loss: 2.30 53ms
    Epoch 4 Avg Loss: 2.12 65ms
    Epoch 5 Avg Loss: 2.16 76ms
    Epoch 6 Avg Loss: 1.88 88ms
    Epoch 7 Avg Loss: 1.84 99ms
    Epoch 8 Avg Loss: 1.71 111ms
    Epoch 9 Avg Loss: 1.62 122ms
    [tester]
    AccuracyMetric: acc=0.142857

----------------------------------
使用 Callback 增强 Trainer
----------------------------------

如果你不想自己实现繁琐的训练过程，只希望在训练过程中实现一些自己的功能（比如：输出从训练开始到当前 batch 结束的总时间），
你可以使用 fastNLP 提供的 :class:`~fastNLP.Callback` 类。下面的例子中，我们继承 :class:`~fastNLP.Callback` 类实现了这个功能。

.. code-block:: python

    from fastNLP import Callback

    start_time = time.time()

    class MyCallback(Callback):
        def on_epoch_end(self):
            print('Sum Time: {:d}ms\n\n'.format(round((time.time()-start_time)*1000)))


    model = CNNText((len(vocab),50), num_classes=5, padding=2, dropout=0.1)
    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data,
                      loss=CrossEntropyLoss(), metrics=AccuracyMetric(), callbacks=[MyCallback()])
    trainer.train()

训练输出如下::

    input fields after batch(if batch size is 2):
        words: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 16])
        seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2])
    target fields after batch(if batch size is 2):
        target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2])

    training epochs started 2019-05-12-21-38-40
    Evaluation at Epoch 1/10. Step:2/20. AccuracyMetric: acc=0.285714

    Sum Time: 51ms


    …………………………


    Evaluation at Epoch 10/10. Step:20/20. AccuracyMetric: acc=0.857143

    Sum Time: 212ms



    In Epoch:10/Step:20, got best dev performance:AccuracyMetric: acc=0.857143
    Reloaded the best model.

这个例子只是介绍了 :class:`~fastNLP.Callback` 类的使用方法。实际应用（比如：负采样、Learning Rate Decay、Early Stop 等）中
很多功能已经被 fastNLP 实现了。你可以直接 import 它们使用，详细请查看文档 :doc:`/fastNLP.core.callback` 。