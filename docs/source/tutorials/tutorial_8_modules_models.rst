======================================
使用Modules和Models快速搭建自定义模型
======================================

:mod:`~fastNLP.modules` 和 :mod:`~fastNLP.models` 用于构建 fastNLP 所需的神经网络模型，它可以和 torch.nn 中的模型一起使用。
下面我们会分三节介绍编写构建模型的具体方法。


使用 models 中的模型
----------------------

fastNLP 在 :mod:`~fastNLP.models` 模块中内置了如 :class:`~fastNLP.models.CNNText` 、
:class:`~fastNLP.models.SeqLabeling` 等完整的模型，以供用户直接使用。
以文本分类的任务为例，我们从 models 中导入 :class:`~fastNLP.models.CNNText` 模型，用它进行训练。

.. code-block:: python

    from fastNLP.models import CNNText

    model_cnn = CNNText((len(vocab),100), num_classes=2, dropout=0.1)

    trainer = Trainer(train_data=train_data, dev_data=dev_data, metrics=metric,
                      loss=loss, device=device, model=model_cnn)
    trainer.train()

在 iPython 环境输入 `model_cnn` ，我们可以看到 ``model_cnn`` 的网络结构

.. parsed-literal::

    CNNText(
      (embed): Embedding(
        (embed): Embedding(16292, 100)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (conv_pool): ConvMaxpool(
        (convs): ModuleList(
          (0): Conv1d(100, 30, kernel_size=(1,), stride=(1,), bias=False)
          (1): Conv1d(100, 40, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
          (2): Conv1d(100, 50, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
        )
      )
      (dropout): Dropout(p=0.1, inplace=False)
      (fc): Linear(in_features=120, out_features=2, bias=True)
    )

FastNLP 中内置的 models 如下表所示，您可以点击具体的名称查看详细的 API：

.. csv-table::
   :header: 名称, 介绍

   :class:`~fastNLP.models.CNNText` , 使用 CNN 进行文本分类的模型
   :class:`~fastNLP.models.SeqLabeling` , 简单的序列标注模型
   :class:`~fastNLP.models.AdvSeqLabel` , 更大网络结构的序列标注模型
   :class:`~fastNLP.models.ESIM` , ESIM 模型的实现
   :class:`~fastNLP.models.StarTransEnc` , 带 word-embedding的Star-Transformer模 型
   :class:`~fastNLP.models.STSeqLabel` , 用于序列标注的 Star-Transformer 模型
   :class:`~fastNLP.models.STNLICls` ,用于自然语言推断 (NLI) 的 Star-Transformer 模型
   :class:`~fastNLP.models.STSeqCls` , 用于分类任务的 Star-Transformer 模型
   :class:`~fastNLP.models.BiaffineParser` , Biaffine 依存句法分析网络的实现
   :class:`~fastNLP.models.BiLSTMCRF`, 使用BiLSTM与CRF进行序列标注


使用 nn.torch 编写模型
----------------------------

FastNLP 完全支持使用 pyTorch 编写的模型，但与 pyTorch 中编写模型的常见方法不同，
用于 fastNLP 的模型中 forward 函数需要返回一个字典，字典中至少需要包含 ``pred`` 这个字段。

下面是使用 pyTorch 中的 torch.nn 模块编写的文本分类，注意观察代码中标注的向量维度。
由于 pyTorch 使用了约定俗成的维度设置，使得 forward 中需要多次处理维度顺序

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

我们同样可以在 iPython 环境中查看这个模型的网络结构

.. parsed-literal::

    LSTMText(
      (embedding): Embedding(16292, 100)
      (lstm): LSTM(100, 64, num_layers=2, dropout=0.5, bidirectional=True)
      (fc): Linear(in_features=128, out_features=2, bias=True)
      (dropout): Dropout(p=0.5, inplace=False)
    )


使用 modules 编写模型
----------------------------

下面我们使用 :mod:`fastNLP.modules` 中的组件来构建同样的网络。由于 fastNLP 统一把 ``batch_size`` 放在第一维，
在编写代码的过程中会有一定的便利。

.. code-block:: python

    from fastNLP.modules import Embedding, LSTM, MLP

    class MyText(nn.Module):
        def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.5):
            super().__init__()

            self.embedding = Embedding((vocab_size, embedding_dim))
            self.lstm = LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
            self.mlp = MLP([hidden_dim*2,output_dim], dropout=dropout)

        def forward(self, words):
            embedded = self.embedding(words)
            _,(hidden,_) = self.lstm(embedded)
            pred = self.mlp(torch.cat((hidden[-1],hidden[-2]),dim=1))
            return {"pred":pred}

我们自己编写模型的网络结构如下

.. parsed-literal::

    MyText(
      (embedding): Embedding(
        (embed): Embedding(16292, 100)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (lstm): LSTM(
        (lstm): LSTM(100, 64, num_layers=2, batch_first=True, bidirectional=True)
      )
      (mlp): MLP(
        (hiddens): ModuleList()
        (output): Linear(in_features=128, out_features=2, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
      )
    )

FastNLP 中包含的各种模块如下表，您可以点击具体的名称查看详细的 API，也可以通过 :doc:`/fastNLP.modules` 进行了解。

.. csv-table::
   :header: 名称, 介绍

   :class:`~fastNLP.modules.ConvolutionCharEncoder` , char级别的卷积 encoder
   :class:`~fastNLP.modules.LSTMCharEncoder` , char级别基于LSTM的 encoder
   :class:`~fastNLP.modules.ConvMaxpool` , 结合了Convolution和Max-Pooling于一体的模块
   :class:`~fastNLP.modules.LSTM` , LSTM模块, 轻量封装了PyTorch的LSTM
   :class:`~fastNLP.modules.StarTransformer` , Star-Transformer 的encoder部分
   :class:`~fastNLP.modules.TransformerEncoder` , Transformer的encoder模块，不包含embedding层
   :class:`~fastNLP.modules.VarRNN` , Variational Dropout RNN 模块
   :class:`~fastNLP.modules.VarLSTM` , Variational Dropout LSTM 模块
   :class:`~fastNLP.modules.VarGRU` , Variational Dropout GRU 模块
   :class:`~fastNLP.modules.MaxPool` , Max-pooling模块
   :class:`~fastNLP.modules.MaxPoolWithMask` , 带mask矩阵的max pooling。在做 max-pooling的时候不会考虑mask值为0的位置。
   :class:`~fastNLP.modules.AvgPool` , Average-pooling模块
   :class:`~fastNLP.modules.AvgPoolWithMask` , 带mask矩阵的average pooling。在做 average-pooling的时候不会考虑mask值为0的位置。
   :class:`~fastNLP.modules.MultiHeadAttention` , MultiHead Attention 模块
   :class:`~fastNLP.modules.MLP` , 简单的多层感知器模块
   :class:`~fastNLP.modules.ConditionalRandomField` , 条件随机场模块
   :class:`~fastNLP.modules.viterbi_decode` , 给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数 （与 :class:`~fastNLP.modules.ConditionalRandomField` 配合使用）
   :class:`~fastNLP.modules.allowed_transitions` , 给定一个id到label的映射表，返回所有可以跳转的列表（与 :class:`~fastNLP.modules.ConditionalRandomField` 配合使用）
   :class:`~fastNLP.modules.TimestepDropout` , 简单包装过的Dropout 组件


----------------------------------
代码下载
----------------------------------

`点击下载 IPython Notebook 文件 <https://sourcegraph.com/github.com/fastnlp/fastNLP@master/-/raw/tutorials/tutorial_8_modules_models.ipynb>`_)