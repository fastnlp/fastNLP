======================================
Modules 和 models 的教程
======================================

:mod:`~fastNLP.modules` 和 :mod:`~fastNLP.models` 用于构建 fastNLP 所需的神经网络模型，它可以和 torch.nn 中的模型一起使用。
下面我们会分三节介绍编写构建模型的具体方法。


----------------------
使用 models 中的模型
----------------------

fastNLP 在 :mod:`~fastNLP.models` 模块中内置了如 :class:`~fastNLP.models.CNNText` 、
:class:`~fastNLP.models.SeqLabeling` 等完整的模型，以供用户直接使用。
以 :class:`~fastNLP.models.CNNText` 为例，我们看一个简单的文本分类的任务的实现过程。

首先是数据读入和处理部分，这里的代码和 :doc:`快速入门 </user/quickstart>` 中一致。

.. code-block:: python

    from fastNLP.io import CSVLoader
    from fastNLP import Vocabulary, CrossEntropyLoss, AccuracyMetric

    loader = CSVLoader(headers=('raw_sentence', 'label'), sep='\t')
    dataset = loader.load("./sample_data/tutorial_sample_dataset.csv")

    dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='sentence')
    dataset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words', is_input=True)
    dataset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

    train_dev_data, test_data = dataset.split(0.1)
    train_data, dev_data = train_dev_data.split(0.1)

    vocab = Vocabulary(min_freq=2).from_dataset(train_data, field_name='words')
    vocab.index_dataset(train_data, dev_data, test_data, field_name='words', new_field_name='words')

然后我们从 :mod:`~fastNLP.models` 中导入 ``CNNText`` 模型，用它进行训练

.. code-block:: python

    from fastNLP.models import CNNText
    from fastNLP import Trainer

    model_cnn = CNNText((len(vocab),50), num_classes=5, padding=2, dropout=0.1)

    trainer = Trainer(model=model_cnn, train_data=train_data, dev_data=dev_data,
                      loss=CrossEntropyLoss(), metrics=AccuracyMetric())
    trainer.train()

在 iPython 环境输入 `model_cnn` ，我们可以看到 ``model_cnn`` 的网络结构

.. parsed-literal::

    CNNText(
      (embed): Embedding(
        169, 50
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

----------------------------
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
      (embedding): Embedding(169, 50)
      (lstm): LSTM(50, 64, num_layers=2, dropout=0.5, bidirectional=True)
      (fc): Linear(in_features=128, out_features=5, bias=True)
      (dropout): Dropout(p=0.5)
    )

----------------------------
使用 modules 编写模型
----------------------------

下面我们使用 :mod:`fastNLP.modules` 中的组件来构建同样的网络。由于 fastNLP 统一把 ``batch_size`` 放在第一维，
在编写代码的过程中会有一定的便利。

.. code-block:: python

    from fastNLP.modules import Embedding, LSTM, MLP

    class Model(nn.Module):
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

    Model(
      (embedding): Embedding(
        169, 50
        (dropout): Dropout(p=0.0)
      )
      (lstm): LSTM(
        (lstm): LSTM(50, 64, num_layers=2, batch_first=True, bidirectional=True)
      )
      (mlp): MLP(
        (hiddens): ModuleList()
        (output): Linear(in_features=128, out_features=5, bias=True)
        (dropout): Dropout(p=0.5)
      )
    )