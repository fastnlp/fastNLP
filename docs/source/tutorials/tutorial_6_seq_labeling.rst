=====================
快速实现序列标注模型
=====================

这一部分的内容主要展示如何使用fastNLP 实现序列标注任务。你可以使用fastNLP的各个组件快捷，方便地完成序列标注任务，达到出色的效果。
在阅读这篇Tutorial前，希望你已经熟悉了fastNLP的基础使用，包括基本数据结构以及数据预处理，embedding的嵌入等，希望你对之前的教程有更进一步的掌握。
我们将对CoNLL-03的英文数据集进行处理，展示如何完成命名实体标注任务整个训练的过程。

载入数据
===================================
fastNLP可以方便地载入各种类型的数据。同时，针对常见的数据集，我们已经预先实现了载入方法，其中包含CoNLL-03数据集。
在设计dataloader时，以DataSetLoader为基类，可以改写并应用于其他数据集的载入。

.. code-block:: python

    class Conll2003DataLoader(DataSetLoader):
    def __init__(self, task:str='ner', encoding_type:str='bioes'):
        assert task in ('ner', 'pos', 'chunk')
        index = {'ner':3, 'pos':1, 'chunk':2}[task]
        #ConllLoader是fastNLP内置的类
        self._loader = ConllLoader(headers=['raw_words', 'target'], indexes=[0, index])
        self._tag_converters = None
        if task in ('ner', 'chunk'):
            #iob和iob2bioes会对tag进行统一，标准化
            self._tag_converters = [iob2]
            if encoding_type == 'bioes':
                self._tag_converters.append(iob2bioes)

    def load(self, path: str):
        dataset = self._loader.load(path)
        def convert_tag_schema(tags):
            for converter in self._tag_converters:
                tags = converter(tags)
            return tags
        if self._tag_converters:
        #使用apply实现convert_tag_schema函数，实际上也支持匿名函数
            dataset.apply_field(convert_tag_schema, field_name=Const.TARGET, new_field_name=Const.TARGET)
        return dataset

输出数据格式如：

    {'raw_words': ['on', 'Friday', ':'] type=list,
    'target': ['O', 'O', 'O'] type=list},


数据处理
----------------------------
我们进一步处理数据。将数据和词表封装在 :class:`~fastNLP.DataInfo` 类中。data是DataInfo的实例。
我们输入模型的数据包括char embedding，以及word embedding。在数据处理部分，我们尝试完成词表的构建。
使用fastNLP中的Vocabulary类来构建词表。

.. code-block:: python

    word_vocab = Vocabulary(min_freq=2)
    word_vocab.from_dataset(data.datasets['train'], field_name=Const.INPUT)
    word_vocab.index_dataset(*data.datasets.values(),field_name=Const.INPUT, new_field_name=Const.INPUT)

处理后的data对象内部为：

    dataset
    vocabs
    dataset保存了train和test中的数据，并保存为dataset类型
    vocab保存了words，raw-words以及target的词表。

模型构建
--------------------------------
我们使用CNN-BILSTM-CRF模型完成这一任务。在网络构建方面，fastNLP的网络定义继承pytorch的 :class:`nn.Module` 类。
自己可以按照pytorch的方式定义网络。需要注意的是命名。fastNLP的标准命名位于 :class:`~fastNLP.Const` 类。

模型的训练
首先实例化模型，导入所需的char embedding以及word embedding。Embedding的载入可以参考教程。
也可以查看 :mod:`~fastNLP.modules.encoder.embedding` 使用所需的embedding 载入方法。
fastNLP将模型的训练过程封装在了 :class:`~fastnlp.trainer` 类中。
根据不同的任务调整trainer中的参数即可。通常，一个trainer实例需要有：指定的训练数据集，模型，优化器，loss函数，评测指标，以及指定训练的epoch数，batch size等参数。

.. code-block:: python

    #实例化模型
    model = CNNBiLSTMCRF(word_embed, char_embed, hidden_size=200, num_layers=1, tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type)
    #定义优化器
    optimizer = Adam(model.parameters(), lr=0.005)
    #定义评估指标
    Metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type)
    #实例化trainer
    trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer, dev_data=data.datasets['test'], batch_size=10, metrics=Metrics,callbacks=callbacks, n_epochs=100)
    #开始训练
    trainer.train()
    
训练中会保存最优的参数配置。
训练的结果如下：

.. code-block:: python

    Evaluation on DataSet test:                                                                                          
    SpanFPreRecMetric: f=0.727661, pre=0.732293, rec=0.723088
    Evaluation at Epoch 1/100. Step:1405/140500. SpanFPreRecMetric: f=0.727661, pre=0.732293, rec=0.723088
    
    Evaluation on DataSet test:
    SpanFPreRecMetric: f=0.784307, pre=0.779371, rec=0.789306
    Evaluation at Epoch 2/100. Step:2810/140500. SpanFPreRecMetric: f=0.784307, pre=0.779371, rec=0.789306
    
    Evaluation on DataSet test:                                                                                          
    SpanFPreRecMetric: f=0.810068, pre=0.811003, rec=0.809136
    Evaluation at Epoch 3/100. Step:4215/140500. SpanFPreRecMetric: f=0.810068, pre=0.811003, rec=0.809136
    
    Evaluation on DataSet test:                                                                                          
    SpanFPreRecMetric: f=0.829592, pre=0.84153, rec=0.817989
    Evaluation at Epoch 4/100. Step:5620/140500. SpanFPreRecMetric: f=0.829592, pre=0.84153, rec=0.817989
    
    Evaluation on DataSet test:
    SpanFPreRecMetric: f=0.828789, pre=0.837096, rec=0.820644
    Evaluation at Epoch 5/100. Step:7025/140500. SpanFPreRecMetric: f=0.828789, pre=0.837096, rec=0.820644


