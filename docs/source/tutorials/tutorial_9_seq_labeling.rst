=====================
快速实现序列标注模型
=====================

这一部分的内容主要展示如何使用fastNLP 实现序列标注任务。你可以使用fastNLP的各个组件快捷，方便地完成序列标注任务，达到出色的效果。
在阅读这篇Tutorial前，希望你已经熟悉了fastNLP的基础使用，尤其是数据的载入以及模型的构建，通过这个小任务的能让你进一步熟悉fastNLP的使用。
我们将对基于Weibo的中文社交数据集进行处理，展示如何完成命名实体标注任务的整个过程。

载入数据
===================================
fastNLP的数据载入主要是由Loader与Pipe两个基类衔接完成的。通过Loader可以方便地载入各种类型的数据。同时，针对常见的数据集，我们已经预先实现了载入方法，其中包含weibo数据集。
在设计dataloader时，以DataSetLoader为基类，可以改写并应用于其他数据集的载入。

.. code-block:: python

	from fastNLP.io import WeiboNERLoader
	data_bundle = WeiboNERLoader().load()



载入后的数据如 ::

	{'dev': DataSet(
	{{'raw_chars': ['用', '最', '大', '努', '力', '去', '做''人', '生', '。', '哈', '哈', '哈', '哈', '哈', '哈', '
    'target': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',, 'O', 'O', 'O', 'O', 'O', 'O'] type=list})}

	{'test': DataSet(
	{{'raw_chars': ['感', '恩', '大', '回', '馈'] type=list,   'target': ['O', 'O', 'O', 'O', 'O'] type=list})}

	{'train': DataSet(
	{'raw_chars': ['国', '安', '老', '球', '迷'] type=list,   'target': ['B-ORG.NAM', 'I-ORG.NAM', 'B-PER.NOM', 'I-PER.NOM', 'I-PER.NOM'] type=list})}



数据处理
----------------------------
我们进一步处理数据。通过Pipe基类处理Loader载入的数据。 如果你还有印象，应该还能想起，实现自定义数据集的Pipe时，至少要编写process 函数或者process_from_file 函数。前者接受 :class:`~fastNLP.DataBundle` 类的数据，并返回该 :class:`~fastNLP.DataBundle`  。后者接收数据集所在文件夹为参数，读取并处理为 :class:`~fastNLP.DataBundle` 后，通过process 函数处理数据。
这里我们已经实现通过Loader载入数据，并已返回 :class:`~fastNLP.DataBundle` 类的数据。我们编写process 函数以处理Loader载入后的数据。

.. code-block:: python

    from fastNLP.io import ChineseNERPipe
    data_bundle = ChineseNERPipe(encoding_type='bioes', bigram=True).process(data_bundle)

载入后的数据如下 ::

    {'raw_chars': ['用', '最', '大', '努', '力', '去', '做', '值', '得', '的', '事', '人', '生', '。', '哈', '哈', '哈', '哈', '哈', '哈', '我', '在'] type=list,
    'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] type=list,
    'chars': [97, 71, 34, 422, 104, 72, 144, 628, 66, 3, 158, 2, 9, 647, 485, 196, 2,19] type=list,
    'bigrams': [5948, 1950, 34840, 98, 8413, 3961, 34841, 631, 34842, 407, 462, 45, 3 1959, 1619, 3, 3, 3, 3, 3, 2663, 29, 90] type=list,
    'seq_len': 30 type=int}

模型构建
--------------------------------
我们使用CNN-BILSTM-CRF模型完成这一任务。在网络构建方面，fastNLP的网络定义继承pytorch的 :class:`nn.Module` 类。
自己可以按照pytorch的方式定义网络。需要注意的是命名。fastNLP的标准命名位于 :class:`~fastNLP.Const` 类。

模型的训练
首先实例化模型，导入所需的char embedding以及word embedding。Embedding的载入可以参考教程。
也可以查看 :mod:`~fastNLP.embedding` 使用所需的embedding 载入方法。
fastNLP将模型的训练过程封装在了 :class:`~fastnlp.Trainer` 类中。
根据不同的任务调整trainer中的参数即可。通常，一个trainer实例需要有：指定的训练数据集，模型，优化器，loss函数，评测指标，以及指定训练的epoch数，batch size等参数。

.. code-block:: python

    #实例化模型
    model = CNBiLSTMCRFNER(char_embed, num_classes=len(data_bundle.vocabs['target']), bigram_embed=bigram_embed)
    #定义评估指标
    Metrics=SpanFPreRecMetric(data_bundle.vocabs['target'], encoding_type='bioes')
    #实例化trainer并训练
    Trainer(data_bundle.datasets['train'], model, batch_size=20, metrics=Metrics, num_workers=2, dev_data=data_bundle. datasets['dev']).train()

    
训练中会保存最优的参数配置。

训练的结果如下 ::

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


