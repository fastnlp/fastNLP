==============================
BertEmbedding的各种用法
==============================

fastNLP的BertEmbedding以pytorch-transformer.BertModel的代码为基础，是一个使用BERT对words进行编码的Embedding。

使用BertEmbedding和fastNLP.models.bert里面模型可以搭建BERT应用到五种下游任务的模型。

预训练好的Embedding参数及数据集的介绍和自动下载功能见 :doc:`/tutorials/tutorial_3_embedding` 和
:doc:`/tutorials/tutorial_4_load_dataset`

1. BERT for Squence Classification
----------------------------------

在文本分类任务中，我们采用SST数据集作为例子来介绍BertEmbedding的使用方法。

.. code-block:: python

    import warnings
    import torch
    warnings.filterwarnings("ignore")

    # 载入数据集
    from fastNLP.io import SSTPipe
    data_bundle = SSTPipe(subtree=False, train_subtree=False, lower=False, tokenizer='raw').process_from_file()
    data_bundle

    # 载入BertEmbedding
    from fastNLP.embeddings import BertEmbedding
    embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='en-base-cased', include_cls_sep=True)

    # 载入模型
    from fastNLP.models import BertForSequenceClassification
    model = BertForSequenceClassification(embed, len(data_bundle.get_vocab('target')))

    # 训练模型
    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Adam
    trainer = Trainer(data_bundle.get_dataset('train'), model,
                      optimizer=Adam(model_params=model.parameters(), lr=2e-5),
                      loss=CrossEntropyLoss(), device=[0],
                      batch_size=64, dev_data=data_bundle.get_dataset('dev'),
                      metrics=AccuracyMetric(), n_epochs=2, print_every=1)
    trainer.train()



    # 测试结果并删除模型
    from fastNLP import Tester
    tester = Tester(data_bundle.get_dataset('test'), model, batch_size=128, metrics=AccuracyMetric())
    tester.test()

2. BERT for Sentence Matching
-----------------------------

在Matching任务中，我们采用RTE数据集作为例子来介绍BertEmbedding的使用方法。

.. code-block:: python

    # 载入数据集
    from fastNLP.io import RTEBertPipe
    data_bundle = RTEBertPipe(lower=False, tokenizer='raw').process_from_file()

    # 载入BertEmbedding
    from fastNLP.embeddings import BertEmbedding
    embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='en-base-cased', include_cls_sep=True)


    # 载入模型
    from fastNLP.models import BertForSentenceMatching
    model = BertForSentenceMatching(embed, len(data_bundle.get_vocab('target')))

    # 训练模型
    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Adam
    trainer = Trainer(data_bundle.get_dataset('train'), model,
                      optimizer=Adam(model_params=model.parameters(), lr=2e-5),
                      loss=CrossEntropyLoss(), device=[0],
                      batch_size=16, dev_data=data_bundle.get_dataset('dev'),
                      metrics=AccuracyMetric(), n_epochs=2, print_every=1)
    trainer.train()



