==============================
BertEmbedding的各种用法
==============================

Bert自从在`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_
中被提出后，因其性能卓越受到了极大的关注，在这里我们展示一下在fastNLP中如何使用Bert进行各类任务。其中中文Bert我们使用的模型的权重来自于
`中文Bert预训练 <https://github.com/ymcui/Chinese-BERT-wwm>`_ 。

为了方便大家的使用，fastNLP提供了预训练的Embedding权重及数据集的自动下载，支持自动下载的Embedding和数据集见
`数据集 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?tab=fed5xh&c=D42A0AC0>`_ 。或您可从 doc:`tutorial/tutorial_3_embedding` 与
 doc:`tutorial/tutorial_4_load_dataset` 了解更多相关信息。

----------------------------------
中文任务
----------------------------------
下面我们将介绍通过使用Bert来进行文本分类, 中文命名实体识别, 文本匹配, 中文问答。

1. 使用Bert进行文本分类
----------------------------------
文本分类是指给定一段文字，判定其所属的类别。例如下面的文本情感分类

.. code-block:: text

    1, 商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!

这里我们使用fastNLP提供自动下载的微博分类进行测试

.. code-block:: python

    from fastNLP.io import WeiboSenti100kPipe

    data_bundle =WeiboSenti100kPipe().process_from_file()
    data_bundle.rename_field('chars', 'words')

    # 载入BertEmbedding
    from fastNLP.embeddings import BertEmbedding

    embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='cn-wwm', include_cls_sep=True)

    # 载入模型
    from fastNLP.models import BertForSequenceClassification

    model = BertForSequenceClassification(embed, len(data_bundle.get_vocab('target')))

    # 训练模型
    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Adam

    trainer = Trainer(data_bundle.get_dataset('train'), model,
                      optimizer=Adam(model_params=model.parameters(), lr=2e-5),
                      loss=CrossEntropyLoss(), device=0,
                      batch_size=8, dev_data=data_bundle.get_dataset('dev'),
                      metrics=AccuracyMetric(), n_epochs=2, print_every=1)
    trainer.train()

    # 测试结果
    from fastNLP import Tester

    tester = Tester(data_bundle.get_dataset('test'), model, batch_size=128, metrics=AccuracyMetric())
    tester.test()

输出结果::

    In Epoch:1/Step:12499, got best dev performance:
    AccuracyMetric: acc=0.9838
    Reloaded the best model.
    Evaluate data in 63.84 seconds!
    [tester]
    AccuracyMetric: acc=0.9815


2. 使用Bert进行命名实体识别
----------------------------------
命名实体识别是给定一句话，标记出其中的实体。一般序列标注的任务都使用conll格式，conll格式是至一行中通过制表符分隔不同的内容，使用空行分隔
两句话，例如下面的例子

.. code-block:: text

    中	B-ORG
    共	I-ORG
    中	I-ORG
    央	I-ORG
    致	O
    中	B-ORG
    国	I-ORG
    致	I-ORG
    公	I-ORG
    党	I-ORG
    十	I-ORG
    一	I-ORG
    大	I-ORG
    的	O
    贺	O
    词	O

这部分内容请参考 :doc:`快速实现序列标注模型 </tutorials/tutorial_9_seq_labeling>`


3. 使用Bert进行文本匹配
----------------------------------
文本匹配任务是指给定两句话判断他们的关系。比如，给定两句话判断前一句是否和后一句具有因果关系或是否是矛盾关系；或者给定两句话判断两句话是否
具有相同的意思。这里我们使用

.. code-block:: python

    data_bundle = CNXNLIBertPipe().process_from_file(paths)
    data_bundle.rename_field('chars', 'words')
    print(data_bundle)

    # 载入BertEmbedding
    from fastNLP.embeddings import BertEmbedding

    embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='cn-wwm', include_cls_sep=True)

    # 载入模型
    from fastNLP.models import BertForSentenceMatching

    model = BertForSentenceMatching(embed, len(data_bundle.get_vocab('target')))

    # 训练模型
    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Adam
    from fastNLP.core.optimizer import AdamW
    from fastNLP.core.callback import WarmupCallback

    callbacks = [WarmupCallback(warmup=0.1, schedule='linear'), ]

    trainer = Trainer(data_bundle.get_dataset('train'), model,
                      optimizer=AdamW(params=model.parameters(), lr=4e-5),
                      loss=CrossEntropyLoss(), device=0,
                      batch_size=8, dev_data=data_bundle.get_dataset('dev'),
                      metrics=AccuracyMetric(), n_epochs=5, print_every=1,
                      update_every=8, callbacks=callbacks)
    trainer.train()

    from fastNLP import Tester
    tester = Tester(data_bundle.get_dataset('test'), model, batch_size=8, metrics=AccuracyMetric())
    tester.test()

运行结果::

    In Epoch:3/Step:73632, got best dev performance:
    AccuracyMetric: acc=0.781928
    Reloaded the best model.
    Evaluate data in 18.54 seconds!
    [tester]
    AccuracyMetric: acc=0.783633


4. 使用Bert进行中文问答
----------------------------------
问答任务是给定一段内容，以及一个问题，需要从这段内容中找到答案。
例如
    "context": "锣鼓经是大陆传统器乐及戏曲里面常用的打击乐记谱方法，以中文字的声音模拟敲击乐的声音，纪录打击乐的各种不同的演奏方法。常
    用的节奏型称为「锣鼓点」。而锣鼓是戏曲节奏的支柱，除了加强演员身段动作的节奏感，也作为音乐的引子和尾声，提示音乐的板式和速度，以及
    作为唱腔和念白的伴奏，令诗句的韵律更加抑扬顿锉，段落分明。锣鼓的运用有约定俗成的程式，依照角色行当的身份、性格、情绪以及环境，配合
    相应的锣鼓点。锣鼓亦可以模仿大自然的音响效果，如雷电、波浪等等。戏曲锣鼓所运用的敲击乐器主要分为鼓、锣、钹和板四类型：鼓类包括有单
    皮鼓（板鼓）、大鼓、大堂鼓(唐鼓)、小堂鼓、怀鼓、花盆鼓等；锣类有大锣、小锣(手锣)、钲锣、筛锣、马锣、镗锣、云锣；钹类有铙钹、大
    钹、小钹、水钹、齐钹、镲钹、铰子、碰钟等；打拍子用的檀板、木鱼、梆子等。因为京剧的锣鼓通常由四位乐师负责，又称为四大件，领奏的师
    傅称为：「鼓佬」，其职责有如西方乐队的指挥，负责控制速度以及利用各种手势提示乐师演奏不同的锣鼓点。粤剧吸收了部份京剧的锣鼓，但以木鱼
    和沙的代替了京剧的板和鼓，作为打拍子的主要乐器。以下是京剧、昆剧和粤剧锣鼓中乐器对应的口诀用字：",
    "question": "锣鼓经是什么？",
    "answers": [
        {
          "text": "大陆传统器乐及戏曲里面常用的打击乐记谱方法",
          "answer_start": 4
        },
        {
          "text": "大陆传统器乐及戏曲里面常用的打击乐记谱方法",
          "answer_start": 4
        },
        {
          "text": "大陆传统器乐及戏曲里面常用的打击乐记谱方法",
          "answer_start": 4
        }
    ]

您可以通过以下的代码训练`CMRC2018 <https://github.com/ymcui/cmrc2018>`_

.. code-block:: python

    from fastNLP.embeddings import BertEmbedding
    from fastNLP.models import BertForQuestionAnswering
    from fastNLP.core.losses import CMRC2018Loss
    from fastNLP.core.metrics import CMRC2018Metric
    from fastNLP.io.pipe.qa import CMRC2018BertPipe
    from fastNLP import Trainer, BucketSampler
    from fastNLP import WarmupCallback, GradientClipCallback
    from fastNLP.core.optimizer import AdamW


    data_bundle = CMRC2018BertPipe().process_from_file()
    data_bundle.rename_field('chars', 'words')

    print(data_bundle)

    embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='cn', requires_grad=True, include_cls_sep=False, auto_truncate=True,
                          dropout=0.5, word_dropout=0.01)
    model = BertForQuestionAnswering(embed)
    loss = CMRC2018Loss()
    metric = CMRC2018Metric()

    wm_callback = WarmupCallback(schedule='linear')
    gc_callback = GradientClipCallback(clip_value=1, clip_type='norm')
    callbacks = [wm_callback, gc_callback]

    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer,
                      sampler=BucketSampler(seq_len_field_name='context_len'),
                      dev_data=data_bundle.get_dataset('dev'), metrics=metric,
                      callbacks=callbacks, device=0, batch_size=6, num_workers=2, n_epochs=2, print_every=1,
                      test_use_tqdm=False, update_every=10)
    trainer.train(load_best_model=False)

训练结果(和论文中报道的基本一致)::

    In Epoch:2/Step:1692, got best dev performance:
    CMRC2018Metric: f1=85.61, em=66.08


