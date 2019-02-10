fastNLP 10分钟上手教程
===============

教程原文见 https://github.com/fastnlp/fastNLP/blob/master/tutorials/fastnlp_10min_tutorial.ipynb

fastNLP提供方便的数据预处理，训练和测试模型的功能

DataSet & Instance
------------------

fastNLP用DataSet和Instance保存和处理数据。每个DataSet表示一个数据集，每个Instance表示一个数据样本。一个DataSet存有多个Instance，每个Instance可以自定义存哪些内容。

有一些read\_\*方法，可以轻松从文件读取数据，存成DataSet。

.. code:: ipython3

    from fastNLP import DataSet
    from fastNLP import Instance
    
    # 从csv读取数据到DataSet
    win_path = "C:\\Users\zyfeng\Desktop\FudanNLP\\fastNLP\\test\\data_for_tests\\tutorial_sample_dataset.csv"
    dataset = DataSet.read_csv(win_path, headers=('raw_sentence', 'label'), sep='\t')
    print(dataset[0])


.. parsed-literal::

    {'raw_sentence': A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .,
    'label': 1}
    

.. code:: ipython3

    # DataSet.append(Instance)加入新数据
    
    dataset.append(Instance(raw_sentence='fake data', label='0'))
    dataset[-1]




.. parsed-literal::

    {'raw_sentence': fake data,
    'label': 0}



.. code:: ipython3

    # DataSet.apply(func, new_field_name)对数据预处理
    
    # 将所有数字转为小写
    dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
    # label转int
    dataset.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)
    # 使用空格分割句子
    dataset.drop(lambda x: len(x['raw_sentence'].split()) == 0)
    def split_sent(ins):
        return ins['raw_sentence'].split()
    dataset.apply(split_sent, new_field_name='words', is_input=True)

.. code:: ipython3

    # DataSet.drop(func)筛除数据
    # 删除低于某个长度的词语
    dataset.drop(lambda x: len(x['words']) <= 3)

.. code:: ipython3

    # 分出测试集、训练集
    
    test_data, train_data = dataset.split(0.3)
    print("Train size: ", len(test_data))
    print("Test size: ", len(train_data))


.. parsed-literal::

    Train size:  54
    Test size: 

Vocabulary
----------

fastNLP中的Vocabulary轻松构建词表，将词转成数字

.. code:: ipython3

    from fastNLP import Vocabulary
    
    # 构建词表, Vocabulary.add(word)
    vocab = Vocabulary(min_freq=2)
    train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
    vocab.build_vocab()
    
    # index句子, Vocabulary.to_index(word)
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
    test_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
    
    
    print(test_data[0])


.. parsed-literal::

    {'raw_sentence': the plot is romantic comedy boilerplate from start to finish .,
    'label': 2,
    'label_seq': 2,
    'words': ['the', 'plot', 'is', 'romantic', 'comedy', 'boilerplate', 'from', 'start', 'to', 'finish', '.'],
    'word_seq': [2, 13, 9, 24, 25, 26, 15, 27, 11, 28, 3]}
    

.. code:: ipython3

    # 假设你们需要做强化学习或者gan之类的项目，也许你们可以使用这里的dataset
    from fastNLP.core.batch import Batch
    from fastNLP.core.sampler import RandomSampler
    
    batch_iterator = Batch(dataset=train_data, batch_size=2, sampler=RandomSampler())
    for batch_x, batch_y in batch_iterator:
        print("batch_x has: ", batch_x)
        print("batch_y has: ", batch_y)
        break


.. parsed-literal::

    batch_x has:  {'words': array([list(['this', 'kind', 'of', 'hands-on', 'storytelling', 'is', 'ultimately', 'what', 'makes', 'shanghai', 'ghetto', 'move', 'beyond', 'a', 'good', ',', 'dry', ',', 'reliable', 'textbook', 'and', 'what', 'allows', 'it', 'to', 'rank', 'with', 'its', 'worthy', 'predecessors', '.']),
           list(['the', 'entire', 'movie', 'is', 'filled', 'with', 'deja', 'vu', 'moments', '.'])],
          dtype=object), 'word_seq': tensor([[  19,  184,    6,    1,  481,    9,  206,   50,   91, 1210, 1609, 1330,
              495,    5,   63,    4, 1269,    4,    1, 1184,    7,   50, 1050,   10,
                8, 1611,   16,   21, 1039,    1,    2],
            [   3,  711,   22,    9, 1282,   16, 2482, 2483,  200,    2,    0,    0,
                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0,    0]])}
    batch_y has:  {'label_seq': tensor([3, 2])}
    

Model
-----

.. code:: ipython3

    # 定义一个简单的Pytorch模型
    
    from fastNLP.models import CNNText
    model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)
    model




.. parsed-literal::

    CNNText(
      (embed): Embedding(
        (embed): Embedding(77, 50, padding_idx=0)
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
      (fc): Linear(
        (linear): Linear(in_features=12, out_features=5, bias=True)
      )
    )



Trainer & Tester
----------------

使用fastNLP的Trainer训练模型

.. code:: ipython3

    from fastNLP import Trainer
    from copy import deepcopy
    from fastNLP import CrossEntropyLoss
    from fastNLP import AccuracyMetric

.. code:: ipython3

    # 进行overfitting测试
    copy_model = deepcopy(model)
    overfit_trainer = Trainer(model=copy_model, 
                              train_data=test_data, 
                              dev_data=test_data,
                              loss=CrossEntropyLoss(pred="output", target="label_seq"),
                              metrics=AccuracyMetric(),
                              n_epochs=10,
                              save_path=None)
    overfit_trainer.train()


.. parsed-literal::

    training epochs started 2018-12-07 14:07:20
    



.. parsed-literal::

    HBox(children=(IntProgress(value=0, layout=Layout(flex='2'), max=20), HTML(value='')), layout=Layout(display='…



.. parsed-literal::

    Epoch 1/10. Step:2/20. AccuracyMetric: acc=0.037037
    Epoch 2/10. Step:4/20. AccuracyMetric: acc=0.296296
    Epoch 3/10. Step:6/20. AccuracyMetric: acc=0.333333
    Epoch 4/10. Step:8/20. AccuracyMetric: acc=0.555556
    Epoch 5/10. Step:10/20. AccuracyMetric: acc=0.611111
    Epoch 6/10. Step:12/20. AccuracyMetric: acc=0.481481
    Epoch 7/10. Step:14/20. AccuracyMetric: acc=0.62963
    Epoch 8/10. Step:16/20. AccuracyMetric: acc=0.685185
    Epoch 9/10. Step:18/20. AccuracyMetric: acc=0.722222
    Epoch 10/10. Step:20/20. AccuracyMetric: acc=0.777778
    

.. code:: ipython3

    # 实例化Trainer，传入模型和数据，进行训练
    trainer = Trainer(model=model, 
                      train_data=train_data, 
                      dev_data=test_data,
                      loss=CrossEntropyLoss(pred="output", target="label_seq"),
                      metrics=AccuracyMetric(),
                      n_epochs=5)
    trainer.train()
    print('Train finished!')


.. parsed-literal::

    training epochs started 2018-12-07 14:08:10
    



.. parsed-literal::

    HBox(children=(IntProgress(value=0, layout=Layout(flex='2'), max=5), HTML(value='')), layout=Layout(display='i…



.. parsed-literal::

    Epoch 1/5. Step:1/5. AccuracyMetric: acc=0.037037
    Epoch 2/5. Step:2/5. AccuracyMetric: acc=0.037037
    Epoch 3/5. Step:3/5. AccuracyMetric: acc=0.037037
    Epoch 4/5. Step:4/5. AccuracyMetric: acc=0.185185
    Epoch 5/5. Step:5/5. AccuracyMetric: acc=0.240741
    Train finished!
    

.. code:: ipython3

    from fastNLP import Tester
    
    tester = Tester(data=test_data, model=model, metrics=AccuracyMetric())
    acc = tester.test()


.. parsed-literal::

    [tester] 
    AccuracyMetric: acc=0.240741
    

In summary
----------

fastNLP Trainer的伪代码逻辑
---------------------------

1. 准备DataSet，假设DataSet中共有如下的fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ['raw_sentence', 'word_seq1', 'word_seq2', 'raw_label','label']
    通过
        DataSet.set_input('word_seq1', word_seq2', flag=True)将'word_seq1', 'word_seq2'设置为input
    通过
        DataSet.set_target('label', flag=True)将'label'设置为target

2. 初始化模型
~~~~~~~~~~~~~

::

    class Model(nn.Module):
        def __init__(self):
            xxx
        def forward(self, word_seq1, word_seq2):
            # (1) 这里使用的形参名必须和DataSet中的input field的名称对应。因为我们是通过形参名, 进行赋值的
            # (2) input field的数量可以多于这里的形参数量。但是不能少于。
            xxxx
            # 输出必须是一个dict

3. Trainer的训练过程
~~~~~~~~~~~~~~~~~~~~

::

    (1) 从DataSet中按照batch_size取出一个batch，调用Model.forward
    (2) 将 Model.forward的结果 与 标记为target的field 传入Losser当中。
           由于每个人写的Model.forward的output的dict可能key并不一样，比如有人是{'pred':xxx}, {'output': xxx}; 
           另外每个人将target可能也会设置为不同的名称, 比如有人是label, 有人设置为target；
        为了解决以上的问题，我们的loss提供映射机制
           比如CrossEntropyLosser的需要的输入是(prediction, target)。但是forward的output是{'output': xxx}; 'label'是target
           那么初始化losser的时候写为CrossEntropyLosser(prediction='output', target='label')即可
     (3) 对于Metric是同理的
         Metric计算也是从 forward的结果中取值 与 设置target的field中取值。 也是可以通过映射找到对应的值        

一些问题.
---------

1. DataSet中为什么需要设置input和target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    只有被设置为input或者target的数据才会在train的过程中被取出来
    (1.1) 我们只会在设置为input的field中寻找传递给Model.forward的参数。
    (1.2) 我们在传递值给losser或者metric的时候会使用来自: 
            (a)Model.forward的output
            (b)被设置为target的field
          

2. 我们是通过forwad中的形参名将DataSet中的field赋值给对应的参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     (1.1) 构建模型过程中，
      例如:
          DataSet中x，seq_lens是input，那么forward就应该是
          def forward(self, x, seq_lens):
              pass
          我们是通过形参名称进行匹配的field的
       

1. 加载数据到DataSet
~~~~~~~~~~~~~~~~~~~~

2. 使用apply操作对DataSet进行预处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

      (2.1) 处理过程中将某些field设置为input，某些field设置为target

3. 构建模型
~~~~~~~~~~~

::

      (3.1) 构建模型过程中，需要注意forward函数的形参名需要和DataSet中设置为input的field名称是一致的。
      例如:
          DataSet中x，seq_lens是input，那么forward就应该是
          def forward(self, x, seq_lens):
              pass
          我们是通过形参名称进行匹配的field的
      (3.2) 模型的forward的output需要是dict类型的。
          建议将输出设置为{"pred": xx}.
          
