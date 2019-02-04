
FastNLP 1分钟上手教程
=====================

教程原文见 https://github.com/fastnlp/fastNLP/blob/master/tutorials/fastnlp_1min_tutorial.ipynb

step 1
------

读取数据集

.. code:: ipython3

    from fastNLP import DataSet
    # linux_path = "../test/data_for_tests/tutorial_sample_dataset.csv"
    win_path = "C:\\Users\zyfeng\Desktop\FudanNLP\\fastNLP\\test\\data_for_tests\\tutorial_sample_dataset.csv"
    ds = DataSet.read_csv(win_path, headers=('raw_sentence', 'label'), sep='\t')

step 2
------

数据预处理 1. 类型转换 2. 切分验证集 3. 构建词典

.. code:: ipython3

    # 将所有数字转为小写
    ds.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
    # label转int
    ds.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)
    
    def split_sent(ins):
        return ins['raw_sentence'].split()
    ds.apply(split_sent, new_field_name='words', is_input=True)
    

.. code:: ipython3

    # 分割训练集/验证集
    train_data, dev_data = ds.split(0.3)
    print("Train size: ", len(train_data))
    print("Test size: ", len(dev_data))


.. parsed-literal::

    Train size:  54
    Test size:  23
    

.. code:: ipython3

    from fastNLP import Vocabulary
    vocab = Vocabulary(min_freq=2)
    train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
    
    # index句子, Vocabulary.to_index(word)
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
    dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
    

step 3
------

定义模型

.. code:: ipython3

    from fastNLP.models import CNNText
    model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)
    

step 4
------

开始训练

.. code:: ipython3

    from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
    trainer = Trainer(model=model, 
                      train_data=train_data, 
                      dev_data=dev_data,
                      loss=CrossEntropyLoss(),
                      metrics=AccuracyMetric()
                      )
    trainer.train()
    print('Train finished!')
    


.. parsed-literal::

    training epochs started 2018-12-07 14:03:41
    



.. parsed-literal::

    HBox(children=(IntProgress(value=0, layout=Layout(flex='2'), max=6), HTML(value='')), layout=Layout(display='i…



.. parsed-literal::

    Epoch 1/3. Step:2/6. AccuracyMetric: acc=0.26087
    Epoch 2/3. Step:4/6. AccuracyMetric: acc=0.347826
    Epoch 3/3. Step:6/6. AccuracyMetric: acc=0.608696
    Train finished!
    

本教程结束。更多操作请参考进阶教程。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
