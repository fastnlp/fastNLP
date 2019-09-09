==============================================================================
动手实现一个文本分类器II-使用DataSetIter实现自定义训练过程
==============================================================================

我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。给出一段评价性文字，预测其情感倾向是积极（label=1）、
消极（label=0）还是中性（label=2），使用 :class:`~fastNLP.DataSetIter` 类来编写自己的训练过程。
自己编写训练过程之前的内容与 :doc:`/tutorials/tutorial_4_loss_optimizer` 中的完全一样，如已经阅读过可以跳过。

--------------
数据处理
--------------

数据读入
    我们可以使用 fastNLP  :mod:`fastNLP.io` 模块中的 :class:`~fastNLP.io.SSTLoader` 类，轻松地读取SST数据集（数据来源：https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip）。
    这里的 dataset 是 fastNLP 中 :class:`~fastNLP.DataSet` 类的对象。

    .. code-block:: python

        from fastNLP.io import SSTLoader

        loader = SSTLoader()
        #这里的all.txt是下载好数据后train.txt、dev.txt、test.txt的组合
        #loader.load(path)会首先判断path是否为none，若是则自动从网站下载数据，若不是则读入数据并返回databundle
        databundle_ = loader.load("./trainDevTestTrees_PTB/trees/all.txt")
        dataset = databundle_.datasets['train']
        print(dataset[0])

    输出数据如下::
	
        {'words': ['It', "'s", 'a', 'lovely', 'film', 'with', 'lovely', 'performances', 'by', 'Buy', 'and', 'Accorsi', '.'] type=list,
        'target': positive type=str}
		
    除了读取数据外，fastNLP 还提供了读取其它文件类型的 Loader 类、读取 Embedding的 Loader 等。详见 :doc:`/fastNLP.io` 。
    

数据处理
    可以使用事先定义的 :class:`~fastNLP.io.SSTPipe` 类对数据进行基本预处理，这里我们手动进行处理。
    我们使用 :class:`~fastNLP.DataSet` 类的 :meth:`~fastNLP.DataSet.apply` 方法将 ``target`` :mod:`~fastNLP.core.field` 转化为整数。
    
    .. code-block:: python

        def label_to_int(x):
            if x['target']=="positive":
                return 1
            elif x['target']=="negative":
                return 0
            else:
                return 2

        # 将label转为整数
        dataset.apply(lambda x: label_to_int(x), new_field_name='target')

    ``words`` 和 ``target`` 已经足够用于 :class:`~fastNLP.models.CNNText` 的训练了，但我们从其文档
    :class:`~fastNLP.models.CNNText` 中看到，在 :meth:`~fastNLP.models.CNNText.forward` 的时候，还可以传入可选参数 ``seq_len`` 。
    所以，我们再使用 :meth:`~fastNLP.DataSet.apply_field` 方法增加一个名为 ``seq_len`` 的 :mod:`~fastNLP.core.field` 。

    .. code-block:: python

        # 增加长度信息
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
    我们再用 :class:`~fastNLP.Vocabulary` 类来统计数据中出现的单词，并使用 :meth:`~fastNLP.Vocabulary.index_dataset`
    将单词序列转化为训练可用的数字序列。

    .. code-block:: python

        from fastNLP import Vocabulary

        # 使用Vocabulary类统计单词，并将单词序列转化为数字序列
        vocab = Vocabulary(min_freq=2).from_dataset(dataset, field_name='words')
        vocab.index_dataset(dataset, field_name='words',new_field_name='words')
        print(dataset[0])
    
    输出数据如下::
	
        {'words': [27, 9, 6, 913, 16, 18, 913, 124, 31, 5715, 5, 1, 2] type=list,
        'target': 1 type=int,
        'seq_len': 13 type=int}


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

        print(Const.INPUT)
        print(Const.INPUT_LEN)
        print(Const.TARGET)
        print(Const.OUTPUT)
    
    输出结果为::
	
        words
        seq_len
        target
        pred
    
    在给 :class:`~fastNLP.DataSet` 中 :mod:`~fastNLP.core.field` 改名后，我们还需要设置训练所需的输入和目标，这里使用的是
    :meth:`~fastNLP.DataSet.set_input` 和 :meth:`~fastNLP.DataSet.set_target` 两个函数。

    .. code-block:: python

        #使用dataset的 set_input 和 set_target函数，告诉模型dataset中那些数据是输入，那些数据是标签（目标输出）
        dataset.set_input(Const.INPUT, Const.INPUT_LEN)
        dataset.set_target(Const.TARGET)

数据集分割
    除了修改 :mod:`~fastNLP.core.field` 之外，我们还可以对 :class:`~fastNLP.DataSet` 进行分割，以供训练、开发和测试使用。
    下面这段代码展示了 :meth:`~fastNLP.DataSet.split` 的使用方法

    .. code-block:: python

        train_dev_data, test_data = dataset.split(0.1)
        train_data, dev_data = train_dev_data.split(0.1)
        print(len(train_data), len(dev_data), len(test_data))

    输出结果为::
	
        9603 1067 1185

评价指标
    训练模型需要提供一个评价指标。这里使用准确率做为评价指标。参数的 `命名规则` 跟上面类似。
    ``pred`` 参数对应的是模型的 forward 方法返回的 dict 中的一个 key 的名字。
    ``target`` 参数对应的是 :class:`~fastNLP.DataSet` 中作为标签的 :mod:`~fastNLP.core.field` 的名字。

    .. code-block:: python

        from fastNLP import AccuracyMetric
	
        # metrics=AccuracyMetric() 在本例中与下面这行代码等价
        metrics=AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)


--------------------------
自己编写训练过程
--------------------------
    如果你想用类似 PyTorch 的使用方法，自己编写训练过程，你可以参考下面这段代码。
    其中使用了 fastNLP 提供的 :class:`~fastNLP.DataSetIter` 来获得小批量训练的小批量数据，
    使用 :class:`~fastNLP.BucketSampler` 做为  :class:`~fastNLP.DataSetIter` 的参数来选择采样的方式。
    
DataSetIter
    fastNLP定义的 :class:`~fastNLP.DataSetIter` 类，用于定义一个batch，并实现batch的多种功能，在初始化时传入的参数有：
	
    * dataset: :class:`~fastNLP.DataSet` 对象, 数据集
    * batch_size: 取出的batch大小
    * sampler: 规定使用的 :class:`~fastNLP.Sampler` 若为 None, 使用 :class:`~fastNLP.RandomSampler` （Default: None）
    * as_numpy: 若为 True, 输出batch为 `numpy.array`. 否则为 `torch.Tensor` （Default: False）
    * prefetch: 若为 True使用多进程预先取出下一batch. （Default: False）

sampler
    fastNLP 实现的采样器有：
	
    * :class:`~fastNLP.BucketSampler` 可以随机地取出长度相似的元素 【初始化参数:  num_buckets：bucket的数量；  batch_size：batch大小；  seq_len_field_name：dataset中对应序列长度的 :mod:`~fastNLP.core.field` 的名字】
    * SequentialSampler： 顺序取出元素的采样器【无初始化参数】
    * RandomSampler：随机化取元素的采样器【无初始化参数】

    以下代码使用BucketSampler作为 :class:`~fastNLP.DataSetIter` 初始化的输入，运用 :class:`~fastNLP.DataSetIter` 自己写训练程序

    .. code-block:: python

        from fastNLP import BucketSampler
        from fastNLP import DataSetIter
        from fastNLP.models import CNNText
        from fastNLP import Tester
        import torch
        import time

        embed_dim = 100
        model = CNNText((len(vocab),embed_dim), num_classes=3, dropout=0.1)

        def train(epoch, data, devdata):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            lossfunc = torch.nn.CrossEntropyLoss()
            batch_size = 32

            # 定义一个Batch，传入DataSet，规定batch_size和去batch的规则。
            # 顺序（Sequential），随机（Random），相似长度组成一个batch（Bucket）
            train_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
            train_batch = DataSetIter(batch_size=batch_size, dataset=data, sampler=train_sampler)

            start_time = time.time()
            print("-"*5+"start training"+"-"*5)
            for i in range(epoch):
                loss_list = []
                for batch_x, batch_y in train_batch:
                    optimizer.zero_grad()
                    output = model(batch_x['words'])
                    loss = lossfunc(output['pred'], batch_y['target'])
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())

                #这里verbose如果为0，在调用Tester对象的test()函数时不输出任何信息，返回评估信息; 如果为1，打印出验证结果，返回评估信息
                #在调用过Tester对象的test()函数后，调用其_format_eval_results(res)函数，结构化输出验证结果
                tester_tmp = Tester(devdata, model, metrics=AccuracyMetric(), verbose=0)
                res=tester_tmp.test()

                print('Epoch {:d} Avg Loss: {:.2f}'.format(i, sum(loss_list) / len(loss_list)),end=" ")
                print(tester._format_eval_results(res),end=" ")
                print('{:d}ms'.format(round((time.time()-start_time)*1000)))
                loss_list.clear()

        train(10, train_data, dev_data)
        #使用tester进行快速测试
        tester = Tester(test_data, model, metrics=AccuracyMetric())
        tester.test()

    这段代码的输出如下::

        -----start training-----
        Epoch 0 Avg Loss: 1.09 AccuracyMetric: acc=0.480787 58989ms
        Epoch 1 Avg Loss: 1.00 AccuracyMetric: acc=0.500469 118348ms
        Epoch 2 Avg Loss: 0.93 AccuracyMetric: acc=0.536082 176220ms
        Epoch 3 Avg Loss: 0.87 AccuracyMetric: acc=0.556701 236032ms
        Epoch 4 Avg Loss: 0.78 AccuracyMetric: acc=0.562324 294351ms
        Epoch 5 Avg Loss: 0.69 AccuracyMetric: acc=0.58388 353673ms
        Epoch 6 Avg Loss: 0.60 AccuracyMetric: acc=0.574508 412106ms
        Epoch 7 Avg Loss: 0.51 AccuracyMetric: acc=0.589503 471097ms
        Epoch 8 Avg Loss: 0.44 AccuracyMetric: acc=0.581068 529174ms
        Epoch 9 Avg Loss: 0.39 AccuracyMetric: acc=0.572634 586216ms
        [tester]
        AccuracyMetric: acc=0.527426


