==============================================================================
使用DataSetIter实现自定义训练过程
==============================================================================

我们使用前面介绍过的 :doc:`/tutorials/文本分类` 任务来进行详细的介绍。这里我们把数据集换成了SST2，使用 :class:`~fastNLP.DataSetIter` 类来编写自己的训练过程。
DataSetIter初探之前的内容与 :doc:`/tutorials/tutorial_5_loss_optimizer` 中的完全一样，如已经阅读过可以跳过。


数据读入和预处理
--------------------

数据读入
    我们可以使用 fastNLP  :mod:`fastNLP.io` 模块中的 :class:`~fastNLP.io.SST2Pipe` 类，轻松地读取以及预处理SST2数据集。:class:`~fastNLP.io.SST2Pipe` 对象的
    :meth:`~fastNLP.io.SST2Pipe.process_from_file` 方法能够对读入的SST2数据集进行数据的预处理，方法的参数为paths, 指要处理的文件所在目录，如果paths为None，则会自动下载数      据集，函数默认paths值为None。
    此函数返回一个 :class:`~fastNLP.io.DataBundle`，包含SST2数据集的训练集、测试集、验证集以及source端和target端的字典。其训练、测试、验证数据集含有四个     :mod:`~fastNLP.core.field` ：

    * raw_words: 原source句子
    * target: 标签值
    * words: index之后的raw_words
    * seq_len: 句子长度

    读入数据代码如下：

    .. code-block:: python

        from fastNLP.io import SST2Pipe
        
        pipe = SST2Pipe()
        databundle = pipe.process_from_file()
        vocab = databundle.vocabs['words']
        print(databundle)
        print(databundle.datasets['train'][0])
        print(databundle.vocabs['words'])


    输出数据如下::
	
        In total 3 datasets:
            test has 1821 instances.
            train has 67349 instances.
            dev has 872 instances.
        In total 2 vocabs:
            words has 16293 entries.
            target has 2 entries.

        +-------------------------------------------+--------+--------------------------------------+---------+
        |                 raw_words                 | target |                words                 | seq_len |
        +-------------------------------------------+--------+--------------------------------------+---------+
        | hide new secretions from the parental ... |   1    | [4111, 98, 12010, 38, 2, 6844, 9042] |    7    |
        +-------------------------------------------+--------+--------------------------------------+---------+
         
        Vocabulary(['hide', 'new', 'secretions', 'from', 'the']...)

    除了可以对数据进行读入的Pipe类，fastNLP还提供了读入和下载数据的Loader类，不同数据集的Pipe和Loader及其用法详见 :doc:`/tutorials/tutorial_4_load_dataset` 。
    
数据集分割
    由于SST2数据集的测试集并不带有标签数值，故我们分割出一部分训练集作为测试集。下面这段代码展示了 :meth:`~fastNLP.DataSet.split`  的使用方法，
    为了能让读者快速运行完整个教程，我们只取了训练集的前5000个数据。

    .. code-block:: python

        train_data = databundle.get_dataset('train')[:5000]
        train_data, test_data = train_data.split(0.015)
        dev_data = databundle.get_dataset('dev')
        print(len(train_data),len(dev_data),len(test_data))

    输出结果为::

        4925 872 75

数据集 :meth:`~fastNLP.DataSet.set_input` 和  :meth:`~fastNLP.DataSet.set_target` 函数
    :class:`~fastNLP.io.SST2Pipe`  类的 :meth:`~fastNLP.io.SST2Pipe.process_from_file` 方法在预处理过程中还将训练、测试、验证集
    的 `words` 、`seq_len` :mod:`~fastNLP.core.field` 设定为input，同时将`target` :mod:`~fastNLP.core.field` 设定为target。
    我们可以通过 :class:`~fastNLP.core.Dataset` 类的 :meth:`~fastNLP.core.Dataset.print_field_meta` 方法查看各个
    :mod:`~fastNLP.core.field` 的设定情况，代码如下：

    .. code-block:: python

        train_data.print_field_meta()

    输出结果为::
	
        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   | False  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    其中is_input和is_target分别表示是否为input和target。ignore_type为true时指使用  :class:`~fastNLP.DataSetIter` 取出batch数
    据时fastNLP不会进行自动padding，pad_value指对应 :mod:`~fastNLP.core.field` padding所用的值，这两者只有当
    :mod:`~fastNLP.core.field` 设定为input或者target的时候才有存在的意义。

    is_input为true的 :mod:`~fastNLP.core.field` 在 :class:`~fastNLP.DataSetIter` 迭代取出的 batch_x 中，
    而 is_target为true的 :mod:`~fastNLP.core.field` 在  :class:`~fastNLP.DataSetIter` 迭代取出的 batch_y 中。
    具体分析见下面DataSetIter的介绍过程。


评价指标
    训练模型需要提供一个评价指标。这里使用准确率做为评价指标。

    * ``pred`` 参数对应的是模型的 forward 方法返回的 dict 中的一个 key 的名字。
    * ``target`` 参数对应的是 :class:`~fastNLP.DataSet` 中作为标签的 :mod:`~fastNLP.core.field` 的名字。

    这里我们用 :class:`~fastNLP.Const` 来辅助命名，如果你自己编写模型中 forward 方法的返回值或
    数据集中 :mod:`~fastNLP.core.field` 的名字与本例不同， 你可以把 ``pred`` 参数和 ``target`` 参数设定符合自己代码的值。代码如下：

    .. code-block:: python

        from fastNLP import AccuracyMetric
        from fastNLP import Const
	
        # metrics=AccuracyMetric() 在本例中与下面这行代码等价
        metrics=AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)


DataSetIter初探
--------------------------

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

Padder
    在fastNLP里，pad是与一个 :mod:`~fastNLP.core.field` 绑定的。即不同的 :mod:`~fastNLP.core.field` 可以使用不同的pad方式，比如在英文任务中word需要的pad和
    character的pad方式往往是不同的。fastNLP是通过一个叫做 :class:`~fastNLP.Padder` 的子类来完成的。
    默认情况下，所有field使用 :class:`~fastNLP.AutoPadder`
    。大多数情况下直接使用 :class:`~fastNLP.AutoPadder` 就可以了。
    如果 :class:`~fastNLP.AutoPadder` 或 :class:`~fastNLP.EngChar2DPadder` 无法满足需求，
    也可以自己写一个 :class:`~fastNLP.Padder` 。

DataSetIter自动padding
    以下代码展示了DataSetIter的简单使用：

    .. code-block:: python

        from fastNLP import BucketSampler
        from fastNLP import DataSetIter

        tmp_data = dev_data[:10]
        # 定义一个Batch，传入DataSet，规定batch_size和去batch的规则。
        # 顺序（Sequential），随机（Random），相似长度组成一个batch（Bucket）
        sampler = BucketSampler(batch_size=2, seq_len_field_name='seq_len')
        batch = DataSetIter(batch_size=2, dataset=tmp_data, sampler=sampler)
        for batch_x, batch_y in batch:
            print("batch_x: ",batch_x)
            print("batch_y: ", batch_y)
    
    输出结果如下::

        batch_x:  {'words': tensor([[   13,   830,  7746,   174,     3,    47,     6,    83,  5752,    15,
                  2177,    15,    63,    57,   406,    84,  1009,  4973,    27,    17,
                 13785,     3,   533,  3687, 15623,    39,   375,     8, 15624,     8,
                  1323,  4398,     7],
                [ 1045, 11113,    16,   104,     5,     4,   176,  1824,  1704,     3,
                     2,    18,    11,     4,  1018,   432,   143,    33,   245,   308,
                     7,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0]]), 'seq_len': tensor([33, 21])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[  14,   10,    4,  311,    5,  154, 1418,  609,    7],
                [  14,   10,  437,   32,   78,    3,   78,  437,    7]]), 'seq_len': tensor([9, 9])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[    4,   277,   685,    18,     7],
                [15618,  3204,     5,  1675,     0]]), 'seq_len': tensor([5, 4])}
        batch_y:  {'target': tensor([1, 1])}
        batch_x:  {'words': tensor([[    2,   155,     3,  4426,     3,   239,     3,   739,     5,  1136,
                    41,    43,  2427,   736,     2,   648,    10, 15620,  2285,     7],
                [   24,    95,    28,    46,     8,   336,    38,   239,     8,  2133,
                     2,    18,    10, 15622,  1421,     6,    61,     5,   387,     7]]), 'seq_len': tensor([20, 20])}
        batch_y:  {'target': tensor([0, 0])}
        batch_x:  {'words': tensor([[  879,    96,     8,  1026,    12,  8067,    11, 13623,     8, 15619,
                     4,   673,   662,    15,     4,  1154,   240,   639,   417,     7],
                [   45,   752,   327,   180,    10, 15621,    16,    72,  8904,     9,
                  1217,     7,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([20, 12])}
        batch_y:  {'target': tensor([0, 1])}

    可以看到那些设定为input的 :mod:`~fastNLP.core.field` 都出现在batch_x中，而设定为target的 :mod:`~fastNLP.core.field` 则出现在batch_y中。同时对于同一个batch_x中的两个数据，长度偏短的那个会被自动padding到和长度偏长的句子长度一致，默认的padding值为0。

Dataset改变padding值
    可以通过 :meth:`~fastNLP.core.Dataset.set_pad_val` 方法修改默认的pad值，代码如下：

    .. code-block:: python

        tmp_data.set_pad_val('words',-1)
        batch = DataSetIter(batch_size=2, dataset=tmp_data, sampler=sampler)
        for batch_x, batch_y in batch:
            print("batch_x: ",batch_x)
            print("batch_y: ", batch_y)

    输出结果如下::

        batch_x:  {'words': tensor([[   13,   830,  7746,   174,     3,    47,     6,    83,  5752,    15,
                  2177,    15,    63,    57,   406,    84,  1009,  4973,    27,    17,
                 13785,     3,   533,  3687, 15623,    39,   375,     8, 15624,     8,
                  1323,  4398,     7],
                [ 1045, 11113,    16,   104,     5,     4,   176,  1824,  1704,     3,
                     2,    18,    11,     4,  1018,   432,   143,    33,   245,   308,
                     7,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
                    -1,    -1,    -1]]), 'seq_len': tensor([33, 21])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[  14,   10,    4,  311,    5,  154, 1418,  609,    7],
                [  14,   10,  437,   32,   78,    3,   78,  437,    7]]), 'seq_len': tensor([9, 9])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[    2,   155,     3,  4426,     3,   239,     3,   739,     5,  1136,
                    41,    43,  2427,   736,     2,   648,    10, 15620,  2285,     7],
                [   24,    95,    28,    46,     8,   336,    38,   239,     8,  2133,
                     2,    18,    10, 15622,  1421,     6,    61,     5,   387,     7]]), 'seq_len': tensor([20, 20])}
        batch_y:  {'target': tensor([0, 0])}
        batch_x:  {'words': tensor([[    4,   277,   685,    18,     7],
                [15618,  3204,     5,  1675,    -1]]), 'seq_len': tensor([5, 4])}
        batch_y:  {'target': tensor([1, 1])}
        batch_x:  {'words': tensor([[  879,    96,     8,  1026,    12,  8067,    11, 13623,     8, 15619,
                     4,   673,   662,    15,     4,  1154,   240,   639,   417,     7],
                [   45,   752,   327,   180,    10, 15621,    16,    72,  8904,     9,
                  1217,     7,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1]]), 'seq_len': tensor([20, 12])}
        batch_y:  {'target': tensor([0, 1])}
 
    可以看到使用了-1进行padding。

Dataset个性化padding
    如果我们希望对某一些 :mod:`~fastNLP.core.field` 进行个性化padding，可以自己构造Padder类，并使用 :meth:`~fastNLP.core.Dataset.set_padder` 函数修改padder来实现。下面通过构造一个将数据padding到固定长度的padder进行展示：

    .. code-block:: python

        from fastNLP.core.field import Padder
        import numpy as np
        class FixLengthPadder(Padder):
            def __init__(self, pad_val=0, length=None):
                super().__init__(pad_val=pad_val)
                self.length = length
                assert self.length is not None, "Creating FixLengthPadder with no specific length!"
        
            def __call__(self, contents, field_name, field_ele_dtype, dim):
                #计算当前contents中的最大长度
                max_len = max(map(len, contents))
                #如果当前contents中的最大长度大于指定的padder length的话就报错
                assert max_len <= self.length, "Fixed padder length smaller than actual length! with length {}".format(max_len)
                array = np.full((len(contents), self.length), self.pad_val, dtype=field_ele_dtype)
                for i, content_i in enumerate(contents):
                    array[i, :len(content_i)] = content_i
                return array

        #设定FixLengthPadder的固定长度为40
        tmp_padder = FixLengthPadder(pad_val=0,length=40)
        #利用dataset的set_padder函数设定words field的padder
        tmp_data.set_padder('words',tmp_padder)
        batch = DataSetIter(batch_size=2, dataset=tmp_data, sampler=sampler)
        for batch_x, batch_y in batch:
            print("batch_x: ",batch_x)
            print("batch_y: ", batch_y)

    输出结果如下::

        batch_x:  {'words': tensor([[   45,   752,   327,   180,    10, 15621,    16,    72,  8904,     9,
                  1217,     7,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [  879,    96,     8,  1026,    12,  8067,    11, 13623,     8, 15619,
                     4,   673,   662,    15,     4,  1154,   240,   639,   417,     7,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([12, 20])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[   13,   830,  7746,   174,     3,    47,     6,    83,  5752,    15,
                  2177,    15,    63,    57,   406,    84,  1009,  4973,    27,    17,
                 13785,     3,   533,  3687, 15623,    39,   375,     8, 15624,     8,
                  1323,  4398,     7,     0,     0,     0,     0,     0,     0,     0],
                [ 1045, 11113,    16,   104,     5,     4,   176,  1824,  1704,     3,
                     2,    18,    11,     4,  1018,   432,   143,    33,   245,   308,
                     7,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([33, 21])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[  14,   10,    4,  311,    5,  154, 1418,  609,    7,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0],
                [  14,   10,  437,   32,   78,    3,   78,  437,    7,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0]]), 'seq_len': tensor([9, 9])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[    2,   155,     3,  4426,     3,   239,     3,   739,     5,  1136,
                    41,    43,  2427,   736,     2,   648,    10, 15620,  2285,     7,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [   24,    95,    28,    46,     8,   336,    38,   239,     8,  2133,
                     2,    18,    10, 15622,  1421,     6,    61,     5,   387,     7,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([20, 20])}
        batch_y:  {'target': tensor([0, 0])}
        batch_x:  {'words': tensor([[    4,   277,   685,    18,     7,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [15618,  3204,     5,  1675,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([5, 4])}
        batch_y:  {'target': tensor([1, 1])}

    在这里所有的 `words` 都被pad成了长度为40的list。


使用DataSetIter自己编写训练过程
------------------------------------
    如果你想用类似 PyTorch 的使用方法，自己编写训练过程，可以参考下面这段代码。
    其中使用了 fastNLP 提供的 :class:`~fastNLP.DataSetIter` 来获得小批量训练的小批量数据，
    使用 :class:`~fastNLP.BucketSampler` 做为  :class:`~fastNLP.DataSetIter` 的参数来选择采样的方式。

    以下代码使用BucketSampler作为 :class:`~fastNLP.DataSetIter` 初始化的输入，运用 :class:`~fastNLP.DataSetIter` 自己写训练程序

    .. code-block:: python

        from fastNLP import BucketSampler
        from fastNLP import DataSetIter
        from fastNLP.models import CNNText
        from fastNLP import Tester
        import torch
        import time

        embed_dim = 100
        model = CNNText((len(vocab),embed_dim), num_classes=2, dropout=0.1)

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
                print(tester_tmp._format_eval_results(res),end=" ")
                print('{:d}ms'.format(round((time.time()-start_time)*1000)))
                loss_list.clear()

        train(10, train_data, dev_data)
        #使用tester进行快速测试
        tester = Tester(test_data, model, metrics=AccuracyMetric())
        tester.test()

    这段代码的输出如下::

        -----start training-----

        Evaluate data in 2.68 seconds!
        Epoch 0 Avg Loss: 0.66 AccuracyMetric: acc=0.708716 29307ms

        Evaluate data in 0.38 seconds!
        Epoch 1 Avg Loss: 0.41 AccuracyMetric: acc=0.770642 52200ms

        Evaluate data in 0.51 seconds!
        Epoch 2 Avg Loss: 0.16 AccuracyMetric: acc=0.747706 70268ms

        Evaluate data in 0.96 seconds!
        Epoch 3 Avg Loss: 0.06 AccuracyMetric: acc=0.741972 90349ms

        Evaluate data in 1.04 seconds!
        Epoch 4 Avg Loss: 0.03 AccuracyMetric: acc=0.740826 114250ms

        Evaluate data in 0.8 seconds!
        Epoch 5 Avg Loss: 0.02 AccuracyMetric: acc=0.738532 134742ms

        Evaluate data in 0.65 seconds!
        Epoch 6 Avg Loss: 0.01 AccuracyMetric: acc=0.731651 154503ms

        Evaluate data in 0.8 seconds!
        Epoch 7 Avg Loss: 0.01 AccuracyMetric: acc=0.738532 175397ms

        Evaluate data in 0.36 seconds!
        Epoch 8 Avg Loss: 0.01 AccuracyMetric: acc=0.733945 192384ms

        Evaluate data in 0.84 seconds!
        Epoch 9 Avg Loss: 0.01 AccuracyMetric: acc=0.744266 214417ms

        Evaluate data in 0.04 seconds!
        [tester]
        AccuracyMetric: acc=0.786667


