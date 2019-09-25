==============================================================================
动手实现一个文本分类器II-使用DataSetIter实现自定义训练过程
==============================================================================

我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。给出一段评价性文字，预测其情感倾向是积极的（label=0）、
还是消极的（label=1），使用 :class:`~fastNLP.DataSetIter` 类来编写自己的训练过程。  
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
    由于SST2数据集的测试集并不带有标签数值，故我们分割出一部分训练集作为测试集。下面这段代码展示了 :meth:`~fastNLP.DataSet.split`  的使用方法

    .. code-block:: python

        train_data = databundle.get_dataset('train')
        train_data, test_data = train_data.split(0.015)
        dev_data = databundle.get_dataset('dev')
        print(len(train_data),len(dev_data),len(test_data))

    输出结果为::
	
        66339 872 1010

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

        batch_x:  {'words': tensor([[    4,   278,   686,    18,     7],
                [15619,  3205,     5,  1676,     0]]), 'seq_len': tensor([5, 4])}
        batch_y:  {'target': tensor([1, 1])}
        batch_x:  {'words': tensor([[   44,   753,   328,   181,    10, 15622,    16,    71,  8905,     9,
                  1218,     7,     0,     0,     0,     0,     0,     0,     0,     0],
                [  880,    97,     8,  1027,    12,  8068,    11, 13624,     8, 15620,
                     4,   674,   663,    15,     4,  1155,   241,   640,   418,     7]]), 'seq_len': tensor([12, 20])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[ 1046, 11114,    16,   105,     5,     4,   177,  1825,  1705,     3,
                     2,    18,    11,     4,  1019,   433,   144,    32,   246,   309,
                     7,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0],
                [   13,   831,  7747,   175,     3,    46,     6,    84,  5753,    15,
                  2178,    15,    62,    56,   407,    85,  1010,  4974,    26,    17,
                 13786,     3,   534,  3688, 15624,    38,   376,     8, 15625,     8,
                 1324,  4399,     7]]), 'seq_len': tensor([21, 33])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[  14,   10,  438,   31,   78,    3,   78,  438,    7],
                [  14,   10,    4,  312,    5,  155, 1419,  610,    7]]), 'seq_len': tensor([9, 9])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[   24,    96,    27,    45,     8,   337,    37,   240,     8,  2134,
                     2,    18,    10, 15623,  1422,     6,    60,     5,   388,     7],
                [    2,   156,     3,  4427,     3,   240,     3,   740,     5,  1137,
                    40,    42,  2428,   737,     2,   649,    10, 15621,  2286,     7]]), 'seq_len': tensor([20, 20])}
        batch_y:  {'target': tensor([0, 0])}

    可以看到那些设定为input的 :mod:`~fastNLP.core.field` 都出现在batch_x中，而设定为target的 :mod:`~fastNLP.core.field` 则出现在batch_y中。同时对于同一个batch_x中的两个数    据，长度偏短的那个会被自动padding到和长度偏长的句子长度一致，默认的padding值为0。

Dataset改变padding值
    可以通过 :meth:`~fastNLP.core.Dataset.set_pad_val` 方法修改默认的pad值，代码如下：

    .. code-block:: python

        tmp_data.set_pad_val('words',-1)
        batch = DataSetIter(batch_size=2, dataset=tmp_data, sampler=sampler)
        for batch_x, batch_y in batch:
            print("batch_x: ",batch_x)
            print("batch_y: ", batch_y)

    输出结果如下::

        batch_x:  {'words': tensor([[15619,  3205,     5,  1676,    -1],
                [    4,   278,   686,    18,     7]]), 'seq_len': tensor([4, 5])}
        batch_y:  {'target': tensor([1, 1])}
        batch_x:  {'words': tensor([[ 1046, 11114,    16,   105,     5,     4,   177,  1825,  1705,     3,
                     2,    18,    11,     4,  1019,   433,   144,    32,   246,   309,
                     7,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
                    -1,    -1,    -1],
                [   13,   831,  7747,   175,     3,    46,     6,    84,  5753,    15,
                  2178,    15,    62,    56,   407,    85,  1010,  4974,    26,    17,
                 13786,     3,   534,  3688, 15624,    38,   376,     8, 15625,     8,
                  1324,  4399,     7]]), 'seq_len': tensor([21, 33])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[  14,   10,    4,  312,    5,  155, 1419,  610,    7],
                [  14,   10,  438,   31,   78,    3,   78,  438,    7]]), 'seq_len': tensor([9, 9])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[    2,   156,     3,  4427,     3,   240,     3,   740,     5,  1137,
                    40,    42,  2428,   737,     2,   649,    10, 15621,  2286,     7],
                [   24,    96,    27,    45,     8,   337,    37,   240,     8,  2134,
                     2,    18,    10, 15623,  1422,     6,    60,     5,   388,     7]]), 'seq_len': tensor([20, 20])}
        batch_y:  {'target': tensor([0, 0])}
        batch_x:  {'words': tensor([[   44,   753,   328,   181,    10, 15622,    16,    71,  8905,     9,
                  1218,     7,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1],
                [  880,    97,     8,  1027,    12,  8068,    11, 13624,     8, 15620,
                     4,   674,   663,    15,     4,  1155,   241,   640,   418,     7]]), 'seq_len': tensor([12, 20])}
        batch_y:  {'target': tensor([1, 0])}
 
    可以看到使用了-1进行padding。

Dataset个性化padding
    如果我们希望对某一些 :mod:`~fastNLP.core.field` 进行个性化padding，可以自己构造Padder类，并使用 :meth:`~fastNLP.core.Dataset.set_padder` 函数修改padder来实现。下面通   过构造一个将数据padding到固定长度的padder进行展示：

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

        batch_x:  {'words': tensor([[    4,   278,   686,    18,     7,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [15619,  3205,     5,  1676,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([5, 4])}
        batch_y:  {'target': tensor([1, 1])}
        batch_x:  {'words': tensor([[    2,   156,     3,  4427,     3,   240,     3,   740,     5,  1137,
                    40,    42,  2428,   737,     2,   649,    10, 15621,  2286,     7,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [   24,    96,    27,    45,     8,   337,    37,   240,     8,  2134,
                     2,    18,    10, 15623,  1422,     6,    60,     5,   388,     7,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([20, 20])}
        batch_y:  {'target': tensor([0, 0])}
        batch_x:  {'words': tensor([[   13,   831,  7747,   175,     3,    46,     6,    84,  5753,    15,
                  2178,    15,    62,    56,   407,    85,  1010,  4974,    26,    17,
                 13786,     3,   534,  3688, 15624,    38,   376,     8, 15625,     8,
                  1324,  4399,     7,     0,     0,     0,     0,     0,     0,     0],
                [ 1046, 11114,    16,   105,     5,     4,   177,  1825,  1705,     3,
                     2,    18,    11,     4,  1019,   433,   144,    32,   246,   309,
                     7,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([33, 21])}
        batch_y:  {'target': tensor([1, 0])}
        batch_x:  {'words': tensor([[  14,   10,    4,  312,    5,  155, 1419,  610,    7,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0],
                [  14,   10,  438,   31,   78,    3,   78,  438,    7,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0]]), 'seq_len': tensor([9, 9])}
        batch_y:  {'target': tensor([0, 1])}
        batch_x:  {'words': tensor([[   44,   753,   328,   181,    10, 15622,    16,    71,  8905,     9,
                  1218,     7,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                [  880,    97,     8,  1027,    12,  8068,    11, 13624,     8, 15620,
                     4,   674,   663,    15,     4,  1155,   241,   640,   418,     7,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]]), 'seq_len': tensor([12, 20])}
        batch_y:  {'target': tensor([1, 0])}

    在这里所有的`words`都被pad成了长度为40的list。


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

        Evaluate data in 0.2 seconds!
        Epoch 0 Avg Loss: 0.33 AccuracyMetric: acc=0.825688 48895ms

        Evaluate data in 0.19 seconds!
        Epoch 1 Avg Loss: 0.16 AccuracyMetric: acc=0.829128 102081ms

        Evaluate data in 0.18 seconds!
        Epoch 2 Avg Loss: 0.10 AccuracyMetric: acc=0.822248 152853ms

        Evaluate data in 0.17 seconds!
        Epoch 3 Avg Loss: 0.08 AccuracyMetric: acc=0.821101 200184ms

        Evaluate data in 0.17 seconds!
        Epoch 4 Avg Loss: 0.06 AccuracyMetric: acc=0.827982 253097ms

        Evaluate data in 0.27 seconds!
        Epoch 5 Avg Loss: 0.05 AccuracyMetric: acc=0.806193 303883ms

        Evaluate data in 0.26 seconds!
        Epoch 6 Avg Loss: 0.04 AccuracyMetric: acc=0.803899 392315ms

        Evaluate data in 0.36 seconds!
        Epoch 7 Avg Loss: 0.04 AccuracyMetric: acc=0.802752 527211ms

        Evaluate data in 0.15 seconds!
        Epoch 8 Avg Loss: 0.03 AccuracyMetric: acc=0.809633 661533ms

        Evaluate data in 0.31 seconds!
        Epoch 9 Avg Loss: 0.03 AccuracyMetric: acc=0.797018 812232ms

        Evaluate data in 0.25 seconds!
        [tester] 
        AccuracyMetric: acc=0.917822
        


