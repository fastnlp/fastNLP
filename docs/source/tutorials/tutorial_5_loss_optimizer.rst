==============================================================================
动手实现一个文本分类器I-使用Trainer和Tester快速训练和测试
==============================================================================

我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。给出一段评价性文字，预测其情感倾向是积极的（label=0）、
还是消极的（label=1），使用 :class:`~fastNLP.Trainer`  和  :class:`~fastNLP.Tester`  来进行快速训练和测试。

-----------------
数据读入和处理
-----------------

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

        train_data = databundle.datasets['train']
        train_data, test_data = train_data.split(0.015)
        dev_data = databundle.datasets['dev']
        print(len(train_data),len(dev_data),len(test_data))

    输出结果为::
	
        66339 872 1010

数据集 :meth:`~fastNLP.DataSet.set_input` 和  :meth:`~fastNLP.DataSet.set_target` 函数
    :class:`~fastNLP.io.SST2Pipe`  类的 :meth:`~fastNLP.io.SST2Pipe.process_from_file` 方法在预处理过程中还将训练、测试、验证集的 `words` 、`seq_len` :mod:`~fastNLP.core.field` 设定为input，同时将 `target`  :mod:`~fastNLP.core.field` 设定为target。我们可以通过 :class:`~fastNLP.core.Dataset` 类的 :meth:`~fastNLP.core.Dataset.print_field_meta` 方法查看各个       :mod:`~fastNLP.core.field` 的设定情况，代码如下：

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

    其中is_input和is_target分别表示是否为input和target。ignore_type为true时指使用  :class:`~fastNLP.DataSetIter` 取出batch数据时fastNLP不会进行自动padding，pad_value指对应 :mod:`~fastNLP.core.field` padding所用的值，这两者只有当 :mod:`~fastNLP.core.field` 设定为input或者target的时候才有存在的意义。

    is_input为true的 :mod:`~fastNLP.core.field` 在 :class:`~fastNLP.DataSetIter` 迭代取出的 batch_x 中，而 is_target为true的 :mod:`~fastNLP.core.field` 在                 :class:`~fastNLP.DataSetIter` 迭代取出的 batch_y 中。具体分析见 :doc:`/tutorials/tutorial_6_datasetiter` 的DataSetIter初探。

---------------------
使用内置模型训练
---------------------
模型定义和初始化
    我们可以导入 fastNLP 内置的文本分类模型 :class:`~fastNLP.models.CNNText` 来对模型进行定义，代码如下：

    .. code-block:: python

        from fastNLP.models import CNNText

        #词嵌入的维度
        EMBED_DIM = 100

        #使用CNNText的时候第一个参数输入一个tuple,作为模型定义embedding的参数
        #还可以传入 kernel_nums, kernel_sizes, padding, dropout的自定义值
        model_cnn = CNNText((len(vocab),EMBED_DIM), num_classes=2, dropout=0.1)

    使用fastNLP快速搭建自己的模型详见 :doc:`/tutorials/tutorial_8_modules_models`  。

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

    
损失函数
    训练模型需要提供一个损失函数
    ,fastNLP中提供了直接可以导入使用的四种loss，分别为：
    
    * :class:`~fastNLP.CrossEntropyLoss`：包装了torch.nn.functional.cross_entropy()函数，返回交叉熵损失（可以运用于多分类场景）  
    * :class:`~fastNLP.BCELoss`：包装了torch.nn.functional.binary_cross_entropy()函数，返回二分类的交叉熵  
    * :class:`~fastNLP.L1Loss`：包装了torch.nn.functional.l1_loss()函数，返回L1 损失  
    * :class:`~fastNLP.NLLLoss`：包装了torch.nn.functional.nll_loss()函数，返回负对数似然损失
    
    下面提供了一个在分类问题中常用的交叉熵损失。注意它的 **初始化参数** 。

    * ``pred`` 参数对应的是模型的 forward 方法返回的 dict 中的一个 key 的名字。
    * ``target`` 参数对应的是 :class:`~fastNLP.DataSet` 中作为标签的 :mod:`~fastNLP.core.field` 的名字。

    这里我们用 :class:`~fastNLP.Const` 来辅助命名，如果你自己编写模型中 forward 方法的返回值或
    数据集中 :mod:`~fastNLP.core.field` 的名字与本例不同， 你可以把 ``pred`` 参数和 ``target`` 参数设定符合自己代码的值。

    .. code-block:: python

        from fastNLP import CrossEntropyLoss
	
        # loss = CrossEntropyLoss() 在本例中与下面这行代码等价
        loss = CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET)
     
    除了使用fastNLP已经包装好的了损失函数，也可以通过fastNLP中的LossFunc类来构建自己的损失函数，方法如下：

    .. code-block:: python

        # 这表示构建了一个损失函数类，由func计算损失函数，其中将从模型返回值或者DataSet的target=True的field
        # 当中找到一个参数名为`pred`的参数传入func一个参数名为`input`的参数；找到一个参数名为`label`的参数
        # 传入func作为一个名为`target`的参数
        #下面自己构建了一个交叉熵函数，和之后直接使用fastNLP中的交叉熵函数是一个效果
        import torch
        from fastNLP import LossFunc
        func = torch.nn.functional.cross_entropy
        loss_func = LossFunc(func, input=Const.OUTPUT, target=Const.TARGET)
	
优化器
    定义模型运行的时候使用的优化器，可以直接使用torch.optim.Optimizer中的优化器，并在实例化 :class:`~fastNLP.Trainer` 类的时候传入优化器实参
    
    .. code-block:: python

        import torch.optim as optim

        #使用 torch.optim 定义优化器
        optimizer=optim.RMSprop(model_cnn.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

快速训练
    现在我们对上面定义的模型使用 :class:`~fastNLP.Trainer` 进行训练。
    除了使用 :class:`~fastNLP.Trainer`进行训练，我们也可以通过使用 :class:`~fastNLP.DataSetIter` 来编写自己的训练过程，具体见 :doc:`/tutorials/tutorial_6_datasetiter`

    .. code-block:: python

        from fastNLP import Trainer
        
        #训练的轮数和batch size
        N_EPOCHS = 10
        BATCH_SIZE = 16

        #如果在定义trainer的时候没有传入optimizer参数，模型默认的优化器为torch.optim.Adam且learning rate为lr=4e-3
        #这里只使用了loss作为损失函数输入，感兴趣可以尝试其他损失函数（如之前自定义的loss_func）作为输入
        trainer = Trainer(model=model_cnn, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics, 
        optimizer=optimizer,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
        trainer.train()

    训练过程的输出如下::

        input fields after batch(if batch size is 2):
        	        words: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 16]) 
        	        seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
        target fields after batch(if batch size is 2):
	        target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 

        training epochs started 2019-09-17-14-29-00

        Evaluate data in 0.11 seconds!
        Evaluation on dev at Epoch 1/10. Step:4147/41470: 
        AccuracyMetric: acc=0.762615

        Evaluate data in 0.19 seconds!
        Evaluation on dev at Epoch 2/10. Step:8294/41470: 
        AccuracyMetric: acc=0.800459

        Evaluate data in 0.16 seconds!
        Evaluation on dev at Epoch 3/10. Step:12441/41470: 
        AccuracyMetric: acc=0.777523

        Evaluate data in 0.11 seconds!
        Evaluation on dev at Epoch 4/10. Step:16588/41470: 
        AccuracyMetric: acc=0.634174

        Evaluate data in 0.11 seconds!
        Evaluation on dev at Epoch 5/10. Step:20735/41470: 
        AccuracyMetric: acc=0.791284

        Evaluate data in 0.15 seconds!
        Evaluation on dev at Epoch 6/10. Step:24882/41470: 
        AccuracyMetric: acc=0.573394

        Evaluate data in 0.18 seconds!
        Evaluation on dev at Epoch 7/10. Step:29029/41470: 
        AccuracyMetric: acc=0.759174

        Evaluate data in 0.17 seconds!
        Evaluation on dev at Epoch 8/10. Step:33176/41470: 
        AccuracyMetric: acc=0.776376

        Evaluate data in 0.18 seconds!
        Evaluation on dev at Epoch 9/10. Step:37323/41470: 
        AccuracyMetric: acc=0.740826

        Evaluate data in 0.2 seconds!
        Evaluation on dev at Epoch 10/10. Step:41470/41470: 
        AccuracyMetric: acc=0.769495

        In Epoch:2/Step:8294, got best dev performance:
        AccuracyMetric: acc=0.800459
        Reloaded the best model.

快速测试
    与 :class:`~fastNLP.Trainer` 对应，fastNLP 也提供了 :class:`~fastNLP.Tester` 用于快速测试，用法如下

    .. code-block:: python

        from fastNLP import Tester

        tester = Tester(test_data, model_cnn, metrics=AccuracyMetric())
        tester.test()
    
    训练过程输出如下::
	
        Evaluate data in 0.19 seconds!
        [tester] 
        AccuracyMetric: acc=0.889109
