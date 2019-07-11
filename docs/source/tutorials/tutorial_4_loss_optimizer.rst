==============================================================================
Loss 和 optimizer 教程 ———— 以文本分类为例
==============================================================================

我们使用和 :doc:`/user/quickstart` 中一样的任务来进行详细的介绍。给出一段评价性文字，预测其情感倾向是积极（label=1）、消极（label=0）还是中性（label=2），使用 :class:`~fastNLP.Trainer`  和  :class:`~fastNLP.Tester`  来进行快速训练和测试，损失函数之前的内容与 :doc:`/tutorials/tutorial_5_datasetiter` 中的完全一样，如已经阅读过可以跳过。

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
        dataset = loader.load("./trainDevTestTrees_PTB/trees/all.txt")
        print(dataset[0])

    输出数据如下::
	
        {'words': ['It', "'s", 'a', 'lovely', 'film', 'with', 'lovely', 'performances', 'by', 'Buy', 'and', 'Accorsi', '.'] type=list,
        'target': positive type=str}

    除了读取数据外，fastNLP 还提供了读取其它文件类型的 Loader 类、读取 Embedding的 Loader 等。详见 :doc:`/fastNLP.io` 。
    

数据处理
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
      
损失函数
    训练模型需要提供一个损失函数
    ,fastNLP中提供了直接可以导入使用的四种loss，分别为：
    * :class:`~fastNLP.CrossEntropyLoss`：包装了torch.nn.functional.cross_entropy()函数，返回交叉熵损失（可以运用于多分类场景）  
    * :class:`~fastNLP.BCELoss`：包装了torch.nn.functional.binary_cross_entropy()函数，返回二分类的交叉熵  
    * :class:`~fastNLP.L1Loss`：包装了torch.nn.functional.l1_loss()函数，返回L1 损失  
    * :class:`~fastNLP.NLLLoss`：包装了torch.nn.functional.nll_loss()函数，返回负对数似然损失
    
    下面提供了一个在分类问题中常用的交叉熵损失。注意它的 **初始化参数** 。
    ``pred`` 参数对应的是模型的 forward 方法返回的 dict 中的一个 key 的名字。
    ``target`` 参数对应的是 :class:`~fastNLP.DataSet` 中作为标签的 :mod:`~fastNLP.core.field` 的名字。
    这里我们用 :class:`~fastNLP.Const` 来辅助命名，如果你自己编写模型中 forward 方法的返回值或
    数据集中 :mod:`~fastNLP.core.field` 的名字与本例不同， 你可以把 ``pred`` 参数和 ``target`` 参数设定符合自己代码的值。

    .. code-block:: python

        from fastNLP import CrossEntropyLoss
	
        # loss = CrossEntropyLoss() 在本例中与下面这行代码等价
        loss = CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET)
	
优化器
    定义模型运行的时候使用的优化器，可以使用fastNLP包装好的优化器：
	
    * :class:`~fastNLP.SGD` ：包装了torch.optim.SGD优化器
    * :class:`~fastNLP.Adam` ：包装了torch.optim.Adam优化器
	
    也可以直接使用torch.optim.Optimizer中的优化器，并在实例化 :class:`~fastNLP.Trainer` 类的时候传入优化器实参
    
    .. code-block:: python

        import torch.optim as optim
        from fastNLP import Adam

        #使用 torch.optim 定义优化器
        optimizer_1=optim.RMSprop(model_cnn.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        #使用fastNLP中包装的 Adam 定义优化器
        optimizer_2=Adam(lr=4e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, model_params=model_cnn.parameters())

快速训练
    现在我们可以导入 fastNLP 内置的文本分类模型 :class:`~fastNLP.models.CNNText` ，并使用 :class:`~fastNLP.Trainer` 进行训练，
    除了使用 :class:`~fastNLP.Trainer`进行训练，我们也可以通过使用 :class:`~fastNLP.DataSetIter` 来编写自己的训练过程，具体见 :doc:`/tutorials/tutorial_5_datasetiter`

    .. code-block:: python

        from fastNLP.models import CNNText

        #词嵌入的维度、训练的轮数和batch size
        EMBED_DIM = 100
        N_EPOCHS = 10
        BATCH_SIZE = 16

        #使用CNNText的时候第一个参数输入一个tuple,作为模型定义embedding的参数
        #还可以传入 kernel_nums, kernel_sizes, padding, dropout的自定义值
        model_cnn = CNNText((len(vocab),EMBED_DIM), num_classes=3, padding=2, dropout=0.1)

        #如果在定义trainer的时候没有传入optimizer参数，模型默认的优化器为torch.optim.Adam且learning rate为lr=4e-3
        #这里只使用了optimizer_1作为优化器输入，感兴趣可以尝试optimizer_2或者其他优化器作为输入
        #这里只使用了loss作为损失函数输入，感兴趣可以尝试其他损失函数输入
        trainer = Trainer(model=model_cnn, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics, 
        optimizer=optimizer_1,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
        trainer.train()

    训练过程的输出如下::
	
        input fields after batch(if batch size is 2):
        	      words: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 40]) 
                seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
        target fields after batch(if batch size is 2):
                target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 

        training epochs started 2019-07-08-15-44-48
        Evaluation at Epoch 1/10. Step:601/6010. AccuracyMetric: acc=0.59044

        Evaluation at Epoch 2/10. Step:1202/6010. AccuracyMetric: acc=0.599813

        Evaluation at Epoch 3/10. Step:1803/6010. AccuracyMetric: acc=0.508903

        Evaluation at Epoch 4/10. Step:2404/6010. AccuracyMetric: acc=0.596064

        Evaluation at Epoch 5/10. Step:3005/6010. AccuracyMetric: acc=0.47985

        Evaluation at Epoch 6/10. Step:3606/6010. AccuracyMetric: acc=0.589503

        Evaluation at Epoch 7/10. Step:4207/6010. AccuracyMetric: acc=0.311153

        Evaluation at Epoch 8/10. Step:4808/6010. AccuracyMetric: acc=0.549203

        Evaluation at Epoch 9/10. Step:5409/6010. AccuracyMetric: acc=0.581068

        Evaluation at Epoch 10/10. Step:6010/6010. AccuracyMetric: acc=0.523899


        In Epoch:2/Step:1202, got best dev performance:AccuracyMetric: acc=0.599813
        Reloaded the best model.

快速测试
    与 :class:`~fastNLP.Trainer` 对应，fastNLP 也提供了 :class:`~fastNLP.Tester` 用于快速测试，用法如下

    .. code-block:: python

        from fastNLP import Tester

        tester = Tester(test_data, model_cnn, metrics=AccuracyMetric())
        tester.test()
    
    训练过程输出如下::
	
        [tester] 
        AccuracyMetric: acc=0.565401
