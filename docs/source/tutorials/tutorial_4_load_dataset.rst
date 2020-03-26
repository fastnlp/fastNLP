=======================================
使用Loader和Pipe加载并处理数据集
=======================================

这一部分是关于如何加载数据集的教程

教程目录：

    - `Part I: 数据集容器DataBundle`_
    - `Part II: 加载的各种数据集的Loader`_
    - `Part III: 使用Pipe对数据集进行预处理`_
    - `Part IV: fastNLP封装好的Loader和Pipe`_
    - `Part V: 不同格式类型的基础Loader`_


Part I: 数据集容器DataBundle
------------------------------------

而由于对于同一个任务，训练集，验证集和测试集会共用同一个词表以及具有相同的目标值，所以在fastNLP中我们使用了 :class:`~fastNLP.io.DataBundle`
来承载同一个任务的多个数据集 :class:`~fastNLP.DataSet` 以及它们的词表 :class:`~fastNLP.Vocabulary` 。下面会有例子介绍 :class:`~fastNLP.io.DataBundle`
的相关使用。

:class:`~fastNLP.io.DataBundle` 在fastNLP中主要在各个 :class:`~fastNLP.io.Loader` 和 :class:`~fastNLP.io.Pipe` 中被使用。
下面我们先介绍一下 :class:`~fastNLP.io.Loader` 和 :class:`~fastNLP.io.Pipe` 。

Part II: 加载的各种数据集的Loader
-------------------------------------

在fastNLP中，所有的 :class:`~fastNLP.io.Loader` 都可以通过其文档判断其支持读取的数据格式，以及读取之后返回的 :class:`~fastNLP.DataSet` 的格式,
例如 :class:`~fastNLP.io.ChnSentiCorpLoader` 。

    - **download()** 函数：自动将该数据集下载到缓存地址，默认缓存地址为~/.fastNLP/datasets/。由于版权等原因，不是所有的Loader都实现了该方法。该方法会返回下载后文件所处的缓存地址。
    - **_load()** 函数：从一个数据文件中读取数据，返回一个 :class:`~fastNLP.DataSet` 。返回的DataSet的格式可从Loader文档判断。
    - **load()** 函数：从文件或者文件夹中读取数据为 :class:`~fastNLP.DataSet` 并将它们组装成 :class:`~fastNLP.io.DataBundle`。支持接受的参数类型有以下的几种

        - None, 将尝试读取自动缓存的数据，仅支持提供了自动下载数据的Loader
        - 文件夹路径, 默认将尝试在该文件夹下匹配文件名中含有 `train` , `test` , `dev` 的文件，如果有多个文件含有相同的关键字，将无法通过该方式读取
        - dict, 例如{'train':"/path/to/tr.conll", 'dev':"/to/validate.conll", "test":"/to/te.conll"}。

.. code-block:: python

    from fastNLP.io import CWSLoader

    loader = CWSLoader(dataset_name='pku')
    data_bundle = loader.load()
    print(data_bundle)

输出内容为::

    In total 3 datasets:
        dev has 1831 instances.
        train has 17223 instances.
        test has 1944 instances.

这里表示一共有3个数据集。其中：

    - 3个数据集的名称分别为train、dev、test，分别有17223、1831、1944个instance

也可以取出DataSet，并打印DataSet中的具体内容

.. code-block:: python

    tr_data = data_bundle.get_dataset('train')
    print(tr_data[:2])

输出为::

    +--------------------------------------------------------------------------------------+
    |                                      raw_words                                       |
    +--------------------------------------------------------------------------------------+
    | 迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）        |
    |                      中共中央  总书记  、  国家  主席  江  泽民                          |
    +--------------------------------------------------------------------------------------+

Part III: 使用Pipe对数据集进行预处理
------------------------------------------
通过 :class:`~fastNLP.io.Loader` 可以将文本数据读入，但并不能直接被神经网络使用，还需要进行一定的预处理。

在fastNLP中，我们使用 :class:`~fastNLP.io.Pipe` 的子类作为数据预处理的类， :class:`~fastNLP.io.Loader` 和 :class:`~fastNLP.io.Pipe` 一般具备一一对应的关系，该关系可以从其名称判断，
例如 :class:`~fastNLP.io.CWSLoader` 与 :class:`~fastNLP.io.CWSPipe` 是一一对应的。一般情况下Pipe处理包含以下的几个过程，(1)将raw_words或
raw_chars进行tokenize以切分成不同的词或字; (2) 再建立词或字的 :class:`~fastNLP.Vocabulary` , 并将词或字转换为index; (3)将target
列建立词表并将target列转为index;

所有的Pipe都可通过其文档查看该Pipe支持处理的 :class:`~fastNLP.DataSet` 以及返回的 :class:`~fastNLP.io.DataBundle` 中的Vocabulary的情况;
如 :class:`~fastNLP.io.OntoNotesNERPipe`

各种数据集的Pipe当中，都包含了以下的两个函数:

    - process() 函数：对输入的 :class:`~fastNLP.io.DataBundle` 进行处理, 然后返回处理之后的 :class:`~fastNLP.io.DataBundle` 。process函数的文档中包含了该Pipe支持处理的DataSet的格式。
    - process_from_file() 函数：输入数据集所在文件夹，使用对应的Loader读取数据(所以该函数支持的参数类型是由于其对应的Loader的load函数决定的)，然后调用相对应的process函数对数据进行预处理。相当于是把Load和process放在一个函数中执行。

接着上面 :class:`~fastNLP.io.CWSLoader` 的例子，我们展示一下 :class:`~fastNLP.io.CWSPipe` 的功能：

.. code-block:: python

    from fastNLP.io import CWSPipe

    data_bundle = CWSPipe().process(data_bundle)
    print(data_bundle)

输出内容为::

    In total 3 datasets:
        dev has 1831 instances.
        train has 17223 instances.
        test has 1944 instances.
    In total 2 vocabs:
        chars has 4777 entries.
        target has 4 entries.

表示一共有3个数据集和2个词表。其中：

    - 3个数据集的名称分别为train、dev、test，分别有17223、1831、1944个instance
    - 2个词表分别为chars词表与target词表。其中chars词表为句子文本所构建的词表，一共有4777个不同的字；target词表为目标标签所构建的词表，一共有4种标签。

相较于之前CWSLoader读取的DataBundle，新增了两个Vocabulary。 我们可以打印一下处理之后的DataSet

.. code-block:: python

    tr_data = data_bundle.get_dataset('train')
    print(tr_data[:2])

输出为::

    +---------------------------------------------------+------------------------------------+------------------------------------+---------+
    |                     raw_words                     |               chars                |               target               | seq_len |
    +---------------------------------------------------+------------------------------------+------------------------------------+---------+
    | 迈向  充满  希望  的  新  世纪  ——  一九九八年...     | [1224, 178, 674, 544, 573, 435,... | [0, 1, 0, 1, 0, 1, 2, 2, 0, 1, ... |    29   |
    |     中共中央  总书记  、  国家  主席  江  泽民        | [11, 212, 11, 335, 124, 256, 10... | [0, 3, 3, 1, 0, 3, 1, 2, 0, 1, ... |    15   |
    +---------------------------------------------------+------------------------------------+------------------------------------+---------+

可以看到有两列为int的field: chars和target。这两列的名称同时也是DataBundle中的Vocabulary的名称。可以通过下列的代码获取并查看Vocabulary的
信息

.. code-block:: python

    vocab = data_bundle.get_vocab('target')
    print(vocab)

输出为::

    Vocabulary(['B', 'E', 'S', 'M']...)


Part IV: fastNLP封装好的Loader和Pipe
------------------------------------------

fastNLP封装了多种任务/数据集的 :class:`~fastNLP.io.Loader` 和 :class:`~fastNLP.io.Pipe` 并提供自动下载功能，具体参见文档
`数据集 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_


Part V: 不同格式类型的基础Loader
--------------------------------------------------------

除了上面提到的针对具体任务的Loader，我们还提供了CSV格式和JSON格式的Loader

:class:`~fastNLP.io.loader.CSVLoader` 读取CSV类型的数据集文件。例子如下：

    .. code-block:: python

        from fastNLP.io.loader import CSVLoader
        data_set_loader = CSVLoader(
            headers=('raw_words', 'target'), sep='\t'
        )
        # 表示将CSV文件中每一行的第一项将填入'raw_words' field，第二项填入'target' field。
        # 其中项之间由'\t'分割开来

        data_set = data_set_loader._load('path/to/your/file')

    文件内容样例如下 ::

        But it does not leave you with much .	1
        You could hate it for the same reason .	1
        The performances are an absolute joy .	4

    读取之后的DataSet具有以下的field

    .. csv-table::
        :header: raw_words, target

        "But it does not leave you with much .", "1"
        "You could hate it for the same reason .", "1"
        "The performances are an absolute joy .", "4"

:class:`~fastNLP.io.JsonLoader` 读取Json类型的数据集文件，数据必须按行存储，每行是一个包含各类属性的Json对象。例子如下：

    .. code-block:: python

        from fastNLP.io.loader import JsonLoader
        loader = JsonLoader(
            fields={'sentence1': 'raw_words1', 'sentence2': 'raw_words2', 'gold_label': 'target'}
        )
        # 表示将Json对象中'sentence1'、'sentence2'和'gold_label'对应的值赋给'raw_words1'、'raw_words2'、'target'这三个fields

        data_set = loader._load('path/to/your/file')

    数据集内容样例如下 ::

        {"annotator_labels": ["neutral"], "captionID": "3416050480.jpg#4", "gold_label": "neutral", "pairID": "3416050480.jpg#4r1n", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is training his horse for a competition.", "sentence2_binary_parse": "( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))"}
        {"annotator_labels": ["contradiction"], "captionID": "3416050480.jpg#4", "gold_label": "contradiction", "pairID": "3416050480.jpg#4r1c", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is at a diner, ordering an omelette.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is ( at ( a diner ) ) ) , ) ( ordering ( an omelette ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (PP (IN at) (NP (DT a) (NN diner))) (, ,) (S (VP (VBG ordering) (NP (DT an) (NN omelette))))) (. .)))"}
        {"annotator_labels": ["entailment"], "captionID": "3416050480.jpg#4", "gold_label": "entailment", "pairID": "3416050480.jpg#4r1e", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is outdoors, on a horse.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is outdoors ) , ) ( on ( a horse ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (ADVP (RB outdoors)) (, ,) (PP (IN on) (NP (DT a) (NN horse)))) (. .)))"}

    读取之后的DataSet具有以下的field

    .. csv-table::
        :header: raw_words0, raw_words1, target

        "A person on a horse jumps over a broken down airplane.", "A person is training his horse for a competition.", "neutral"
        "A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette.", "contradiction"
        "A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse.", "entailment"


----------------------------------
代码下载
----------------------------------

.. raw:: html

    <a href="../_static/notebooks/tutorial_4_load_dataset.ipynb" download="tutorial_4_load_dataset.ipynb">点击下载 IPython Notebook 文件</a><hr>
