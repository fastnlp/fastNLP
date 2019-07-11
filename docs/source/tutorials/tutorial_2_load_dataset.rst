=================================
使用DataSetLoader加载数据集
=================================

这一部分是一个关于如何加载数据集的教程

教程目录：

    - `Part I: 数据集容器`_
    - `Part II: 数据集的使用方式`_
    - `Part III: 不同数据类型的DataSetLoader`_
    - `Part IV: DataSetLoader举例`_
    - `Part V: fastNLP封装好的数据集加载器`_


----------------------------
Part I: 数据集容器
----------------------------

在fastNLP中，我们使用 :class:`~fastNLP.io.base_loader.DataBundle` 来存储数据集信息。
:class:`~fastNLP.io.base_loader.DataBundle` 类包含了两个重要内容： `datasets` 和 `vocabs` 。

`datasets` 是一个 `key` 为数据集名称（如 `train` ， `dev` ，和 `test` 等）， `value` 为 :class:`~fastNLP.DataSet` 的字典。

`vocabs` 是一个 `key` 为词表名称（如 :attr:`fastNLP.Const.INPUT` 表示输入文本的词表名称， :attr:`fastNLP.Const.TARGET` 表示目标
的真实标签词表的名称，等等）， `value` 为词表内容（ :class:`~fastNLP.Vocabulary` ）的字典。

----------------------------
Part II: 数据集的使用方式
----------------------------

在fastNLP中，我们采用 :class:`~fastNLP.io.base_loader.DataSetLoader` 来作为加载数据集的基类。
:class:`~fastNLP.io.base_loader.DataSetLoader` 定义了各种DataSetLoader所需的API接口，开发者应该继承它实现各种的DataSetLoader。
在各种数据集的DataSetLoader当中，至少应该编写如下内容:

    - _load 函数：从一个数据文件中读取数据到一个 :class:`~fastNLP.DataSet`
    - load 函数（可以使用基类的方法）：从一个或多个数据文件中读取数据到一个或多个 :class:`~fastNLP.DataSet`
    - process 函数：一个或多个从数据文件中读取数据，并处理成可以训练的 :class:`~fastNLP.io.DataInfo`

    **\*process函数中可以调用load函数或_load函数**

DataSetLoader的_load或者load函数返回的 :class:`~fastNLP.DataSet` 当中，内容为数据集的文本信息，process函数返回的
:class:`~fastNLP.io.DataInfo` 当中， `datasets` 的内容为已经index好的、可以直接被 :class:`~fastNLP.Trainer`
接受的内容。

--------------------------------------------------------
Part III: 不同数据类型的DataSetLoader
--------------------------------------------------------

:class:`~fastNLP.io.dataset_loader.CSVLoader`
    读取CSV类型的数据集文件。例子如下：

    .. code-block:: python

        data_set_loader = CSVLoader(
            headers=('words', 'target'), sep='\t'
        )
        # 表示将CSV文件中每一行的第一项填入'words' field，第二项填入'target' field。
        # 其中每两项之间由'\t'分割开来

        data_set = data_set_loader._load('path/to/your/file')

    数据集内容样例如下 ::

        But it does not leave you with much .	1
        You could hate it for the same reason .	1
        The performances are an absolute joy .	4


:class:`~fastNLP.io.dataset_loader.JsonLoader`
    读取Json类型的数据集文件，数据必须按行存储，每行是一个包含各类属性的Json对象。例子如下：

    .. code-block:: python

        data_set_loader = JsonLoader(
            fields={'sentence1': 'words1', 'sentence2': 'words2', 'gold_label': 'target'}
        )
        # 表示将Json对象中'sentence1'、'sentence2'和'gold_label'对应的值赋给'words1'、'words2'、'target'这三个fields

        data_set = data_set_loader._load('path/to/your/file')

    数据集内容样例如下 ::

        {"annotator_labels": ["neutral"], "captionID": "3416050480.jpg#4", "gold_label": "neutral", "pairID": "3416050480.jpg#4r1n", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is training his horse for a competition.", "sentence2_binary_parse": "( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))"}
        {"annotator_labels": ["contradiction"], "captionID": "3416050480.jpg#4", "gold_label": "contradiction", "pairID": "3416050480.jpg#4r1c", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is at a diner, ordering an omelette.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is ( at ( a diner ) ) ) , ) ( ordering ( an omelette ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (PP (IN at) (NP (DT a) (NN diner))) (, ,) (S (VP (VBG ordering) (NP (DT an) (NN omelette))))) (. .)))"}
        {"annotator_labels": ["entailment"], "captionID": "3416050480.jpg#4", "gold_label": "entailment", "pairID": "3416050480.jpg#4r1e", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is outdoors, on a horse.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is outdoors ) , ) ( on ( a horse ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (ADVP (RB outdoors)) (, ,) (PP (IN on) (NP (DT a) (NN horse)))) (. .)))"}

------------------------------------------
Part IV: DataSetLoader举例
------------------------------------------

以Matching任务为例子：

    :class:`~fastNLP.io.data_loader.MatchingLoader`
        我们在fastNLP当中封装了一个Matching任务数据集的数据加载类： :class:`~fastNLP.io.data_loader.MatchingLoader` .

        在MatchingLoader类当中我们封装了一个对数据集中的文本内容进行进一步的预处理的函数：
        :meth:`~fastNLP.io.data_loader.MatchingLoader.process`
        这个函数具有各种预处理option，如：
        - 是否将文本转成全小写
        - 是否需要序列长度信息，需要什么类型的序列长度信息
        - 是否需要用BertTokenizer来获取序列的WordPiece信息
        - 等等

        具体内容参见 :meth:`fastNLP.io.MatchingLoader.process` 。

    :class:`~fastNLP.io.data_loader.SNLILoader`
        一个关于SNLI数据集的DataSetLoader。SNLI数据集来自
        `SNLI Data Set <https://nlp.stanford.edu/projects/snli/snli_1.0.zip>`_ .

        在 :class:`~fastNLP.io.data_loader.SNLILoader` 的 :meth:`~fastNLP.io.data_loader.SNLILoader._load`
        函数中，我们用以下代码将数据集内容从文本文件读入内存：

        .. code-block:: python

                data = SNLILoader().process(
                    paths='path/to/snli/data', to_lower=False, seq_len_type='seq_len',
                    get_index=True, concat=False,
                )
                print(data)

        输出的内容是::

            In total 3 datasets:
                train has 549367 instances.
                dev has 9842 instances.
                test has 9824 instances.
            In total 2 vocabs:
                words has 43154 entries.
                target has 3 entries.


        这里的data是一个 :class:`~fastNLP.io.base_loader.DataBundle` ，取 ``datasets`` 字典里的内容即可直接传入
        :class:`~fastNLP.Trainer` 或者 :class:`~fastNLP.Tester` 进行训练或者测试。

    :class:`~fastNLP.io.data_loader.IMDBLoader`
        以IMDB数据集为例，在 :class:`~fastNLP.io.data_loader.IMDBLoader` 的 :meth:`~fastNLP.io.data_loader.IMDBLoader._load`
        函数中，我们用以下代码将数据集内容从文本文件读入内存：

        .. code-block:: python

                data = IMDBLoader().process(
                    paths={'train': 'path/to/train/file', 'test': 'path/to/test/file'}
                )
                print(data)

        输出的内容是::

            In total 3 datasets:
                train has 22500 instances.
                test has 25000 instances.
                dev has 2500 instances.
            In total 2 vocabs:
                words has 82846 entries.
                target has 2 entries.


        这里的将原来的train集按9:1的比例分成了训练集和验证集。


------------------------------------------
Part V: fastNLP封装好的数据集加载器
------------------------------------------

fastNLP封装好的数据集加载器可以适用于多种类型的任务：

    - `文本分类任务`_
    - `序列标注任务`_
    - `Matching任务`_


文本分类任务
-------------------

==========================    ==================================================================
数据集名称                      数据集加载器
--------------------------    ------------------------------------------------------------------
IMDb                          :class:`~fastNLP.io.data_loader.IMDBLoader`
--------------------------    ------------------------------------------------------------------
SST                           :class:`~fastNLP.io.data_loader.SSTLoader`
--------------------------    ------------------------------------------------------------------
SST-2                         :class:`~fastNLP.io.data_loader.SST2Loader`
--------------------------    ------------------------------------------------------------------
Yelp Polarity                 :class:`~fastNLP.io.data_loader.YelpLoader`
--------------------------    ------------------------------------------------------------------
Yelp Full                     :class:`~fastNLP.io.data_loader.YelpLoader`
--------------------------    ------------------------------------------------------------------
MTL16                         :class:`~fastNLP.io.data_loader.MTL16Loader`
==========================    ==================================================================



序列标注任务
-------------------

==========================    ==================================================================
数据集名称                      数据集加载器
--------------------------    ------------------------------------------------------------------
Conll                         :class:`~fastNLP.io.data_loader.ConllLoader`
--------------------------    ------------------------------------------------------------------
Conll2003                     :class:`~fastNLP.io.data_loader.Conll2003Loader`
--------------------------    ------------------------------------------------------------------
人民日报数据集                   :class:`~fastNLP.io.data_loader.PeopleDailyCorpusLoader`
==========================    ==================================================================



Matching任务
-------------------

==========================    ==================================================================
数据集名称                      数据集加载器
--------------------------    ------------------------------------------------------------------
SNLI                          :class:`~fastNLP.io.data_loader.SNLILoader`
--------------------------    ------------------------------------------------------------------
MultiNLI                      :class:`~fastNLP.io.data_loader.MNLILoader`
--------------------------    ------------------------------------------------------------------
QNLI                          :class:`~fastNLP.io.data_loader.QNLILoader`
--------------------------    ------------------------------------------------------------------
RTE                           :class:`~fastNLP.io.data_loader.RTELoader`
--------------------------    ------------------------------------------------------------------
Quora Pair Dataset            :class:`~fastNLP.io.data_loader.QuoraLoader`
==========================    ==================================================================

