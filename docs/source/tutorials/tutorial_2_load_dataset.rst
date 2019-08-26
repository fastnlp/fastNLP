=======================================
使用Loader和Pipe加载并处理数据集
=======================================

这一部分是一个关于如何加载数据集的教程

教程目录：

    - `Part I: 数据集容器DataBundle`_
    - `Part II: 加载数据集的基类Loader`_
    - `Part III: 不同格式类型的基础Loader`_
    - `Part IV: 使用Pipe对数据集进行预处理`_
    - `Part V: fastNLP封装好的Loader和Pipe`_


------------------------------------
Part I: 数据集容器DataBundle
------------------------------------

在fastNLP中，我们使用 :class:`~fastNLP.io.data_bundle.DataBundle` 来存储数据集信息。
:class:`~fastNLP.io.data_bundle.DataBundle` 类包含了两个重要内容： `datasets` 和 `vocabs` 。

`datasets` 是一个 `key` 为数据集名称（如 `train` ， `dev` ，和 `test` 等）， `value` 为 :class:`~fastNLP.DataSet` 的字典。

`vocabs` 是一个 `key` 为词表名称（如 :attr:`fastNLP.Const.INPUT` 表示输入文本的词表名称， :attr:`fastNLP.Const.TARGET` 表示目标
的真实标签词表的名称，等等）， `value` 为词表内容（ :class:`~fastNLP.Vocabulary` ）的字典。

-------------------------------------
Part II: 加载数据集的基类Loader
-------------------------------------

在fastNLP中，我们采用 :class:`~fastNLP.io.loader.Loader` 来作为加载数据集的基类。
:class:`~fastNLP.io.loader.Loader` 定义了各种Loader所需的API接口，开发者应该继承它实现各种的Loader。
在各种数据集的Loader当中，至少应该编写如下内容:

    - _load 函数：从一个数据文件中读取数据，返回一个 :class:`~fastNLP.DataSet`
    - load 函数：从文件或者文件夹中读取数据并组装成 :class:`~fastNLP.io.data_bundle.DataBundle`

Loader的load函数返回的 :class:`~fastNLP.io.data_bundle.DataBundle` 里面包含了数据集的原始数据。

--------------------------------------------------------
Part III: 不同格式类型的基础Loader
--------------------------------------------------------

:class:`~fastNLP.io.loader.CSVLoader`
    读取CSV类型的数据集文件。例子如下：

    .. code-block:: python

        from fastNLP.io.loader import CSVLoader
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


:class:`~fastNLP.io.loader.JsonLoader`
    读取Json类型的数据集文件，数据必须按行存储，每行是一个包含各类属性的Json对象。例子如下：

    .. code-block:: python

        from fastNLP.io.loader import JsonLoader
        oader = JsonLoader(
            fields={'sentence1': 'words1', 'sentence2': 'words2', 'gold_label': 'target'}
        )
        # 表示将Json对象中'sentence1'、'sentence2'和'gold_label'对应的值赋给'words1'、'words2'、'target'这三个fields

        data_set = loader._load('path/to/your/file')

    数据集内容样例如下 ::

        {"annotator_labels": ["neutral"], "captionID": "3416050480.jpg#4", "gold_label": "neutral", "pairID": "3416050480.jpg#4r1n", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is training his horse for a competition.", "sentence2_binary_parse": "( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))"}
        {"annotator_labels": ["contradiction"], "captionID": "3416050480.jpg#4", "gold_label": "contradiction", "pairID": "3416050480.jpg#4r1c", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is at a diner, ordering an omelette.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is ( at ( a diner ) ) ) , ) ( ordering ( an omelette ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (PP (IN at) (NP (DT a) (NN diner))) (, ,) (S (VP (VBG ordering) (NP (DT an) (NN omelette))))) (. .)))"}
        {"annotator_labels": ["entailment"], "captionID": "3416050480.jpg#4", "gold_label": "entailment", "pairID": "3416050480.jpg#4r1e", "sentence1": "A person on a horse jumps over a broken down airplane.", "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))", "sentence2": "A person is outdoors, on a horse.", "sentence2_binary_parse": "( ( A person ) ( ( ( ( is outdoors ) , ) ( on ( a horse ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (ADVP (RB outdoors)) (, ,) (PP (IN on) (NP (DT a) (NN horse)))) (. .)))"}

------------------------------------------
Part IV: 使用Pipe对数据集进行预处理
------------------------------------------

在fastNLP中，我们采用 :class:`~fastNLP.io.pipe.Pipe` 来作为加载数据集的基类。
:class:`~fastNLP.io.pipe.Pipe` 定义了各种Pipe所需的API接口，开发者应该继承它实现各种的Pipe。
在各种数据集的Pipe当中，至少应该编写如下内容:

    - process 函数：对输入的 :class:`~fastNLP.io.data_bundle.DataBundle` 进行处理（如构建词表、
      将dataset的文本内容转成index等等），然后返回该 :class:`~fastNLP.io.data_bundle.DataBundle`
    - process_from_file 函数：输入数据集所在文件夹，读取内容并组装成 :class:`~fastNLP.io.data_bundle.DataBundle` ，
      然后调用相对应的process函数对数据进行预处理

以SNLI数据集为例，写一个自定义Pipe的例子如下：

.. code-block:: python

    from fastNLP.io.loader import SNLILoader
    from fastNLP.io.pipe import MatchingPipe

    class MySNLIPipe(MatchingPipe):

        def process(self, data_bundle):
            data_bundle = super(MySNLIPipe, self).process(data_bundle)
            # MatchingPipe类里封装了一个关于matching任务的process函数，可以直接继承使用
            # 如果有需要进行额外的预处理操作可以在这里加入您的代码
            return data_bundle

        def process_from_file(self, paths=None):
            data_bundle = SNLILoader().load(paths) # 使用SNLILoader读取原始数据集
            # SNLILoader的load函数中，paths如果为None则会自动下载
            return self.process(data_bundle)  # 调用相对应的process函数对data_bundle进行处理

调用Pipe示例：

.. code-block:: python

    from fastNLP.io.pipe import SNLIBertPipe
    data_bundle = SNLIBertPipe(lower=True, tokenizer=arg.tokenizer).process_from_file()
    print(data_bundle)

输出的内容是::

    In total 3 datasets:
            train has 549367 instances.
            dev has 9842 instances.
            test has 9824 instances.
    In total 2 vocabs:
            words has 34184 entries.
            target has 3 entries.

这里表示一共有3个数据集和2个词表。其中：

    - 3个数据集分别为train、dev、test数据集，分别有549367、9842、9824个instance
    - 2个词表分别为words词表与target词表。其中words词表为句子文本所构建的词表，一共有34184个单词；
      target词表为目标标签所构建的词表，一共有3种标签。（注：如果有多个输入，则句子文本所构建的词表将
      会被命名为words1以对应相对应的列名）

------------------------------------------
Part V: fastNLP封装好的Loader和Pipe
------------------------------------------

fastNLP封装了多种任务/数据集的Loader和Pipe并提供自动下载功能，具体参见文档

`fastNLP可加载的embedding与数据集 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_

