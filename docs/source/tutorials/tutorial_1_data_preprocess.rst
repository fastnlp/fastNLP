==============================
fastNLP中的DataSet
==============================

:class:`~fastNLP.DataSet` 是fastNLP用于承载数据的类，一般训练集、验证集和测试集会被加载为三个单独的 :class:`~fastNLP.DataSet` 对象。

:class:`~fastNLP.DataSet` 中的数据组织形式类似一个表格，比如下面 :class:`~fastNLP.DataSet` 一共有3列，列在fastNLP中被称为field。

.. csv-table::
   :header: "raw_chars", "chars", "seq_len"

   "历任公司副总经理、总工程师，", "[历 任 公 司 副 总 经 理 、 总 工 程 师 ，]", 6
   "Third instance .", "[Third, instance, .]", 3
   "...", "[...]", "..."

每一行是一个instance (在fastNLP中被称为 :mod:`~fastNLP.core.Instance` )，
每一列是一个field (在fastNLP中称为 :mod:`~fastNLP.core.FieldArray` )。

DataSet构建和删除
-----------------------------

我们使用传入字典的方式构建一个数据集，这是 :class:`~fastNLP.DataSet` 初始化的最基础的方式

.. code-block:: python

    from fastNLP import DataSet
    data = {'raw_words':["This is the first instance .", "Second instance .", "Third instance ."],
            'words': [['this', 'is', 'the', 'first', 'instance', '.'], ['Second', 'instance', '.'], ['Third', 'instance', '.']],
            'seq_len': [6, 3, 3]}
    dataset = DataSet(data)
    # 传入的dict的每个key的value应该为具有相同长度的list
    print(dataset)

输出为::

    +------------------------------+------------------------------------------------+---------+
    |           raw_words          |                     words                      | seq_len |
    +------------------------------+------------------------------------------------+---------+
    | This is the first instance . | ['this', 'is', 'the', 'first', 'instance', ... |    6    |
    |      Second instance .       |          ['Second', 'instance', '.']           |    3    |
    |       Third instance .       |           ['Third', 'instance', '.']           |    3    |
    +------------------------------+------------------------------------------------+---------+


我们还可以使用 :func:`~fastNLP.DataSet.append` 方法向数据集内增加数据

.. code-block:: python

    from fastNLP import DataSet
    from fastNLP import Instance
    dataset = DataSet()
    instance = Instance(raw_words="This is the first instance",
                        words=['this', 'is', 'the', 'first', 'instance', '.'],
                        seq_len=6)
    dataset.append(instance)
    # 可以继续append更多内容，但是append的instance应该和前面的instance拥有完全相同的field

另外，我们还可以用 :class:`~fastNLP.Instance` 数组的方式构建数据集

.. code-block:: python

    from fastNLP import DataSet
    from fastNLP import Instance
    dataset = DataSet([
        Instance(raw_words="This is the first instance",
            words=['this', 'is', 'the', 'first', 'instance', '.'],
            seq_len=6),
        Instance(raw_words="Second instance .",
            words=['Second', 'instance', '.'],
            seq_len=3)
        ])

在初步构建完数据集之后，我们可以通过 `for` 循环遍历 :class:`~fastNLP.DataSet` 中的内容。

.. code-block:: python

    for instance in dataset:
        # do something

FastNLP 同样提供了多种删除数据的方法 :func:`~fastNLP.DataSet.drop` 、 :func:`~fastNLP.DataSet.delete_instance` 和 :func:`~fastNLP.DataSet.delete_field`

.. code-block:: python

    from fastNLP import DataSet
    dataset = DataSet({'a': list(range(-5, 5))})
    # 返回满足条件的instance,并放入DataSet中
    dropped_dataset = dataset.drop(lambda ins:ins['a']<0, inplace=False)
    # 在dataset中删除满足条件的instance
    dataset.drop(lambda ins:ins['a']<0)  # dataset的instance数量减少
    #  删除第3个instance
    dataset.delete_instance(2)
    #  删除名为'a'的field
    dataset.delete_field('a')


简单的数据预处理
-----------------------------

因为 fastNLP 中的数据是按列存储的，所以大部分的数据预处理操作是以列（ :mod:`~fastNLP.core.field` ）为操作对象的。
首先，我们可以检查特定名称的 :mod:`~fastNLP.core.field` 是否存在，并对其进行改名。

.. code-block:: python

    #  检查是否存在名为'a'的field
    dataset.has_field('a')  # 或 ('a' in dataset)
    #  将名为'a'的field改名为'b'
    dataset.rename_field('a', 'b')
    #  DataSet的长度
    len(dataset)

其次，我们可以使用 :func:`~fastNLP.DataSet.apply` 或 :func:`~fastNLP.DataSet.apply_field` 进行数据预处理操作操作。
这两个方法通过传入一个对单一 :mod:`~fastNLP.core.instance` 操作的函数，
自动地帮助你对一个 :mod:`~fastNLP.core.field` 中的每个 :mod:`~fastNLP.core.instance` 调用这个函数，完成整体的操作。
这个传入的函数可以是 lambda 匿名函数，也可以是完整定义的函数。同时，你还可以用 ``new_field_name`` 参数指定数据处理后存储的 :mod:`~fastNLP.core.field` 的名称。

.. code-block:: python

    from fastNLP import DataSet
    data = {'raw_words':["This is the first instance .", "Second instance .", "Third instance ."]}
    dataset = DataSet(data)

    # 将句子分成单词形式, 详见DataSet.apply()方法
    dataset.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')

    # 或使用DataSet.apply_field()
    dataset.apply_field(lambda sent:sent.split(), field_name='raw_words', new_field_name='words')

    # 除了匿名函数，也可以定义函数传递进去
    def get_words(instance):
        sentence = instance['raw_words']
        words = sentence.split()
        return words
    dataset.apply(get_words, new_field_name='words')

除了手动处理数据集之外，你还可以使用 fastNLP 提供的各种 :class:`~fastNLP.io.Loader` 和 :class:`~fastNLP.io.Pipe` 来进行数据处理。
详细请参考这篇教程  :doc:`使用Loader和Pipe处理数据 </tutorials/tutorial_4_load_dataset>` 。


fastNLP中field的命名习惯
-----------------------------

在英文任务中，fastNLP常用的field名称有:

    - **raw_words**: 表示的是原始的str。例如"This is a demo sentence ."。存在多个raw_words的情况，例如matching任务，它们会被定义为raw_words0, raw_words1。但在conll格式下，raw_words列也可能为["This", "is", "a", "demo", "sentence", "."]的形式。
    - **words**: 表示的是已经tokenize后的词语。例如["This", "is", "a", "demo", "sentence"], 但由于str并不能直接被神经网络所使用，所以words中的内容往往被转换为int，如[3, 10, 4, 2, 7, ...]等。多列words的情况，会被命名为words0, words1
    - **target**: 表示目标值。分类场景下，只有一个值；序列标注场景下是一个序列。
    - **seq_len**: 一般用于表示words列的长度

在中文任务中，fastNLP常用的field名称有:

    - **raw_words**: 如果原始汉字序列中已经包含了词语的边界，则该列称为raw_words。如"上海 浦东 开发 与 法制 建设 同步"。
    - **words**: 表示单独的汉字词语序列。例如["上海", "", "浦东", "开发", "与", "法制", "建设", ...]或[2, 3, 4, ...]
    - **raw_chars**: 表示的是原始的连续汉字序列。例如"这是一个示例。"
    - **chars**: 表示已经切分为单独的汉字的序列。例如["这", "是", "一", "个", "示", "例", "。"]。但由于神经网络不能识别汉字，所以一般该列会被转为int形式，如[3, 4, 5, 6, ...]。
    - **target**: 表示目标值。分类场景下，只有一个值；序列标注场景下是一个序列
    - **seq_len**: 表示输入序列的长度
