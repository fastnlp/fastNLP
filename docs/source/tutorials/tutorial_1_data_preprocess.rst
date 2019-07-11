==============================
数据格式及预处理教程
==============================

:class:`~fastNLP.DataSet` 是fastNLP中用于承载数据的容器。可以将DataSet看做是一个表格，
每一行是一个sample (在fastNLP中被称为 :mod:`~fastNLP.core.instance` )，
每一列是一个feature (在fastNLP中称为 :mod:`~fastNLP.core.field` )。

.. csv-table::
   :header: "sentence", "words", "seq_len"

   "This is the first instance .", "[This, is, the, first, instance, .]", 6
   "Second instance .", "[Second, instance, .]", 3
   "Third instance .", "[Third, instance, .]", 3
   "...", "[...]", "..."

上面是一个样例数据中 DataSet 的存储结构。其中它的每一行是一个 :class:`~fastNLP.Instance` 对象； 每一列是一个 :class:`~fastNLP.FieldArray` 对象。


-----------------------------
数据集构建和删除
-----------------------------

我们使用传入字典的方式构建一个数据集，这是 :class:`~fastNLP.DataSet` 初始化的最基础的方式

.. code-block:: python

    from fastNLP import DataSet
    data = {'sentence':["This is the first instance .", "Second instance .", "Third instance ."],
            'words': [['this', 'is', 'the', 'first', 'instance', '.'], ['Second', 'instance', '.'], ['Third', 'instance', '.']],
            'seq_len': [6, 3, 3]}
    dataset = DataSet(data)
    # 传入的dict的每个key的value应该为具有相同长度的list

我们还可以使用 :func:`~fastNLP.DataSet.append` 方法向数据集内增加数据

.. code-block:: python

    from fastNLP import DataSet
    from fastNLP import Instance
    dataset = DataSet()
    instance = Instance(sentence="This is the first instance",
                        words=['this', 'is', 'the', 'first', 'instance', '.'],
                        seq_len=6)
    dataset.append(instance)
    # 可以继续append更多内容，但是append的instance应该和前面的instance拥有完全相同的field

另外，我们还可以用 :class:`~fastNLP.Instance` 数组的方式构建数据集

.. code-block:: python

    from fastNLP import DataSet
    from fastNLP import Instance
    dataset = DataSet([
        Instance(sentence="This is the first instance",
            words=['this', 'is', 'the', 'first', 'instance', '.'],
            seq_len=6),
        Instance(sentence="Second instance .",
            words=['Second', 'instance', '.'],
            seq_len=3)
        ])

在初步构建完数据集之后，我们可可以通过 `for` 循环遍历 :class:`~fastNLP.DataSet` 中的内容。

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

-----------------------------
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
    data = {'sentence':["This is the first instance .", "Second instance .", "Third instance ."]}
    dataset = DataSet(data)

    # 将句子分成单词形式, 详见DataSet.apply()方法
    dataset.apply(lambda ins: ins['sentence'].split(), new_field_name='words')

    # 或使用DataSet.apply_field()
    dataset.apply_field(lambda sent:sent.split(), field_name='sentence', new_field_name='words')

    # 除了匿名函数，也可以定义函数传递进去
    def get_words(instance):
        sentence = instance['sentence']
        words = sentence.split()
        return words
    dataset.apply(get_words, new_field_name='words')

除了手动处理数据集之外，你还可以使用 fastNLP 提供的各种 :class:`~fastNLP.io.base_loader.DataSetLoader` 来进行数据处理。
详细请参考这篇教程  :doc:`使用DataSetLoader加载数据集 </tutorials/tutorial_2_load_dataset>` 。

-----------------------------
DataSet与pad
-----------------------------

在fastNLP里，pad是与一个 :mod:`~fastNLP.core.field` 绑定的。即不同的 :mod:`~fastNLP.core.field` 可以使用不同的pad方式，比如在英文任务中word需要的pad和
character的pad方式往往是不同的。fastNLP是通过一个叫做 :class:`~fastNLP.Padder` 的子类来完成的。
默认情况下，所有field使用 :class:`~fastNLP.AutoPadder`
。可以通过使用以下方式设置Padder(如果将padder设置为None，则该field不会进行pad操作)。
大多数情况下直接使用 :class:`~fastNLP.AutoPadder` 就可以了。
如果 :class:`~fastNLP.AutoPadder` 或 :class:`~fastNLP.EngChar2DPadder` 无法满足需求，
也可以自己写一个 :class:`~fastNLP.Padder` 。

.. code-block:: python

    from fastNLP import DataSet
    from fastNLP import EngChar2DPadder
    import random
    dataset = DataSet()
    max_chars, max_words, sent_num = 5, 10, 20
    contents = [[
                    [random.randint(1, 27) for _ in range(random.randint(1, max_chars))]
                        for _ in range(random.randint(1, max_words))
                ]  for _ in range(sent_num)]
    #  初始化时传入
    dataset.add_field('chars', contents, padder=EngChar2DPadder())
    #  直接设置
    dataset.set_padder('chars', EngChar2DPadder())
    #  也可以设置pad的value
    dataset.set_pad_val('chars', -1)
