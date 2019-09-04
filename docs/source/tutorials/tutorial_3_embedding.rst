=========================================
使用Embedding模块将文本转成向量
=========================================

这一部分是一个关于在fastNLP当中使用embedding的教程。

教程目录：

    - `Part I: embedding介绍`_
    - `Part II: 使用随机初始化的embedding`_
    - `Part III: 使用预训练的静态embedding`_
    - `Part IV: 使用预训练的Contextual Embedding(ELMo & BERT)`_
    - `Part V: 使用character-level的embedding`_
    - `Part VI: 叠加使用多个embedding`_
    - `Part VII: fastNLP支持的预训练Embedding`_




---------------------------------------
Part I: embedding介绍
---------------------------------------

与torch.nn.Embedding类似，fastNLP的embedding接受的输入是一个被index好的序列，输出的内容是这个序列的embedding结果。

fastNLP的embedding包括了预训练embedding和随机初始化embedding。


---------------------------------------
Part II: 使用随机初始化的embedding
---------------------------------------

使用随机初始化的embedding参见 :class:`~fastNLP.embeddings.embedding.Embedding` 。

可以传入词表大小和embedding维度：

.. code-block:: python

    from fastNLP import Embedding
    embed = Embedding(10000, 50)

也可以传入一个初始化的参数矩阵：

.. code-block:: python

    from fastNLP import Embedding
    embed = Embedding(init_embed)

其中的init_embed可以是torch.FloatTensor、torch.nn.Embedding或者numpy.ndarray。


---------------------------------------
Part III: 使用预训练的静态embedding
---------------------------------------

在使用预训练的embedding之前，需要根据数据集的内容构建一个词表 :class:`~fastNLP.core.vocabulary.Vocabulary` ，在
预训练embedding类初始化的时候需要将这个词表作为参数传入。

在fastNLP中，我们提供了 :class:`~fastNLP.embeddings.StaticEmbedding` 这一个类。
通过 :class:`~fastNLP.embeddings.StaticEmbedding` 可以加载预训练好的静态
Embedding，例子如下：

.. code-block:: python

    from fastNLP import StaticEmbedding
    embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50', requires_grad=True)

vocab为根据数据集构建的词表，model_dir_or_name可以是一个路径，也可以是embedding模型的名称：

    1 如果传入的是路径，那么fastNLP将会根据该路径来读取预训练的权重文件并将embedding加载进来(glove
    和word2vec类型的权重文件都支持)

    2 如果传入的是模型名称，那么fastNLP将会根据名称查找embedding模型，如果在cache目录下找到模型则会
    自动加载；如果找不到则会自动下载到cache目录。默认的cache目录为 `~/.fastNLP` 文件夹。可以通过环境
    变量 ``FASTNLP_CACHE_DIR`` 来自定义cache目录，如::

        $ FASTNLP_CACHE_DIR=~/fastnlp_cache_dir python your_python_file.py

这个命令表示fastNLP将会在 `~/fastnlp_cache_dir` 这个目录下寻找模型，找不到则会自动将模型下载到这个目录

-----------------------------------------------------------
Part IV: 使用预训练的Contextual Embedding(ELMo & BERT)
-----------------------------------------------------------

在fastNLP中，我们提供了ELMo和BERT的embedding： :class:`~fastNLP.embeddings.ElmoEmbedding`
和 :class:`~fastNLP.embeddings.BertEmbedding` 。

与静态embedding类似，ELMo的使用方法如下：

.. code-block:: python

    from fastNLP import ElmoEmbedding
    embed = ElmoEmbedding(vocab, model_dir_or_name='small', requires_grad=False)

BERT-embedding的使用方法如下：

.. code-block:: python

    from fastNLP import BertEmbedding
    embed = BertEmbedding(
        vocab, model_dir_or_name='en-base-cased', requires_grad=False, layers='4,-2,-1'
    )

其中layers变量表示需要取哪几层的encode结果。

-----------------------------------------------------
Part V: 使用character-level的embedding
-----------------------------------------------------

除了预训练的embedding以外，fastNLP还提供了CharEmbedding： :class:`~fastNLP.embeddings.CNNCharEmbedding` 和
:class:`~fastNLP.embeddings.LSTMCharEmbedding` 。

CNNCharEmbedding的使用例子如下：

.. code-block:: python

    from fastNLP import CNNCharEmbedding
    embed = CNNCharEmbedding(vocab, embed_size=100, char_emb_size=50)

这表示这个CNNCharEmbedding当中character的embedding维度大小为50，返回的embedding结果维度大小为100。

与CNNCharEmbedding类似，LSTMCharEmbedding的使用例子如下：

.. code-block:: python

    from fastNLP import LSTMCharEmbedding
    embed = LSTMCharEmbedding(vocab, embed_size=100, char_emb_size=50)

这表示这个LSTMCharEmbedding当中character的embedding维度大小为50，返回的embedding结果维度大小为100。


-----------------------------------------------------
Part VI: 叠加使用多个embedding
-----------------------------------------------------

在fastNLP中，我们使用 :class:`~fastNLP.embeddings.StackEmbedding` 来叠加多个embedding

例子如下：

.. code-block:: python

    from fastNLP import StaticEmbedding, StackEmbedding
    embed_1 = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50', requires_grad=True)
    embed_2 = StaticEmbedding(vocab, model_dir_or_name='en-word2vec-300', requires_grad=True)

    stack_embed = StackEmbedding([embed_1, embed_2])

StackEmbedding会把多个embedding的结果拼接起来，如上面例子的stack_embed返回的embedding维度为350维。

除此以外，还可以把静态embedding跟上下文相关的embedding拼接起来：

.. code-block:: python

    from fastNLP import StaticEmbedding, StackEmbedding, ElmoEmbedding
    elmo_embedding = ElmoEmbedding(vocab, model_dir_or_name='medium', layers='0,1,2', requires_grad=False)
    glove_embedding = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50', requires_grad=True)

    stack_embed = StackEmbedding([elmo_embedding, glove_embedding])

------------------------------------------
Part VII: fastNLP支持的预训练Embedding
------------------------------------------

fastNLP支持多种预训练Embedding并提供自动下载功能，具体参见文档

`fastNLP可加载的embedding与数据集 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_

