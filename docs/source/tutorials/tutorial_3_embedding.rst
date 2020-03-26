=========================================
使用Embedding模块将文本转成向量
=========================================

这一部分是一个关于在fastNLP当中使用embedding的教程。

教程目录：

    - `Part I: embedding介绍`_
    - `Part II: 使用预训练的静态embedding`_
    - `Part III: 使用随机初始化的embedding`_
    - `Part IV: ELMo Embedding`_
    - `Part V: Bert Embedding`_
    - `Part VI: 使用character-level的embedding`_
    - `Part VII: 叠加使用多个embedding`_
    - `Part VIII: Embedding的其它说明`_
    - `Part IX: StaticEmbedding的使用建议`_



Part I: embedding介绍
---------------------------------------

Embedding是一种词嵌入技术，可以将字或者词转换为实向量。目前使用较多的预训练词嵌入有word2vec, fasttext, glove, character embedding,
elmo以及bert。
但使用这些词嵌入方式的时候都需要做一些加载上的处理，比如预训练的word2vec, fasttext以及glove都有着超过几十万个词语的表示，但一般任务大概
只会用到其中的几万个词，如果直接加载所有的词汇，会导致内存占用变大以及训练速度变慢，需要从预训练文件中抽取本次实验的用到的词汇；而对于英文的
elmo和character embedding, 需要将word拆分成character才能使用；Bert的使用更是涉及到了Byte pair encoding(BPE)相关的内容。为了方便
大家的使用，fastNLP通过 :class:`~fastNLP.Vocabulary` 统一了不同embedding的使用。下面我们将讲述一些例子来说明一下



Part II: 使用预训练的静态embedding
---------------------------------------

在fastNLP中，加载预训练的word2vec, glove以及fasttext都使用的是 :class:`~fastNLP.embeddings.StaticEmbedding` 。另外，为了方便大家的
使用，fastNLP提供了多种静态词向量的自动下载并缓存(默认缓存到~/.fastNLP/embeddings文件夹下)的功能，支持自动下载的预训练向量可以在
`下载文档 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_ 查看。

.. code-block:: python

    import torch
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d')

    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])  # 将文本转为index
    print(embed(words).size())  # StaticEmbedding的使用和pytorch的nn.Embedding是类似的

输出为::

    torch.Size([1, 5, 50])

fastNLP的StaticEmbedding在初始化之后，就和pytorch中的Embedding是类似的了。 :class:`~fastNLP.embeddings.StaticEmbedding` 的初始化
主要是从model_dir_or_name提供的词向量中抽取出 :class:`~fastNLP.Vocabulary` 中词语的vector。

除了可以通过使用预先提供的Embedding, :class:`~fastNLP.embeddings.StaticEmbedding` 也支持加载本地的预训练词向量，glove, word2vec以及
fasttext格式的。通过将model_dir_or_name修改为本地的embedding文件路径，即可使用本地的embedding。


Part III: 使用随机初始化的embedding
---------------------------------------

有时候需要使用随机初始化的Embedding，也可以通过使用 :class:`~fastNLP.embeddings.StaticEmbedding` 获得。只需要将model_dir_or_name
置为None，且传入embedding_dim，如下例所示

.. code-block:: python

    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=30)

    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 30])



Part IV: ELMo Embedding
-----------------------------------------------------------

在fastNLP中，我们提供了ELMo和BERT的embedding： :class:`~fastNLP.embeddings.ElmoEmbedding`
和 :class:`~fastNLP.embeddings.BertEmbedding` 。可自动下载的ElmoEmbedding可以
从 `下载文档 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_ 找到。

与静态embedding类似，ELMo的使用方法如下：

.. code-block:: python

    from fastNLP.embeddings import ElmoEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    embed = ElmoEmbedding(vocab, model_dir_or_name='en-small', requires_grad=False)
    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 256])

也可以输出多层的ELMo结果，fastNLP将在不同层的结果在最后一维上拼接，下面的代码需要在上面的代码执行结束之后执行

.. code-block:: python

    embed = ElmoEmbedding(vocab, model_dir_or_name='en-small', requires_grad=False, layers='1,2')
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 512])

另外，根据 `Deep contextualized word representations <https://arxiv.org/abs/1802.05365>`_ ，不同层之间使用可学习的权重可以使得ELMo的效果更好，在fastNLP中可以通过以下的初始化
实现3层输出的结果通过可学习的权重进行加法融合。

.. code-block:: python

    embed = ElmoEmbedding(vocab, model_dir_or_name='en-small', requires_grad=True, layers='mix')
    print(embed(words).size())  # 三层输出按照权重element-wise的加起来

输出为::

    torch.Size([1, 5, 256])



Part V: Bert Embedding
-----------------------------------------------------------

虽然Bert并不算严格意义上的Embedding，但通过将Bert封装成Embedding的形式将极大减轻使用的复杂程度。可自动下载的Bert Embedding可以
从 `下载文档 <https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?c=A1A0A0>`_ 找到。我们将使用下面的例子讲述一下
BertEmbedding的使用

.. code-block:: python

    from fastNLP.embeddings import BertEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased')
    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 768])

可以通过申明使用指定层数的output也可以使用多层的output，下面的代码需要在上面的代码执行结束之后执行

.. code-block:: python

    #  使用后面两层的输出
    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='10,11')
    print(embed(words).size())  # 结果将是在最后一维做拼接

输出为::

    torch.Size([1, 5, 1536])

在Bert中还存在两个特殊的字符[CLS]和[SEP]，默认情况下这两个字符是自动加入并且在计算结束之后会自动删除，以使得输入的序列长度和输出的序列
长度是一致的，但是有些分类的情况，必须需要使用[CLS]的表示，这种情况可以通过在初始化时申明一下需要保留[CLS]的表示，如下例所示

.. code-block:: python

    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', include_cls_sep=True)
    print(embed(words).size())  # 结果将在序列维度上增加2
    # 取出句子的cls表示
    cls_reps = embed(words)[:, 0]  # shape: [batch_size, 768]

输出为::

    torch.Size([1, 7, 768])

在英文Bert模型中，一个英文单词可能会被切分为多个subword，例如"fairness"会被拆分为 ``["fair", "##ness"]`` ，这样一个word对应的将有两个输出，
:class:`~fastNLP.embeddings.BertEmbedding` 会使用pooling方法将一个word的subword的表示合并成一个vector，通过pool_method可以控制
该pooling方法，支持的有"first"(即使用fair的表示作为fairness的表示), "last"(使用##ness的表示作为fairness的表示), "max"(对fair和
##ness在每一维上做max),"avg"(对fair和##ness每一维做average)。

.. code-block:: python

    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', pool_method='max')
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 768])

另外，根据 `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ ，
Bert在针对具有两句话的任务时（如matching，Q&A任务），句子之间通过[SEP]拼接起来，前一句话的token embedding为0，
后一句话的token embedding为1。BertEmbedding能够自动识别句子中间的[SEP]来正确设置对应的token_type_id的。

.. code-block:: python

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo . [SEP] another sentence .".split())

    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', pool_method='max')
    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo . [SEP] another sentence .".split()]])
    print(embed(words).size())

输出为::

    torch.Size([1, 9, 768])

在多个[SEP]的情况下，将会使token_type_id不断0，1循环。比如"first sentence [SEP] second sentence [SEP] third sentence", 它们的
token_type_id将是[0, 0, 0, 1, 1, 1, 0, 0]。但请注意[SEP]一定要大写的，不能是[sep]，否则无法识别。

更多 :class:`~fastNLP.embedding.BertEmbedding` 的使用，请参考 :doc:`/tutorials/extend_1_bert_embedding`


Part VI: 使用character-level的embedding
-----------------------------------------------------

除了预训练的embedding以外，fastNLP还提供了两种Character Embedding： :class:`~fastNLP.embeddings.CNNCharEmbedding` 和
:class:`~fastNLP.embeddings.LSTMCharEmbedding` 。一般在使用character embedding时，需要在预处理的时候将word拆分成character，这
会使得预处理过程变得非常繁琐。在fastNLP中，使用character embedding也只需要传入 :class:`~fastNLP.Vocabulary` 即可，而且该
Vocabulary与其它Embedding使用的Vocabulary是一致的，下面我们看两个例子。

CNNCharEmbedding的使用例子如下：

.. code-block:: python

    from fastNLP.embeddings import CNNCharEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    # character的embedding维度大小为50，返回的embedding结果维度大小为64。
    embed = CNNCharEmbedding(vocab, embed_size=64, char_emb_size=50)
    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 64])

与CNNCharEmbedding类似，LSTMCharEmbedding的使用例子如下：

.. code-block:: python

    from fastNLP.embeddings import LSTMCharEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    # character的embedding维度大小为50，返回的embedding结果维度大小为64。
    embed = LSTMCharEmbedding(vocab, embed_size=64, char_emb_size=50)
    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])
    print(embed(words).size())

输出为::

    torch.Size([1, 5, 64])


Part VII: 叠加使用多个embedding
-----------------------------------------------------

单独使用Character Embedding往往效果并不是很好，需要同时结合word embedding。在fastNLP中可以通过 :class:`~fastNLP.embeddings.StackEmbedding`
来叠加embedding，具体的例子如下所示

.. code-block:: python

    from fastNLP.embeddings import StaticEmbedding, StackEmbedding, CNNCharEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    word_embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d')
    char_embed = CNNCharEmbedding(vocab, embed_size=64, char_emb_size=50)
    embed = StackEmbedding([word_embed, char_embed])

    words = torch.LongTensor([[vocab.to_index(word) for word in "this is a demo .".split()]])
    print(embed(words).size())  # 输出embedding的维度为50+64=114

输出为::

    torch.Size([1, 5, 114])

:class:`~fastNLP.embeddings.StaticEmbedding` , :class:`~fastNLP.embeddings.ElmoEmbedding` ,
:class:`~fastNLP.embeddings.CNNCharEmbedding` , :class:`~fastNLP.embeddings.BertEmbedding` 等都可以互相拼接。
:class:`~fastNLP.embeddings.StackEmbedding` 的使用也是和其它Embedding是一致的，即输出index返回对应的表示。但能够拼接起来的Embedding
必须使用同样的 :class:`~fastNLP.Vocabulary` ，因为只有使用同样的 :class:`~fastNLP.Vocabulary` 才能保证同一个index指向的是同一个词或字



Part VIII: Embedding的其它说明
-----------------------------------------------------------

(1) 获取各种Embedding的dimension

.. code-block:: python

    from fastNLP.embeddings import *

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    static_embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d')
    print(static_embed.embedding_dim)  # 50
    char_embed = CNNCharEmbedding(vocab, embed_size=30)
    print(char_embed.embedding_dim)    # 30
    elmo_embed_1 = ElmoEmbedding(vocab, model_dir_or_name='en-small', layers='2')
    print(elmo_embed_1.embedding_dim)  # 256
    elmo_embed_2 = ElmoEmbedding(vocab, model_dir_or_name='en-small', layers='1,2')
    print(elmo_embed_2.embedding_dim)  # 512
    bert_embed_1 = BertEmbedding(vocab, layers='-1', model_dir_or_name='en-base-cased')
    print(bert_embed_1.embedding_dim)  # 768
    bert_embed_2 = BertEmbedding(vocab, layers='2,-1', model_dir_or_name='en-base-cased')
    print(bert_embed_2.embedding_dim)  # 1536
    stack_embed = StackEmbedding([static_embed, char_embed])
    print(stack_embed.embedding_dim)  # 80

(2) 设置Embedding的权重是否更新

.. code-block:: python

    from fastNLP.embeddings import *

    vocab = Vocabulary()
    vocab.add_word_lst("this is a demo .".split())

    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', requires_grad=True)  # 初始化时设定为需要更新
    embed.requires_grad = False  # 修改BertEmbedding的权重为不更新

(3) 各种Embedding中word_dropout与dropout的说明

fastNLP中所有的Embedding都支持传入word_dropout和dropout参数，word_dropout指示的是以多大概率将输入的word置为unk的index，这样既可以
是的unk得到训练，也可以有一定的regularize效果; dropout参数是在获取到word的表示之后，以多大概率将一些维度的表示置为0。

如果使用 :class:`~fastNLP.embeddings.StackEmbedding` 且需要用到word_dropout，建议将word_dropout设置在 :class:`~fastNLP.embeddings.StackEmbedding` 上。



Part IX: StaticEmbedding的使用建议
-----------------------------------------------------------

在英文的命名实体识别(NER)任务中，由 `Named Entity Recognition with Bidirectional LSTM-CNNs <http://xxx.itp.ac.cn/pdf/1511.08308.pdf>`_ 指出，同时使用cnn character embedding和word embedding
会使得NER的效果有比较大的提升。正如你在上节中看到的那样，fastNLP支持将 :class:`~fastNLP.embeddings.CNNCharEmbedding`
与 :class:`~fastNLP.embeddings.StaticEmbedding` 拼成一个 :class:`~fastNLP.embeddings.StackEmbedding` 。如果通过这种方式使用，需要
在预处理文本时，不要将词汇小写化(因为Character Embedding需要利用词语中的大小写信息)且不要将出现频次低于某个阈值的word设置为unk(因为
Character embedding需要利用字形信息)；但 :class:`~fastNLP.embeddings.StaticEmbedding` 使用的某些预训练词嵌入的词汇表中只有小写的词
语, 且某些低频词并未在预训练中出现需要被剔除。即(1) character embedding需要保留大小写，而预训练词向量不需要保留大小写。(2)
character embedding需要保留所有的字形, 而static embedding需要设置一个最低阈值以学到更好的表示。

(1) fastNLP如何解决关于大小写的问题

fastNLP通过在 :class:`~fastNLP.embeddings.StaticEmbedding` 增加了一个lower参数解决该问题。如下面的例子所示

.. code-block:: python

    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary().add_word_lst("The the a A".split())
    #  下面用随机的StaticEmbedding演示，但与使用预训练词向量时效果是一致的
    embed = StaticEmbedding(vocab, model_name_or_dir=None, embedding_dim=5)
    print(embed(torch.LongTensor([vocab.to_index('The')])))
    print(embed(torch.LongTensor([vocab.to_index('the')])))

输出为::

    tensor([[-0.4685,  0.4572,  0.5159, -0.2618, -0.6871]], grad_fn=<EmbeddingBackward>)
    tensor([[ 0.2615,  0.1490, -0.2491,  0.4009, -0.3842]], grad_fn=<EmbeddingBackward>)

可以看到"The"与"the"的vector是不一致的。但如果我们在初始化 :class:`~fastNLP.embeddings.StaticEmbedding` 将lower设置为True，效果将
如下所示

.. code-block:: python

    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary().add_word_lst("The the a A".split())
    #  下面用随机的StaticEmbedding演示，但与使用预训练时效果是一致的
    embed = StaticEmbedding(vocab, model_name_or_dir=None, embedding_dim=5, lower=True)
    print(embed(torch.LongTensor([vocab.to_index('The')])))
    print(embed(torch.LongTensor([vocab.to_index('the')])))

输出为::

    tensor([[-0.2237,  0.6825, -0.3459, -0.1795,  0.7516]], grad_fn=<EmbeddingBackward>)
    tensor([[-0.2237,  0.6825, -0.3459, -0.1795,  0.7516]], grad_fn=<EmbeddingBackward>)

可以看到"The"与"the"的vector是一致的。他们实际上也是引用的同一个vector。通过将lower设置为True，可以在 :class:`~fastNLP.embeddings.StaticEmbedding`
实现类似具备相同小写结果的词语引用同一个vector。

(2) fastNLP如何解决min_freq的问题

fastNLP通过在 :class:`~fastNLP.embeddings.StaticEmbedding` 增加了一个min_freq参数解决该问题。如下面的例子所示

.. code-block:: python

    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary().add_word_lst("the the the a".split())
    #  下面用随机的StaticEmbedding演示，但与使用预训练时效果是一致的
    embed = StaticEmbedding(vocab, model_name_or_dir=None, embedding_dim=5, min_freq=2)
    print(embed(torch.LongTensor([vocab.to_index('the')])))
    print(embed(torch.LongTensor([vocab.to_index('a')])))
    print(embed(torch.LongTensor([vocab.unknown_idx])))

输出为::

    tensor([[ 0.0454,  0.3375,  0.6758, -0.2026, -0.4715]], grad_fn=<EmbeddingBackward>)
    tensor([[-0.7602,  0.0149,  0.2733,  0.3974,  0.7371]], grad_fn=<EmbeddingBackward>)
    tensor([[-0.7602,  0.0149,  0.2733,  0.3974,  0.7371]], grad_fn=<EmbeddingBackward>)

其中最后一行为unknown值的vector，可以看到a的vector表示与unknown是一样的，这是由于a的频次低于了2，所以被指向了unknown的表示；而the由于
词频超过了2次，所以它是单独的表示。

在计算min_freq时，也会考虑到lower的作用，比如

.. code-block:: python

    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary().add_word_lst("the the the a A".split())
    #  下面用随机的StaticEmbedding演示，但与使用预训练时效果是一致的
    embed = StaticEmbedding(vocab, model_name_or_dir=None, embedding_dim=5, min_freq=2, lower=True)
    print(embed(torch.LongTensor([vocab.to_index('the')])))
    print(embed(torch.LongTensor([vocab.to_index('a')])))
    print(embed(torch.LongTensor([vocab.to_index('A')])))
    print(embed(torch.LongTensor([vocab.unknown_idx])))

输出为::

    tensor([[-0.7453, -0.5542,  0.5039,  0.6195, -0.4723]], grad_fn=<EmbeddingBackward>)  # the
    tensor([[ 0.0170, -0.0995, -0.5743, -0.2469, -0.2095]], grad_fn=<EmbeddingBackward>)  # a
    tensor([[ 0.0170, -0.0995, -0.5743, -0.2469, -0.2095]], grad_fn=<EmbeddingBackward>)  # A
    tensor([[ 0.6707, -0.5786, -0.6967,  0.0111,  0.1209]], grad_fn=<EmbeddingBackward>)  # unk

可以看到a不再和最后一行的unknown共享一个表示了，这是由于a与A都算入了a的词频，且A的表示也是a的表示。


----------------------------------
代码下载
----------------------------------

`点击下载 IPython Notebook 文件 <https://sourcegraph.com/github.com/fastnlp/fastNLP@master/-/raw/tutorials/tutorial_3_embedding.ipynb>`_)