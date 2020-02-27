==============================
fastNLP中的Vocabulary
==============================

:class:`~fastNLP.Vocabulary` 是包含字或词与index关系的类，用于将文本转换为index。


构建Vocabulary
-----------------------------

.. code-block:: python

    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word_lst(['复', '旦', '大', '学'])  # 加入新的字
    vocab.add_word('上海')  # `上海`会作为一个整体
    vocab.to_index('复')  # 应该会为3
    vocab.to_index('我')  # 会输出1，Vocabulary中默认pad的index为0, unk(没有找到的词)的index为1

    #  在构建target的Vocabulary时，词表中应该用不上pad和unk，可以通过以下的初始化
    vocab = Vocabulary(unknown=None, padding=None)
    vocab.add_word_lst(['positive', 'negative'])
    vocab.to_index('positive')  # 输出0
    vocab.to_index('neutral')  # 会报错，因为没有unk这种情况

除了通过以上的方式建立词表，Vocabulary还可以通过使用下面的函数直从 :class:`~fastNLP.DataSet` 中的某一列建立词表以及将该列转换为index

.. code-block:: python

    from fastNLP import Vocabulary
    from fastNLP import DataSet

    dataset = DataSet({'chars': [
                                    ['今', '天', '天', '气', '很', '好', '。'],
                                    ['被', '这', '部', '电', '影', '浪', '费', '了', '两', '个', '小', '时', '。']
                                ],
                        'target': ['neutral', 'negative']
    })

    vocab = Vocabulary()
    vocab.from_dataset(dataset, field_name='chars')
    vocab.index_dataset(dataset, field_name='chars')

    target_vocab = Vocabulary(padding=None, unknown=None)
    target_vocab.from_dataset(dataset, field_name='target')
    target_vocab.index_dataset(dataset, field_name='target')
    print(dataset)

输出内容为::

    +---------------------------------------------------+--------+
    |                       chars                       | target |
    +---------------------------------------------------+--------+
    |               [4, 2, 2, 5, 6, 7, 3]               |   0    |
    | [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 3] |   1    |
    +---------------------------------------------------+--------+


一些使用tips
-----------------------------

在通过使用from_dataset()函数在DataSet上建立词表时，将测试集和验证集放入参数no_create_entry_dataset中，如下所示

.. code-block:: python

    from fastNLP import Vocabulary
    from fastNLP import DataSet

    tr_data = DataSet({'chars': [
                                    ['今', '天', '心', '情', '很', '好', '。'],
                                    ['被', '这', '部', '电', '影', '浪', '费', '了', '两', '个', '小', '时', '。']
                                ],
                        'target': ['positive', 'negative']
    })
    dev_data = DataSet({'chars': [
                                    ['住', '宿', '条', '件', '还', '不', '错'],
                                    ['糟', '糕', '的', '天', '气', '，', '无', '法', '出', '行', '。']
                                ],
                        'target': ['positive', 'negative']
    })

    vocab = Vocabulary()
    #  将验证集或者测试集在建立词表是放入no_create_entry_dataset这个参数中。
    vocab.from_dataset(tr_data, field_name='chars', no_create_entry_dataset=[dev_data])

:class:`~fastNLP.Vocabulary` 中的 `no_create_entry` , 建议在添加来自于测试集和验证集的词的时候将该参数置为True, 或将验证集和测试集
传入 `no_create_entry_dataset` 参数。它们的意义是在接下来的模型会使用pretrain的embedding(包括glove, word2vec, elmo与bert)且会finetune的
情况下，如果仅使用来自于train的数据建立vocabulary，会导致只出现在test与dev中的词语无法充分利用到来自于预训练embedding的信息(因为他们
会被认为是unk)，所以在建立词表的时候将test与dev考虑进来会使得最终的结果更好。

通过与fastNLP中的各种Embedding配合使用，会有如下的效果，
如果一个词出现在了train中，但是没在预训练模型中，embedding会为随机初始化，且它单独的一个vector，如果finetune embedding的话，
这个词在更新之后可能会有更好的表示; 而如果这个词仅出现在了dev或test中，那么就不能为它们单独建立vector，而应该让它指向unk这个vector的
值(当unk的值更新时，这个词也使用的是更新之后的vector)。所以被认为是no_create_entry的token，将首先从预训练的词表中寻找它的表示，如
果找到了，就使用该表示; 如果没有找到，则认为该词的表示应该为unk的表示。

下面我们结合部分 :class:`~fastNLP.embeddings.StaticEmbedding` 的例子来说明下该值造成的影响，如果您对 :class:`~fastNLP.embeddings.StaticEmbedding` 不太了解，您可以先参考 :doc:`使用Embedding模块将文本转成向量 </tutorials/tutorial_3_embedding>` 部分再来阅读该部分

.. code-block:: python

    import torch
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import Vocabulary

    vocab = Vocabulary()
    vocab.add_word('train')
    vocab.add_word('only_in_train')  # 仅在train出现，但肯定在预训练词表中不存在
    vocab.add_word('test', no_create_entry=True)  # 该词只在dev或test中出现
    vocab.add_word('only_in_test', no_create_entry=True)  # 这个词在预训练的词表中找不到

    embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d')
    print(embed(torch.LongTensor([vocab.to_index('train')])))
    print(embed(torch.LongTensor([vocab.to_index('only_in_train')])))
    print(embed(torch.LongTensor([vocab.to_index('test')])))
    print(embed(torch.LongTensor([vocab.to_index('only_in_test')])))
    print(embed(torch.LongTensor([vocab.unknown_idx])))

输出结果(只截取了部分vector)::

    tensor([[ 0.9497,  0.3433,  0.8450, -0.8852, ...]], grad_fn=<EmbeddingBackward>)  # train，en-glove-6b-50d，找到了该词
    tensor([[ 0.0540, -0.0557, -0.0514, -0.1688, ...]], grad_fn=<EmbeddingBackward>)  # only_in_train，en-glove-6b-50d，使用了随机初始化
    tensor([[ 0.1318, -0.2552, -0.0679,  0.2619, ...]], grad_fn=<EmbeddingBackward>)  # test，在en-glove-6b-50d中找到了这个词
    tensor([[0., 0., 0., 0., 0., ...]], grad_fn=<EmbeddingBackward>)   # only_in_test, en-glove-6b-50d中找不到这个词，使用unk的vector
    tensor([[0., 0., 0., 0., 0., ...]], grad_fn=<EmbeddingBackward>)   # unk，使用zero初始化

首先train和test都能够从预训练中找到对应的vector，所以它们是各自的vector表示; only_in_train在预训练中找不到，StaticEmbedding为它
新建了一个entry，所以它有一个单独的vector; 而only_in_test在预训练中找不到改词，因此被指向了unk的值(fastNLP用零向量初始化unk)，与最后一行unk的
表示相同。