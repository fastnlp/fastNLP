r"""
.. todo::
    doc
"""
import numpy as np
from ...envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch import nn as nn

from ...core.vocabulary import Vocabulary

__all__ = [
    'get_embeddings',
    'get_sinusoid_encoding_table'
]


def _construct_char_vocab_from_vocab(vocab: Vocabulary, min_freq: int = 1, include_word_start_end=True):
    r"""
    给定一个word的vocabulary生成character的vocabulary.

    :param vocab: 从vocab
    :param min_freq:
    :param include_word_start_end: 是否需要包含特殊的<bow>和<eos>
    :return:
    """
    char_vocab = Vocabulary(min_freq=min_freq)
    for word, index in vocab:
        if not vocab._is_word_no_create_entry(word):
            char_vocab.add_word_lst(list(word))
    if include_word_start_end:
        char_vocab.add_word_lst(['<bow>', '<eow>'])
    return char_vocab


def get_embeddings(init_embed, padding_idx=None):
    r"""
    根据输入的 ``init_embed`` 返回 ``Embedding`` 对象。

    :param init_embed: 支持以下几种输入类型：

            - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
            - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
            - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
            - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；

    :param padding_idx: 当传入 :class:`tuple` 时，``padding_idx`` 有效
    :return:
    """
    if isinstance(init_embed, tuple):
        res = nn.Embedding(
            num_embeddings=init_embed[0], embedding_dim=init_embed[1], padding_idx=padding_idx)
        nn.init.uniform_(res.weight.data, a=-np.sqrt(3 / res.weight.data.size(1)),
                         b=np.sqrt(3 / res.weight.data.size(1)))
    elif isinstance(init_embed, nn.Module):
        res = init_embed
    elif isinstance(init_embed, torch.Tensor):
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    elif isinstance(init_embed, np.ndarray):
        init_embed = torch.tensor(init_embed, dtype=torch.float32)
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    else:
        raise TypeError(
            'invalid init_embed type: {}'.format((type(init_embed))))
    return res


def get_sinusoid_encoding_table(n_position: int, d_hid: int, padding_idx=None) -> "torch.FloatTensor":
    """
    sinusoid 的 embedding，其中 ``position`` 的表示中，偶数维 ``(0,2,4,...)`` 是 sin，奇数 ``(1,3,5...)`` 是 cos。

    :param int n_position: 一共多少个 position
    :param int d_hid: 多少维度，需要为偶数
    :param padding_idx:
    :return: 形状为 ``[n_position, d_hid]`` 的张量
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def _check_vocab_has_same_index(vocab, other_vocab):
    """
    检查两个vocabulary是否含有相同的word idx

    :param Vocabulary vocab:
    :param Vocabulary other_vocab:
    :return:
    """
    if other_vocab != vocab:
        for word, word_ix in vocab:
            other_word_idx = other_vocab.to_index(word)
            assert other_word_idx == word_ix, f"Word {word} has different index in vocabs, {word_ix} Vs. {other_word_idx}."