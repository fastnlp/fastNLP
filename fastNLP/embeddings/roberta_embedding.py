r"""
.. todo::
    doc
"""

__all__ = [
    "RobertaEmbedding",
    "RobertaWordPieceEncoder"
]


from functools import partial
import os
import json
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

from .contextual_embedding import ContextualEmbedding
from ..core import logger, Vocabulary
from ..modules.encoder.roberta import RobertaModel
from ..modules.tokenizer import RobertaTokenizer


VOCAB_NAME = 'vocab.txt'
ROBERTA_EMBED_HYPER = 'roberta_hyper.json'
ROBERTA_ENCODER_HYPER = 'roberta_hyper.json'
ROBERTA_EMBED_FOLDER = 'roberta'
ROBERTA_ENCODER_FOLDER = 'roberta'


class RobertaEmbedding(ContextualEmbedding):
    r"""
    使用RoBERTa对words进行编码的Embedding。建议将输入的words长度限制在430以内，而不要使用512(根据预训练模型参数，可能有变化)。这是由于
    预训练的bert模型长度限制为512个token，而因为输入的word是未进行word piece分割的(word piece的分割有RobertaEmbedding在输入word
    时切分)，在分割之后长度可能会超过最大长度限制。

    RobertaEmbedding可以支持自动下载权重，当前支持的模型:
        en: roberta-base
        en-large: roberta-large

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import RobertaEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = RobertaEmbedding(vocab, model_dir_or_name='en', requires_grad=False, layers='4,-2,-1')
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5, 2304])
    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False, **kwargs):
        r"""

        :param ~fastNLP.Vocabulary vocab: 词表
        :param str model_dir_or_name: 模型所在目录或者模型的名称。当传入模型所在目录时，目录中应该包含一个词表文件
            (以vocab.json作为后缀名), 权重文件(以.bin作为文件后缀名), 配置文件(以config.json作为后缀名)。
        :param str,list layers: 输出embedding表示来自于哪些层，不同层的结果按照layers中的顺序在最后一维concat起来。以','隔开层数，层的序号是
            从0开始，可以以负数去索引倒数几层。layer=0为embedding层（包括wordpiece embedding, position embedding）
        :param str pool_method: 因为在bert中，每个word会被表示为多个word pieces, 当获取一个word的表示的时候，怎样从它的word pieces
            中计算得到它对应的表示。支持 ``last`` , ``first`` , ``avg`` , ``max``。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool include_cls_sep: bool，在bert计算句子的表示的时候，需要在前面加上[CLS]和[SEP], 是否在结果中保留这两个内容。 这样
            会使得word embedding的结果比输入的结果长两个token。如果该值为True，则在使用 :class::StackEmbedding 可能会与其它类型的
            embedding长度不匹配。
        :param bool pooled_cls: 返回的<s>是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取<s>做预测，
            一般该值为True。
        :param bool requires_grad: 是否需要gradient以更新Bert的权重。
        :param bool auto_truncate: 当句子words拆分为word pieces长度超过bert最大允许长度(一般为512), 自动截掉拆分后的超过510个
            word pieces后的内容，并将第512个word piece置为</s>。超过长度的部分的encode结果直接全部置零。一般仅有只使用<s>
            来进行分类的任务将auto_truncate置为True。
        :param kwargs:
            int min_freq: 小于该次数的词会被unk代替， 默认为1
        """
        super().__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        if word_dropout > 0:
            assert vocab.unknown is not None, "When word_drop > 0, Vocabulary must contain the unknown token."

        self._word_sep_index = -100
        if '</s>' in vocab:
            self._word_sep_index = vocab['</s>']

        self._word_cls_index = -100
        if '<s>' in vocab:
            self._word_cls_index = vocab['<s>']

        min_freq = kwargs.get('min_freq', 1)
        self._min_freq = min_freq

        self.model = _RobertaWordModel(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                       pool_method=pool_method, include_cls_sep=include_cls_sep,
                                       pooled_cls=pooled_cls, auto_truncate=auto_truncate, min_freq=min_freq)
        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
        r"""
            计算words的roberta embedding表示。计算之前会在每句话的开始增加<s>在结束增加</s>, 并根据include_cls_sep判断要不要
            删除这两个token的表示。

        :param torch.LongTensor words: [batch_size, max_len]
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout(outputs)

    def drop_word(self, words):
        r"""
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                pad_mask = words.ne(self._word_pad_index)
                mask = pad_mask.__and__(mask)  # pad的位置不为unk
                if self._word_sep_index!=-100:
                    not_sep_mask = words.ne(self._word_sep_index)
                    mask = mask.__and__(not_sep_mask)
                if self._word_cls_index!=-100:
                    not_cls_mask = words.ne(self._word_cls_index)
                    mask = mask.__and__(not_cls_mask)
                words = words.masked_fill(mask, self._word_unk_index)
        return words

    def save(self, folder):
        """
        将roberta embedding保存到folder，保存之后包含三个文件vocab.txt, roberta_embed_hyper.txt, roberta_embed/,

        :param str folder: 保存地址
        :return:
        """
        os.makedirs(folder, exist_ok=True)
        self.get_word_vocab().save(os.path.join(folder, VOCAB_NAME))

        hyper = {}
        hyper['min_freq'] = self._min_freq
        hyper['layers'] = ','.join(map(str, self.model.layers))
        hyper['pool_method'] = self.model.pool_method
        hyper['dropout'] = self.dropout_layer.p
        hyper['word_dropout'] = self.word_dropout
        hyper['include_cls_sep'] = self.model.include_cls_sep
        hyper['pooled_cls'] = self.model.pooled_cls
        hyper['auto_truncate'] = self.model.auto_truncate
        hyper['requires_grad'] = bool(self.requires_grad)

        with open(os.path.join(folder, ROBERTA_EMBED_HYPER), 'w', encoding='utf-8') as f:
            json.dump(hyper, f, indent=2)

        os.makedirs(os.path.join(folder, ROBERTA_EMBED_FOLDER), exist_ok=True)
        self.model.save(os.path.join(folder, ROBERTA_EMBED_FOLDER))

    @classmethod
    def load(cls, folder):
        """
        从folder中读取数据初始化RobertaEmbedding

        :param folder:
        :return:
        """
        for name in [VOCAB_NAME, ROBERTA_EMBED_HYPER, ROBERTA_EMBED_FOLDER]:
            assert os.path.exists(os.path.join(folder, name)), f"{name} not found in {folder}."

        vocab = Vocabulary.load(os.path.join(folder, VOCAB_NAME))
        with open(os.path.join(folder, ROBERTA_EMBED_HYPER), 'r', encoding='utf-8') as f:
            hyper = json.load(f)
        model_name_or_path = os.path.join(folder, ROBERTA_EMBED_FOLDER)

        roberta = cls(vocab=vocab, model_dir_or_name=model_name_or_path, **hyper)
        return roberta


class _RobertaWordModel(nn.Module):
    def __init__(self, model_dir_or_name: str, vocab: Vocabulary, layers: str = '-1', pool_method: str = 'first',
                 include_cls_sep: bool = False, pooled_cls: bool = False, auto_truncate: bool = False, min_freq=2):
        super().__init__()

        self.tokenzier = RobertaTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = RobertaModel.from_pretrained(model_dir_or_name)
        # 由于RobertaEmbedding中设置了padding_idx为1, 且使用了非常神奇的position计算方式，所以-2
        self._max_position_embeddings = self.encoder.config.max_position_embeddings - 2
        #  检查encoder_layer_number是否合理
        encoder_layer_number = len(self.encoder.encoder.layer)

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")

        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a roberta model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                     f"a roberta model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '<pad>'
            elif index == vocab.unknown_idx:
                word = '<unk>'
            elif vocab.word_count[word]<min_freq:
                word = '<unk>'
            word_pieces = self.tokenzier.tokenize(word)
            word_pieces = self.tokenzier.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))
        self._cls_index = self.tokenzier.encoder['<s>']
        self._sep_index = self.tokenzier.encoder['</s>']
        self._word_pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenzier.encoder['<pad>']  # 需要用于生成word_piece
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.register_buffer('word_pieces_lengths', torch.LongTensor(word_pieces_lengths))
        logger.debug("Successfully generate word pieces.")

    def forward(self, words):
        r"""

        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        with torch.no_grad():
            batch_size, max_word_len = words.size()
            word_mask = words.ne(self._word_pad_index)  # 为1的地方有word
            seq_len = word_mask.sum(dim=-1)
            batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False), 0)  # batch_size x max_len
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)  # batch_size
            max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()  # 表示word piece的长度(包括padding)
            if max_word_piece_length + 2 > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths + 2 > self._max_position_embeddings, self._max_position_embeddings - 2)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            # +2是由于需要加入<s>与</s>
            word_pieces = words.new_full((batch_size, min(max_word_piece_length + 2, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            attn_masks = torch.zeros_like(word_pieces)
            # 1. 获取words的word_pieces的id，以及对应的span范围
            word_indexes = words.cpu().numpy()
            for i in range(batch_size):
                word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i, :seq_len[i]]]))
                if self.auto_truncate and len(word_pieces_i) > self._max_position_embeddings - 2:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings - 2]
                word_pieces[i, 1:word_pieces_lengths[i] + 1] = torch.LongTensor(word_pieces_i)
                attn_masks[i, :word_pieces_lengths[i] + 2].fill_(1)
            word_pieces[:, 0].fill_(self._cls_index)
            batch_indexes = torch.arange(batch_size).to(words)
            word_pieces[batch_indexes, word_pieces_lengths + 1] = self._sep_index
            token_type_ids = torch.zeros_like(word_pieces)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids,
                                                attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        # output_layers = [self.layers]  # len(self.layers) x batch_size x real_word_piece_length x hidden_size

        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                 bert_outputs[-1].size(-1))

        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))
        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len

        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()]
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, 1:seq_len.max() + 1] - 1
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if max_word_piece_length > real_word_piece_length:  # 如果实际上是截取出来的
                paddings = output_layer.new_zeros(batch_size,
                                                  max_word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            # 从word_piece collapse到word的表示
            truncate_output_layer = output_layer[:, 1:-1]  # 删除<s>与</s> batch_size x len x hidden_size
            if self.pool_method == 'first':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1) + s_shift] = tmp

            elif self.pool_method == 'last':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1) + s_shift] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift], _ = torch.max(truncate_output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift] = torch.mean(truncate_output_layer[i, start:end], dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len + s_shift] = output_layer[batch_indexes, word_pieces_lengths + s_shift]

        # 3. 最终的embedding结果
        return outputs

    def save(self, folder):
        self.tokenzier.save_pretrained(folder)
        self.encoder.save_pretrained(folder)


class RobertaWordPieceEncoder(nn.Module):
    r"""
    读取bert模型，读取之后调用index_dataset方法在dataset中生成word_pieces这一列。

    RobertaWordPieceEncoder可以支持自动下载权重，当前支持的模型:
        en: roberta-base
        en-large: roberta-large

    """

    def __init__(self, model_dir_or_name: str = 'en', layers: str = '-1', pooled_cls: bool = False,
                 word_dropout=0, dropout=0, requires_grad: bool = True, **kwargs):
        r"""

        :param str model_dir_or_name: 模型所在目录或者模型的名称。默认值为 ``en-base-uncased``
        :param str layers: 最终结果中的表示。以','隔开层数，可以以负数去索引倒数几层。layer=0为embedding层（包括wordpiece embedding,
                position embedding）
        :param bool pooled_cls: 返回的句子开头的<s>是否使用预训练中的BertPool映射一下。如果下游任务取<s>做预测，一般该值为True。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool requires_grad: 是否需要gradient。
        """
        super().__init__()

        self.model = _WordPieceRobertaModel(model_dir_or_name=model_dir_or_name, layers=layers, pooled_cls=pooled_cls)
        self._sep_index = self.model._sep_index
        self._cls_index = self.model._cls_index
        self._wordpiece_pad_index = self.model._wordpiece_pad_index
        self._wordpiece_unk_index = self.model._wordpiece_unknown_index
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad
        self.word_dropout = word_dropout
        self.dropout_layer = nn.Dropout(dropout)

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def embedding_dim(self):
        return self._embed_size

    @property
    def num_embedding(self):
        return self.model.encoder.config.vocab_size

    def index_datasets(self, *datasets, field_name, add_cls_sep=True, add_prefix_space=True):
        r"""
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input,且将word_pieces这一列的pad value设置为了
        bert的pad value。

        :param ~fastNLP.DataSet datasets: DataSet对象
        :param str field_name: 基于哪一列的内容生成word_pieces列。这一列中每个数据应该是List[str]的形式。
        :param bool add_cls_sep: 如果首尾不是<s>与</s>会在首尾额外加入<s>与</s>。
        :param bool add_prefix_spance: 是否在句首添加额外的空格，RoBERTa预训练时该值为True
        :return:
        """
        self.model.index_datasets(*datasets, field_name=field_name, add_cls_sep=add_cls_sep, add_prefix_space=add_prefix_space)

    def forward(self, word_pieces, token_type_ids=None):
        r"""
        计算words的bert embedding表示。传入的words中应该自行包含[CLS]与[SEP]的tag。

        :param words: batch_size x max_len
        :param token_type_ids: batch_size x max_len, 用于区分前一句和后一句话. 如果不传入，则自动生成(大部分情况，都不需要输入),
            第一个[SEP]及之前为0, 第二个[SEP]及到第一个[SEP]之间为1; 第三个[SEP]及到第二个[SEP]之间为0，依次往后推。
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        word_pieces = self.drop_word(word_pieces)
        outputs = self.model(word_pieces)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout_layer(outputs)

    def drop_word(self, words):
        r"""
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                not_sep_mask = words.ne(self._sep_index)
                not_cls_mask = words.ne(self._cls_index)
                replaceable_mask = not_sep_mask.__and__(not_cls_mask)
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                pad_mask = words.ne(self._wordpiece_pad_index)
                mask = pad_mask.__and__(mask).__and__(replaceable_mask)  # pad的位置不为unk
                words = words.masked_fill(mask, self._wordpiece_unk_index)
        return words

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        hyper = {}
        hyper['layers'] = ','.join(map(str, self.model.layers))
        hyper['dropout'] = self.dropout_layer.p
        hyper['word_dropout'] = self.word_dropout
        hyper['pooled_cls'] = self.model.pooled_cls
        hyper['requires_grad'] = bool(self.requires_grad)

        with open(os.path.join(folder, ROBERTA_ENCODER_HYPER), 'w', encoding='utf-8') as f:
            json.dump(hyper, f, indent=2)

        os.makedirs(os.path.join(folder, ROBERTA_ENCODER_FOLDER), exist_ok=True)
        self.model.save(os.path.join(folder, ROBERTA_ENCODER_FOLDER))
        logger.debug(f"RobertaWordPieceEncoder has been saved in {folder}")

    @classmethod
    def load(cls, folder):
        for name in [ROBERTA_ENCODER_HYPER, ROBERTA_ENCODER_FOLDER]:
            assert os.path.exists(os.path.join(folder, name)), f"{name} not found in {folder}."

        with open(os.path.join(folder, ROBERTA_ENCODER_HYPER), 'r', encoding='utf-8') as f:
            hyper = json.load(f)

        model_dir_or_name = os.path.join(os.path.join(folder, ROBERTA_ENCODER_FOLDER))

        bert_encoder = cls(model_dir_or_name=model_dir_or_name, **hyper)
        return bert_encoder


class _WordPieceRobertaModel(nn.Module):
    def __init__(self, model_dir_or_name: str, layers: str = '-1', pooled_cls: bool=False):
        super().__init__()

        self.tokenzier = RobertaTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = RobertaModel.from_pretrained(model_dir_or_name)
        #  检查encoder_layer_number是否合理
        encoder_layer_number = len(self.encoder.encoder.layer)

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")

        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a RoBERTa model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a RoBERTa model with {encoder_layer_number} layers."

        self._cls_index = self.tokenzier.encoder['<s>']
        self._sep_index = self.tokenzier.encoder['</s>']
        self._wordpiece_pad_index = self.tokenzier.encoder['<pad>']  # 需要用于生成word_piece
        self._wordpiece_unknown_index = self.tokenzier.encoder['<unk>']
        self.pooled_cls = pooled_cls

    def index_datasets(self, *datasets, field_name, add_cls_sep=True, add_prefix_space=True):
        r"""
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input。如果首尾不是
            [CLS]与[SEP]会在首尾额外加入[CLS]与[SEP], 且将word_pieces这一列的pad value设置为了bert的pad value。

        :param datasets: DataSet对象
        :param field_name: 基于哪一列index
        :param bool add_cls_sep: 是否在句首句尾添加cls和sep的index
        :param bool add_prefix_space: 是否在句子开头添加空格，预训练时RoBERTa该值为True
        :return:
        """

        encode_func = partial(self.tokenzier.encode, add_special_tokens=add_cls_sep, add_prefix_space=add_prefix_space)

        for index, dataset in enumerate(datasets):
            try:
                dataset.apply_field(encode_func, field_name=field_name, new_field_name='word_pieces',
                                    is_input=True)
                dataset.set_pad_val('word_pieces', self._wordpiece_pad_index)
            except Exception as e:
                logger.error(f"Exception happens when processing the {index} dataset.")
                raise e

    def forward(self, word_pieces):
        r"""

        :param word_pieces: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        batch_size, max_len = word_pieces.size()

        attn_masks = word_pieces.ne(self._wordpiece_pad_index)
        roberta_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=torch.zeros_like(word_pieces),
                                                attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        # output_layers = [self.layers]  # len(self.layers) x batch_size x max_word_piece_length x hidden_size
        outputs = roberta_outputs[0].new_zeros((len(self.layers), batch_size, max_len, roberta_outputs[0].size(-1)))
        for l_index, l in enumerate(self.layers):
            roberta_output = roberta_outputs[l]
            if l in (len(roberta_output)-1, -1) and self.pooled_cls:
                roberta_output[:, 0] = pooled_cls
            outputs[l_index] = roberta_output
        return outputs

    def save(self, folder):
        self.tokenzier.save_pretrained(folder)
        self.encoder.save_pretrained(folder)