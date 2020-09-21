"""
.. todo::
    doc
"""

__all__ = [
    "GPT2Embedding",
    "GPT2WordPieceEncoder"
]

import warnings
from functools import partial
from itertools import chain
from collections import OrderedDict

import torch
from torch import nn
import numpy as np

from .contextual_embedding import ContextualEmbedding
from ..core import logger
from ..core.utils import _get_model_device
from ..core.vocabulary import Vocabulary
from ..io.file_utils import PRETRAINED_BERT_MODEL_DIR
from ..modules.tokenizer import GPT2Tokenizer
from ..modules.encoder.gpt2 import GPT2LMHeadModel, GPT2Model


class GPT2Embedding(ContextualEmbedding):
    """
    使用GPT2对words进行编码的Embedding。

    GPT2Embedding可以支持自动下载权重，当前支持的模型:
        en: gpt2
        en-medium: gpt2-medium

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import BertEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = GPT2Embedding(vocab, model_dir_or_name='en-small', requires_grad=False, layers='4,-2,-1')
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5, 3096])
    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en', layers: str = '-1',
                 pool_method: str = 'first', dropout=0, requires_grad: bool = True,
                 auto_truncate: bool = False, language_model: bool = False, **kwargs):
        """

        :param ~fastNLP.Vocabulary vocab: 词表
        :param str model_dir_or_name: 模型所在目录或者模型的名称。当传入模型所在目录时，目录中应该包含一个词表文件(以.txt作为后缀名),
            权重文件(以.bin作为文件后缀名), 配置文件(以.json作为后缀名)。
        :param str layers: 输出embedding表示来自于哪些层，不同层的结果按照layers中的顺序在最后一维concat起来。以','隔开层数，层的序号是
            从0开始，可以以负数去索引倒数几层。
        :param str pool_method: 因为在bert中，每个word会被表示为多个word pieces, 当获取一个word的表示的时候，怎样从它的word pieces
            中计算得到它对应的表示。支持 ``last`` , ``first`` , ``avg`` , ``max``。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool requires_grad: 是否需要gradient以更新Bert的权重。
        :param bool auto_truncate: 当句子words拆分为word pieces长度超过bert最大允许长度(一般为512), 自动截掉拆分后的超过510个
            word pieces后的内容，并将第512个word piece置为[SEP]。超过长度的部分的encode结果直接全部置零。一般仅有只使用[CLS]
            来进行分类的任务将auto_truncate置为True。
        :param bool language_model: 是否计算gpt2的lm loss，可以通过get_loss()获取，输入一个batch之后的get_loss调用即为batch的language
            model的loss
        :param **kwargs:
            bool only_use_pretrain_bpe: 仅使用出现在pretrain词表中的bpe，如果该词没法tokenize则使用unk。如果embedding不需要更新
                建议设置为True。
            int min_freq: 仅在only_use_pretrain_bpe为False有效，大于等于该次数的词会被新加入GPT2的BPE词表中
            bool truncate_embed: 是否仅保留用到的bpe(这样会减内存占用和加快速度)
        """
        super().__init__(vocab, word_dropout=0, dropout=dropout)

        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in ('first', 'last'):
                logger.warning("For Chinese GPT, pooled_method should choose from 'first', 'last' in order to achieve"
                               " faster speed.")
                warnings.warn("For Chinese GPT, pooled_method should choose from 'first', 'last' in order to achieve"
                              " faster speed.")

        only_use_pretrain_bpe = kwargs.get('only_use_pretrain_bpe', False)
        truncate_embed = kwargs.get('truncate_embed', True)
        min_freq = kwargs.get('min_freq', 1)

        self.lm_loss =language_model
        self.model = _GPT2Model(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                    pool_method=pool_method, auto_truncate=auto_truncate, language_model=language_model,
                                only_use_pretrain_bpe=only_use_pretrain_bpe, truncate_embed=truncate_embed,
                                min_freq=min_freq)

        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * self.model.encoder.config.n_embd

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
        """
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个token的表示。

        :param torch.LongTensor words: [batch_size, max_len]
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout(outputs)

    def drop_word(self, words):
        """
        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                words = words.masked_fill(mask, self._word_unk_index)
        return words

    def get_lm_loss(self, release=True):
        """
        当language_model=True时，可以通过该接口获取当前batch的language model loss的大小

        :param bool release: 如果为True，获取了lm_loss后在下一次forward完成之前都无法获取lm_loss了
        :return: torch.FloatTensor([])
        """
        if hasattr(self.model, '_lm_loss_value'):
            lm_loss_value = self.model._lm_loss_value
            if release:
                delattr(self.model, '_lm_loss_value')
            return lm_loss_value
        elif self.lm_loss:
            raise RuntimeError("Make sure you have passed a batch into GPT2Embdding before accessing loss.")
        else:
            raise RuntimeError("Initialize your GPT2Embedding with language_model=True.")


class GPT2WordPieceEncoder(nn.Module):
    """
    GPT2模型，使用时先使用本模型对应的Tokenizer对数据进行tokenize
    GPT2WordPieceEncoder可以支持自动下载权重，当前支持的模型:
        en: gpt2
        en-medium: gpt2-medium

    """

    def __init__(self, model_dir_or_name: str = 'en', layers: str = '-1',
                 word_dropout=0, dropout=0, requires_grad: bool = True, language_model:bool=False):
        """

        :param str model_dir_or_name: 模型所在目录或者模型的名称。
        :param str,list layers: 最终结果中的表示。以','隔开层数，可以以负数去索引倒数几层
        :param float word_dropout: 多大概率将word piece置为<|endoftext|>
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool language_model: 是否使用language model
        :param bool requires_grad: 是否需要gradient。
        """
        super().__init__()

        self.model = _GPT2WordPieceModel(model_dir_or_name=model_dir_or_name, layers=layers, language_model=language_model)
        self._wordpiece_pad_index = self.model._wordpiece_pad_index
        self._embed_size = len(self.model.layers) * self.model.encoder.config.n_embd
        self.requires_grad = requires_grad
        self.dropout_layer = nn.Dropout(dropout)
        self._wordpiece_endoftext_index = self.model._endoftext_index
        self.word_dropout = word_dropout
        self.language_model = language_model

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def embedding_dim(self):
        return self._embed_size

    @property
    def num_embedding(self):
        return self.model.encoder.config.vocab_size

    def index_datasets(self, *datasets, field_name, add_endoftext=False, add_prefix_space=True):
        """
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input,且将word_pieces这一列的pad value设置为了
        bert的pad value。

        :param ~fastNLP.DataSet datasets: DataSet对象
        :param list[str] field_name: 基于哪一列的内容生成word_pieces列。这一列中每个数据应该是List[str]的形式。
        :param bool add_endoftext: 在句子开头加入<|endofline|>。
        :param bool add_prefix_space: 是否在句首增加空格
        :return:
        """
        self.model.index_datasets(*datasets, field_name=field_name, add_endoftext=add_endoftext,
                                 add_prefix_space=add_prefix_space)

    def forward(self, word_pieces, token_type_ids=None):
        """
        计算words的bert embedding表示。传入的words中应该在开头包含<|endofline|>。

        :param word_pieces: batch_size x max_len
        :param token_type_ids: batch_size x max_len,
        :return: torch.FloatTensor.
        """

        outputs = self.model(word_pieces)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout_layer(outputs)

    def drop_word(self, words):
        """

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                endoftext_mask = words.ne(self._wordpiece_endoftext_index)
                mask = endoftext_mask.__and__(mask)  # pad的位置不为unk
                words = words.masked_fill(mask, self._wordpiece_unk_index)
        return words

    def generate_from_str(self, text='', max_len=40, do_sample=True, num_beams=1, temperature=1, top_k=50, top_p=1.0,
                          repetition_penalty=1.0, length_penalty=1.0):
        """

        :param str text: 故事的开头
        :param int max_len: 生成多长的句子
        :param bool do_sample: 是否使用采样的方式生成，如果使用采样，相同的参数可能出现不同的句子。
        :param int num_beams: 使用多大的beam size
        :param float temperature: 用以调节采样分布的
        :param int top_k: 只保留此表中top_k个词进行生成。范围1-infinity
        :param float top_p: 保留概率累积为top_p的词汇，范围0-1.
        :param float repetition_penalty: 对重复token的惩罚
        :param float length_penalty: 惩罚过长的句子
        :return: list[str]
        """
        if len(text)==0:
            word_pieces = torch.LongTensor([[self.model.tokenizer.bos_index]])
            start_idx = 1
        else:
            assert isinstance(text, str), "Only string input allowed."
            assert self.language_model, "You must set `language_model=True`."
            word_pieces = self.model.convert_words_to_word_pieces(text, add_prefix_space=True)
            word_pieces = torch.LongTensor([word_pieces])
            start_idx = 0
        device = _get_model_device(self)
        word_pieces = word_pieces.to(device)
        outputs = self.model.encoder.generate(input_ids=word_pieces,
                        max_length=max_len,
                        do_sample=do_sample,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        bos_token_id=self.model.tokenizer.bos_index,
                        pad_token_id=self.model.tokenizer.eos_index,  # 使用<|endoftext|>代替pad
                        eos_token_ids=self.model.tokenizer.eos_index,
                        length_penalty=length_penalty).squeeze(0)

        output_strs = []
        if outputs.dim()==1:
            outputs = outputs[None]
        outputs = outputs[:, start_idx:]
        for i in range(len(outputs)):
            str_ = self.model.tokenizer.convert_tokens_to_string(self.model.tokenizer.convert_ids_to_tokens(outputs[i].tolist()))
            output_strs.append(str_)

        return output_strs

    def generate(self, word_pieces=None, max_len=40, do_sample=True, num_beams=1, temperature=1, top_k=50, top_p=1.0,
                    repetition_penalty=1.0, length_penalty=1.0):
        """

        :param torch.LongTensor,None word_pieces: 如果传入tensor，shape应该为batch_size x start_len; 如果传入None，会随机生成。
        :param int max_len: 生成多长的句子
        :param bool do_sample: 是否使用采样的方式生成，如果使用采样，相同的参数可能出现不同的句子。
        :param int num_beams: 使用多大的beam size
        :param float temperature: 用以调节采样分布的
        :param int top_k: 只保留此表中top_k个词进行生成。范围1-infinity
        :param float top_p: 保留概率累积为top_p的词汇，范围0-1.
        :param float repetition_penalty: 对重复token的惩罚
        :param float length_penalty: 惩罚过长的句子
        :return:
        """
        raise NotImplemented

    def get_lm_loss(self, release=True):
        """
        当language_model=True时，可以通过该接口获取当前batch的language model loss的大小

        :param bool release: 如果为True，获取了lm_loss后在下一次forward完成之前都无法获取lm_loss了
        :return: torch.FloatTensor([])
        """
        if hasattr(self.model, '_lm_loss_value'):
            lm_loss_value = self.model._lm_loss_value
            if release:
                delattr(self.model, '_lm_loss_value')
            return lm_loss_value
        elif self.lm_loss:
            raise RuntimeError("Make sure you have passed a batch into GPT2Embdding before accessing loss.")
        else:
            raise RuntimeError("Initialize your GPT2Embedding with language_model=True.")


class _GPT2Model(nn.Module):
    def __init__(self, model_dir_or_name, vocab, layers,  pool_method='first', auto_truncate=True, language_model=False,
                 only_use_pretrain_bpe=False, min_freq=1, truncate_embed=False):
        super().__init__()

        self.tokenzier = GPT2Tokenizer.from_pretrained(model_dir_or_name)
        if language_model:
            self.encoder = GPT2LMHeadModel.from_pretrained(model_dir_or_name)
        else:
            self.encoder = GPT2Model.from_pretrained(model_dir_or_name)

        self.lm_loss = language_model
        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        #  检查encoder_layer_number是否合理
        encoder_layer_number = self.encoder.config.n_layer
        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a GPT2 model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                     f"a GPT2 model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.auto_truncate = auto_truncate

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑<s>和</s>
        logger.info("Start to generate word pieces for word.")
        # 第一步统计出需要的word_piece, 然后创建新的embed和word_piece_vocab, 然后填入值
        word_piece_dict = {'<|endoftext|>': 1}  # 用到的word_piece以及新增的
        found_count = 0
        new_add_to_bpe_vocab = 0
        unsegment_count = 0

        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '<|endoftext|>'
            elif index == vocab.unknown_idx:
                word = '<|endoftext|>'
            # _words = self.tokenzier.basic_tokenizer._tokenize_chinese_chars(word).split()  # 这里暂时不考虑中文内容
            word_pieces = []
            word_pieces.extend(self.tokenzier.tokenize(word, add_prefix_space=True))
            if len(word_pieces) == 1:
                if not vocab._is_word_no_create_entry(word):  # 如果是train中的值, 但是却没有找到
                    if index not in (vocab.unknown_idx, vocab.padding_idx) and word_pieces[0] == '<|endoftext|>':  # 说明这个词不在原始的word里面
                        if vocab.word_count[word] >= min_freq and not vocab._is_word_no_create_entry(
                                word) and not only_use_pretrain_bpe:  # 出现次数大于这个次数才新增
                            word_piece_dict[word] = 1  # 新增一个值
                            new_add_to_bpe_vocab += 1
                        unsegment_count += 1
                        continue
            for word_piece in word_pieces:
                word_piece_dict[word_piece] = 1
            found_count += 1

        if unsegment_count>0:
            if only_use_pretrain_bpe or new_add_to_bpe_vocab==0:
                logger.info(f"{unsegment_count} words are unsegmented.")
            else:
                logger.info(f"{unsegment_count} words are unsegmented. Among them, {new_add_to_bpe_vocab} added to the BPE vocab.")

        original_embed = self.encoder.get_input_embeddings().weight
        # 特殊词汇要特殊处理
        if not truncate_embed:  # 如果不删除的话需要将已有的加上
            word_piece_dict.update(self.tokenzier.encoder)

        embed = nn.Embedding(len(word_piece_dict), original_embed.size(1))  # 新的embed
        new_word_piece_vocab = OrderedDict()

        for index, token in enumerate(['<|endoftext|>']):
            index = word_piece_dict.pop(token, None)
            if index is not None:
                new_word_piece_vocab[token] = len(new_word_piece_vocab)
                embed.weight.data[new_word_piece_vocab[token]] = original_embed[self.tokenzier.encoder[token]]

        for token in word_piece_dict.keys():
            if token not in new_word_piece_vocab:
                new_word_piece_vocab[token] = len(new_word_piece_vocab)
            index = new_word_piece_vocab[token]
            if token in self.tokenzier.encoder:
                embed.weight.data[index] = original_embed[self.tokenzier.encoder[token]]
            else:
                embed.weight.data[index] = original_embed[self.tokenzier.encoder['<|endoftext|>']]

        self.tokenzier._reinit_on_new_vocab(new_word_piece_vocab)
        self.encoder.set_input_embeddings(embed)
        self.encoder.tie_weights()
        self.encoder.config.vocab_size = len(new_word_piece_vocab)

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '<|endoftext|>'
            elif index == vocab.unknown_idx:
                word = '<|endoftext|>'
            word_pieces = self.tokenzier.tokenize(word)
            word_pieces = self.tokenzier.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))
        self._word_pad_index = vocab.padding_idx
        self._endoftext_index = self.tokenzier.encoder.get('<|endoftext|>')
        self._wordpiece_pad_index = self.tokenzier.encoder.get('<|endoftext|>')  # 需要用于生成word_piece
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.register_buffer('word_pieces_lengths', torch.LongTensor(word_pieces_lengths))
        logger.debug("Successfully generate word pieces.")

    def forward(self, words):
        """

        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        with torch.no_grad():
            batch_size, max_word_len = words.size()
            word_mask = words.ne(self._word_pad_index)  # 为1的地方有word
            seq_len = word_mask.sum(dim=-1)
            batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False),
                                                                                   0)  # batch_size x max_len
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)  # batch_size
            max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()  # 表示word piece的长度(包括padding)
            if max_word_piece_length > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths > self._max_position_embeddings,
                        self._max_position_embeddings)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of GPT2. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            word_pieces = words.new_full((batch_size, min(max_word_piece_length, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            word_labels = word_pieces.clone()
            attn_masks = torch.zeros_like(word_pieces)
            # 1. 获取words的word_pieces的id，以及对应的span范围
            word_indexes = words.cpu().numpy()
            for i in range(batch_size):
                word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i, :seq_len[i]]]))
                if self.auto_truncate and len(word_pieces_i) > self._max_position_embeddings:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings]
                word_pieces[i, :word_pieces_lengths[i]] = torch.LongTensor(word_pieces_i)
                word_labels[i, word_pieces_lengths[i]:].fill_(-100)  # 计算lm_loss用的
                attn_masks[i, :word_pieces_lengths[i]].fill_(1)
            # 添加<|endoftext|>, 默认不添加了
            # word_pieces[:, 0].fill_(self._endoftext_index)
            batch_indexes = torch.arange(batch_size).to(words)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        if self.lm_loss:
            gpt2_outputs = self.encoder(word_pieces, token_type_ids=None, attention_mask=attn_masks, labels=word_labels,
                                        output_attentions=False)
            gpt2_outputs, self._lm_loss_value = gpt2_outputs[-1], gpt2_outputs[0]  # n_layers x batch_size x max_len x hidden_size
        else:
            gpt2_outputs = self.encoder(word_pieces, token_type_ids=None, attention_mask=attn_masks,
                                        output_attentions=False)[-1]
        outputs = gpt2_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                             gpt2_outputs[-1].size(-1))

        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len+1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len

        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()]
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()] - 1
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = gpt2_outputs[l]
            real_word_piece_length = output_layer.size(1)
            if max_word_piece_length > real_word_piece_length:  # 如果实际上是截取出来的
                paddings = output_layer.new_zeros(batch_size,
                                                  max_word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            # 从word_piece collapse到word的表示
            # truncate_output_layer = output_layer  # 删除endoftext batch_size x len x hidden_size
            if self.pool_method == 'first':
                tmp = output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, :batch_word_pieces_cum_length.size(1)] = tmp
            elif self.pool_method == 'last':
                tmp = output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, :batch_word_pieces_cum_length.size(1)] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j], _ = torch.max(output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j] = torch.mean(output_layer[i, start:end], dim=-2)

        # 3. 最终的embedding结果
        return outputs

    def get_lm_loss(self):
        """
        当language_model为True时，通过该接口可以获取最近传入的一个batch的lanuage model loss

        :return:
        """
        return self._lm_loss_value


class _GPT2WordPieceModel(nn.Module):
    """
    这个模块用于直接计算word_piece的结果.

    """

    def __init__(self, model_dir_or_name: str, layers: str = '-1', language_model: bool=False):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir_or_name)
        if language_model:
            self.encoder = GPT2LMHeadModel.from_pretrained(model_dir_or_name)
        else:
            self.encoder = GPT2Model.from_pretrained(model_dir_or_name)

        self.lm_loss = language_model

        #  检查encoder_layer_number是否合理
        encoder_layer_number = self.encoder.config.n_layer

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")

        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a gpt2 model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a gpt2 model with {encoder_layer_number} layers."

        self._endoftext_index = self.tokenizer.encoder.get('<|endoftext|>')
        self._wordpiece_pad_index = self.tokenizer.encoder.get('<|endoftext|>') # 原来并没有pad，使用这个值替代一下。这个pad值并不重要，因为是从左到右计算的
        self._max_position_embeddings = self.encoder.config.max_position_embeddings

    def index_datasets(self, *datasets, field_name, add_endoftext=False, add_prefix_space=True):
        """
        使用gpt2的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input。如果开头不是<|endoftext|>, 且将
            word_pieces这一列的pad value设置为了bert的pad value。

        :param datasets: DataSet对象
        :param field_name: 基于哪一列index
        :param bool add_prefix_space: 是否添加句首的空格
        :return:
        """
        convert_words_to_word_pieces = partial(self.convert_words_to_word_pieces, add_endoftext=add_endoftext,
                                               add_prefix_space=add_prefix_space)
        for index, dataset in enumerate(datasets):
            try:
                dataset.apply_field(convert_words_to_word_pieces, field_name=field_name, new_field_name='word_pieces',
                                    is_input=True)
                dataset.set_pad_val('word_pieces', self._wordpiece_pad_index)
            except Exception as e:
                logger.error(f"Exception happens when processing the {index} dataset.")
                raise e

    def convert_words_to_word_pieces(self, words, add_endoftext=False, add_prefix_space=True):
        """

        :param list[str],str words: 将str数据转换为index
        :param bool add_endoftext: 是否在句首增加endoftext
        :param bool add_prefix_space: 是否添加句首的空格
        :return:
        """
        word_pieces = []
        if isinstance(words, str):
            words = self.tokenizer.tokenize(words, add_prefix_space=add_prefix_space)
            word_piece_ids = self.tokenizer.convert_tokens_to_ids(words)
            word_pieces.extend(word_piece_ids)
        else:
            for word in words:
                tokens = self.tokenizer.tokenize(word, add_prefix_space=add_prefix_space)
                word_piece_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                word_pieces.extend(word_piece_ids)
        if add_endoftext:
            if word_pieces[0] != self._endoftext_index:
                word_pieces.insert(0, self._endoftext_index)
        if len(word_pieces) > self._max_position_embeddings:
            word_pieces[self._max_position_embeddings - 1] = word_pieces[-1]
            word_pieces = word_pieces[:self._max_position_embeddings]
        return word_pieces

    def forward(self, word_pieces, token_type_ids=None):
        """

        :param word_pieces: torch.LongTensor, batch_size x max_len
        :param token_type_ids: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        batch_size, max_len = word_pieces.size()

        attn_masks = word_pieces.ne(self._wordpiece_pad_index)  # 可能会错误导致开头的词被mask掉
        word_pieces = word_pieces.masked_fill(attn_masks.eq(0), self._endoftext_index)  # 替换pad的值
        if self.lm_loss:
            labels = word_pieces.clone()
            labels = labels.masked_fill(labels.eq(self._wordpiece_pad_index), -100)
            gpt_outputs = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                        output_attentions=False, labels=labels)
            gpt_outputs, self._lm_loss_value = gpt_outputs[-1], gpt_outputs[0]  # n_layers x batch_size x max_len x hidden_size
        else:
            gpt_outputs = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                        output_attentions=False)
            gpt_outputs = gpt_outputs[-1]
        # output_layers = [self.layers]  # len(self.layers) x batch_size x max_word_piece_length x hidden_size
        outputs = gpt_outputs[0].new_zeros((len(self.layers), batch_size, max_len, gpt_outputs[0].size(-1)))
        for l_index, l in enumerate(self.layers):
            outputs[l_index] = gpt_outputs[l]  # 删除开头
        return outputs

    def get_lm_loss(self):
        """
        当language_model为True时，通过该接口可以获取最近传入的一个batch的lanuage model loss

        :return:
        """
        return self._lm_loss_value

