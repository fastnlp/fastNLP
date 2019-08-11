
import os
import collections

from torch import nn
import torch
import numpy as np
from itertools import chain

from ..core.vocabulary import Vocabulary
from ..io.file_utils import _get_base_url, cached_path, PRETRAINED_BERT_MODEL_DIR
from ..modules.encoder.bert import _WordPieceBertModel, BertModel, BertTokenizer
from .contextual_embedding import ContextualEmbedding


class BertEmbedding(ContextualEmbedding):
    """
    别名：:class:`fastNLP.embeddings.BertEmbedding`   :class:`fastNLP.embeddings.bert_embedding.BertEmbedding`

    使用BERT对words进行编码的Embedding。建议将输入的words长度限制在430以内，而不要使用512(根据预训练模型参数，可能有变化)。这是由于
    预训练的bert模型长度限制为512个token，而因为输入的word是未进行word piece分割的(word piece的分割有BertEmbedding在输入word
    时切分)，在分割之后长度可能会超过最大长度限制。

    BertEmbedding可以支持自动下载权重，当前支持的模型有以下的几种(待补充):

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = BertEmbedding(vocab, model_dir_or_name='en-base-uncased', requires_grad=False, layers='4,-2,-1')
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5, 2304])

    :param ~fastNLP.Vocabulary vocab: 词表
    :param str model_dir_or_name: 模型所在目录或者模型的名称。当传入模型所在目录时，目录中应该包含一个词表文件(以.txt作为后缀名),
        权重文件(以.bin作为文件后缀名), 配置文件(以.json作为后缀名)。
    :param str layers: 输出embedding表示来自于哪些层，不同层的结果按照layers中的顺序在最后一维concat起来。以','隔开层数，层的序号是
        从0开始，可以以负数去索引倒数几层。
    :param str pool_method: 因为在bert中，每个word会被表示为多个word pieces, 当获取一个word的表示的时候，怎样从它的word pieces
        中计算得到它对应的表示。支持 ``last`` , ``first`` , ``avg`` , ``max``。
    :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
    :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
    :param bool include_cls_sep: bool，在bert计算句子的表示的时候，需要在前面加上[CLS]和[SEP], 是否在结果中保留这两个内容。 这样
        会使得word embedding的结果比输入的结果长两个token。如果该值为True，则在使用 :class::StackEmbedding 可能会与其它类型的
        embedding长度不匹配。
    :param bool pooled_cls: 返回的[CLS]是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取[CLS]做预测，
        一般该值为True。
    :param bool requires_grad: 是否需要gradient以更新Bert的权重。
    :param bool auto_truncate: 当句子words拆分为word pieces长度超过bert最大允许长度(一般为512), 自动截掉拆分后的超过510个
        word pieces后的内容，并将第512个word piece置为[SEP]。超过长度的部分的encode结果直接全部置零。一般仅有只使用[CLS]
        来进行分类的任务将auto_truncate置为True。
    """
    def __init__(self, vocab: Vocabulary, model_dir_or_name: str='en-base-uncased', layers: str='-1',
                 pool_method: str='first', word_dropout=0, dropout=0, include_cls_sep: bool=False,
                 pooled_cls=True, requires_grad: bool=False, auto_truncate:bool=False):
        super(BertEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        # 根据model_dir_or_name检查是否存在并下载
        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            PRETRAIN_URL = _get_base_url('bert')
            model_name = PRETRAINED_BERT_MODEL_DIR[model_dir_or_name]
            model_url = PRETRAIN_URL + model_name
            model_dir = cached_path(model_url)
            # 检查是否存在
        elif os.path.isdir(os.path.expanduser(os.path.abspath(model_dir_or_name))):
            model_dir = os.path.expanduser(os.path.abspath(model_dir_or_name))
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        self.model = _WordBertModel(model_dir=model_dir, vocab=vocab, layers=layers,
                                    pool_method=pool_method, include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls, auto_truncate=auto_truncate)

        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers)*self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
        """
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个token的表示。

        :param torch.LongTensor words: [batch_size, max_len]
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(words)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout(outputs)

    @property
    def requires_grad(self):
        """
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        
        :return:
        """
        requires_grads = set([param.requires_grad for name, param in self.named_parameters()
                             if 'word_pieces_lengths' not in name])
        if len(requires_grads) == 1:
            return requires_grads.pop()
        else:
            return None

    @requires_grad.setter
    def requires_grad(self, value):
        for name, param in self.named_parameters():
            if 'word_pieces_lengths' in name:  # 这个不能加入到requires_grad中
                continue
            param.requires_grad = value


class BertWordPieceEncoder(nn.Module):
    """
    读取bert模型，读取之后调用index_dataset方法在dataset中生成word_pieces这一列。

    :param str model_dir_or_name: 模型所在目录或者模型的名称。默认值为 ``en-base-uncased``
    :param str layers: 最终结果中的表示。以','隔开层数，可以以负数去索引倒数几层
    :param bool pooled_cls: 返回的句子开头的[CLS]是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取
        [CLS]做预测，一般该值为True。
    :param bool requires_grad: 是否需要gradient。
    """
    def __init__(self, model_dir_or_name: str='en-base-uncased', layers: str='-1',
                    pooled_cls: bool = False, requires_grad: bool=False):
        super().__init__()

        if model_dir_or_name in PRETRAINED_BERT_MODEL_DIR:
            PRETRAIN_URL = _get_base_url('bert')
            model_name = PRETRAINED_BERT_MODEL_DIR[model_dir_or_name]
            model_url = PRETRAIN_URL + model_name
            model_dir = cached_path(model_url)
            # 检查是否存在
        elif os.path.isdir(os.path.expanduser(os.path.abspath(model_dir_or_name))):
            model_dir = model_dir_or_name
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        self.model = _WordPieceBertModel(model_dir=model_dir, layers=layers, pooled_cls=pooled_cls)
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        """
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        requires_grads = set([param.requires_grad for name, param in self.named_parameters()])
        if len(requires_grads) == 1:
            return requires_grads.pop()
        else:
            return None

    @requires_grad.setter
    def requires_grad(self, value):
        for name, param in self.named_parameters():
            param.requires_grad = value

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def embedding_dim(self):
        return self._embed_size

    @property
    def num_embedding(self):
        return self.model.encoder.config.vocab_size

    def index_datasets(self, *datasets, field_name, add_cls_sep=True):
        """
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input,且将word_pieces这一列的pad value设置为了
            bert的pad value。

        :param DataSet datasets: DataSet对象
        :param str field_name: 基于哪一列的内容生成word_pieces列。这一列中每个数据应该是List[str]的形式。
        :param bool add_cls_sep: 如果首尾不是[CLS]与[SEP]会在首尾额外加入[CLS]与[SEP]。
        :return:
        """
        self.model.index_dataset(*datasets, field_name=field_name, add_cls_sep=add_cls_sep)

    def forward(self, word_pieces, token_type_ids=None):
        """
        计算words的bert embedding表示。传入的words中应该自行包含[CLS]与[SEP]的tag。

        :param words: batch_size x max_len
        :param token_type_ids: batch_size x max_len, 用于区分前一句和后一句话
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        outputs = self.model(word_pieces, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)

        return outputs


class _WordBertModel(nn.Module):
    def __init__(self, model_dir:str, vocab:Vocabulary, layers:str='-1', pool_method:str='first',
                 include_cls_sep:bool=False, pooled_cls:bool=False, auto_truncate:bool=False):
        super().__init__()

        self.tokenzier = BertTokenizer.from_pretrained(model_dir)
        self.encoder = BertModel.from_pretrained(model_dir)
        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        #  检查encoder_layer_number是否合理
        encoder_layer_number = len(self.encoder.encoder.layer)
        self.layers = list(map(int, layers.split(',')))
        for layer in self.layers:
            if layer<0:
                assert -layer<=encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer<encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a bert model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑[CLS]和[SEP]
        print("Start to generating word pieces for word.")
        # 第一步统计出需要的word_piece, 然后创建新的embed和word_piece_vocab, 然后填入值
        word_piece_dict = {'[CLS]':1, '[SEP]':1}  # 用到的word_piece以及新增的
        found_count = 0
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            word_pieces = self.tokenzier.wordpiece_tokenizer.tokenize(word)
            if len(word_pieces)==1:
                if not vocab._is_word_no_create_entry(word):  # 如果是train中的值, 但是却没有找到
                    if index!=vocab.unknown_idx and word_pieces[0]=='[UNK]': # 说明这个词不在原始的word里面
                        word_piece_dict[word] = 1  # 新增一个值
                        continue
            for word_piece in word_pieces:
                word_piece_dict[word_piece] = 1
            found_count += 1
        original_embed = self.encoder.embeddings.word_embeddings.weight.data
        # 特殊词汇要特殊处理
        embed = nn.Embedding(len(word_piece_dict), original_embed.size(1))  # 新的embed
        new_word_piece_vocab = collections.OrderedDict()
        for index, token in enumerate(['[PAD]', '[UNK]']):
            word_piece_dict.pop(token, None)
            embed.weight.data[index] = original_embed[self.tokenzier.vocab[token]]
            new_word_piece_vocab[token] = index
        for token in word_piece_dict.keys():
            if token in self.tokenzier.vocab:
                embed.weight.data[len(new_word_piece_vocab)] = original_embed[self.tokenzier.vocab[token]]
            else:
                embed.weight.data[len(new_word_piece_vocab)] = original_embed[self.tokenzier.vocab['[UNK]']]
            new_word_piece_vocab[token] = len(new_word_piece_vocab)
        self.tokenzier._reinit_on_new_vocab(new_word_piece_vocab)
        self.encoder.embeddings.word_embeddings = embed

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            word_pieces = self.tokenzier.wordpiece_tokenizer.tokenize(word)
            word_pieces = self.tokenzier.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))
        print("Found(Or seg into word pieces) {} words out of {}.".format(found_count, len(vocab)))
        self._cls_index = self.tokenzier.vocab['[CLS]']
        self._sep_index = self.tokenzier.vocab['[SEP]']
        self._pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenzier.vocab['[PAD]']  # 需要用于生成word_piece
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.word_pieces_lengths = nn.Parameter(torch.LongTensor(word_pieces_lengths), requires_grad=False)
        print("Successfully generate word pieces.")

    def forward(self, words):
        """

        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        batch_size, max_word_len = words.size()
        word_mask = words.ne(self._pad_index)
        seq_len = word_mask.sum(dim=-1)
        batch_word_pieces_length = self.word_pieces_lengths[words]  # batch_size x max_len
        word_pieces_lengths = batch_word_pieces_length.masked_fill(word_mask.eq(0), 0).sum(dim=-1)
        max_word_piece_length = word_pieces_lengths.max().item()
        real_max_word_piece_length = max_word_piece_length  # 表示没有截断的word piece的长度
        if max_word_piece_length+2>self._max_position_embeddings:
            if self.auto_truncate:
                word_pieces_lengths = word_pieces_lengths.masked_fill(word_pieces_lengths+2>self._max_position_embeddings,
                                                                      self._max_position_embeddings-2)
                max_word_piece_length = self._max_position_embeddings-2
            else:
                raise RuntimeError("After split words into word pieces, the lengths of word pieces are longer than the "
                                   f"maximum allowed sequence length:{self._max_position_embeddings} of bert.")


        # +2是由于需要加入[CLS]与[SEP]
        word_pieces = words.new_full((batch_size, max_word_piece_length+2), fill_value=self._wordpiece_pad_index)
        word_pieces[:, 0].fill_(self._cls_index)
        batch_indexes = torch.arange(batch_size).to(words)
        word_pieces[batch_indexes, word_pieces_lengths+1] = self._sep_index
        attn_masks = torch.zeros_like(word_pieces)
        # 1. 获取words的word_pieces的id，以及对应的span范围
        word_indexes = words.tolist()
        for i in range(batch_size):
            word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i]]))
            if self.auto_truncate and len(word_pieces_i)>self._max_position_embeddings-2:
                word_pieces_i = word_pieces_i[:self._max_position_embeddings-2]
            word_pieces[i, 1:len(word_pieces_i)+1] = torch.LongTensor(word_pieces_i)
            attn_masks[i, :word_pieces_lengths[i]+2].fill_(1)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=None, attention_mask=attn_masks,
                                           output_all_encoded_layers=True)
        # output_layers = [self.layers]  # len(self.layers) x batch_size x max_word_piece_length x hidden_size

        if self.include_cls_sep:
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                 bert_outputs[-1].size(-1))
            s_shift = 1
        else:
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))
            s_shift = 0
        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len
        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            if real_max_word_piece_length > max_word_piece_length:  # 如果实际上是截取出来的
                paddings = output_layer.new_zeros(batch_size,
                                                  real_max_word_piece_length-max_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            # 从word_piece collapse到word的表示
            truncate_output_layer = output_layer[:, 1:-1]  # 删除[CLS]与[SEP] batch_size x len x hidden_size
            outputs_seq_len = seq_len + s_shift
            if self.pool_method == 'first':
                for i in range(batch_size):
                    i_word_pieces_cum_length = batch_word_pieces_cum_length[i, :seq_len[i]]  # 每个word的start位置
                    outputs[l_index, i, s_shift:outputs_seq_len[i]] = truncate_output_layer[i, i_word_pieces_cum_length]  # num_layer x batch_size x len x hidden_size
            elif self.pool_method == 'last':
                for i in range(batch_size):
                    i_word_pieces_cum_length = batch_word_pieces_cum_length[i, 1:seq_len[i]+1] - 1 # 每个word的end
                    outputs[l_index, i, s_shift:outputs_seq_len[i]] = truncate_output_layer[i, i_word_pieces_cum_length]
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j+1]
                        outputs[l_index, i, j+s_shift], _ = torch.max(truncate_output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j+1]
                        outputs[l_index, i, j+s_shift] = torch.mean(truncate_output_layer[i, start:end], dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs)-1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len+s_shift] = output_layer[batch_indexes, seq_len+s_shift]
        # 3. 最终的embedding结果
        return outputs

