


"""
这个页面的代码很大程度上参考了https://github.com/huggingface/pytorch-pretrained-BERT的代码
"""


import torch
from torch import nn

from ... import Vocabulary
import collections

import os
import unicodedata
from ...io.file_utils import _get_base_url, cached_path
from .bert import BertModel
import numpy as np
from itertools import chain

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
          self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            print(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    @classmethod
    def from_pretrained(cls, model_dir, *inputs, **kwargs):
        """
        给定path，直接读取vocab.

        """
        pretrained_model_name_or_path = os.path.join(model_dir, VOCAB_NAME)
        print("loading vocabulary file {}".format(pretrained_model_name_or_path))
        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(pretrained_model_name_or_path, *inputs, **kwargs)
        return tokenizer

VOCAB_NAME = 'vocab.txt'

class _WordBertModel(nn.Module):
    def __init__(self, model_dir:str, vocab:Vocabulary, layers:str='-1', pool_method:str='first', include_cls_sep:bool=False):
        super().__init__()

        self.tokenzier = BertTokenizer.from_pretrained(model_dir)
        self.encoder = BertModel.from_pretrained(model_dir)
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

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑[CLS]和[SEP]
        print("Start to generating word pieces for word.")
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
        self._cls_index = len(vocab)
        self._sep_index = len(vocab) + 1
        self._pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenzier.convert_tokens_to_ids(['[PAD]'])[0]  # 需要用于生成word_piece
        word_to_wordpieces.append(self.tokenzier.convert_tokens_to_ids(['[CLS]']))
        word_to_wordpieces.append(self.tokenzier.convert_tokens_to_ids(['[SEP]']))
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.word_pieces_lengths = nn.Parameter(torch.LongTensor(word_pieces_lengths), requires_grad=False)
        print("Successfully generate word pieces.")

    def forward(self, words):
        """

        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        batch_size, max_word_len = words.size()
        seq_len = words.ne(self._pad_index).sum(dim=-1)
        batch_word_pieces_length = self.word_pieces_lengths[words]  # batch_size x max_len
        word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)
        max_word_piece_length = word_pieces_lengths.max().item()
        # +2是由于需要加入[CLS]与[SEP]
        word_pieces = words.new_full((batch_size, max_word_piece_length+2), fill_value=self._wordpiece_pad_index)
        word_pieces[:, 0].fill_(self._cls_index)
        word_pieces[:, word_pieces_lengths+1] = self._sep_index
        attn_masks = torch.zeros_like(word_pieces)
        # 1. 获取words的word_pieces的id，以及对应的span范围
        word_indexes = words.tolist()
        for i in range(batch_size):
            word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i]]))
            word_pieces[i, 1:len(word_pieces_i)+1] = torch.LongTensor(word_pieces_i)
            attn_masks[i, :len(word_pieces_i)+2].fill_(1)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        bert_outputs, _ = self.encoder(word_pieces, token_type_ids=None, attention_mask=attn_masks,
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
                outputs[:, :, 0] = output_layer[:, 0]
                outputs[:, :, seq_len+s_shift] = output_layer[:, seq_len+s_shift]
        # 3. 最终的embedding结果
        return outputs


class _WordPieceBertModel(nn.Module):
    """
    这个模块用于直接计算word_piece的结果.

    """
    def __init__(self, model_dir:str, vocab:Vocabulary, layers:str='-1'):
        super().__init__()

        self.tokenzier = BertTokenizer.from_pretrained(model_dir)
        self.encoder = BertModel.from_pretrained(model_dir)
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

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑[CLS]和[SEP]
        print("Start to generating word pieces for word.")
        self.word_to_wordpieces = []
        self.word_pieces_length = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            word_pieces = self.tokenzier.wordpiece_tokenizer.tokenize(word)
            word_pieces = self.tokenzier.convert_tokens_to_ids(word_pieces)
            self.word_to_wordpieces.append(word_pieces)
            self.word_pieces_length.append(len(word_pieces))
        self._cls_index = len(vocab)
        self._sep_index = len(vocab) + 1
        self._pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenzier.convert_tokens_to_ids(['[PAD]'])[0]  # 需要用于生成word_piece
        self.word_to_wordpieces.append(self.tokenzier.convert_tokens_to_ids(['[CLS]']))
        self.word_to_wordpieces.append(self.tokenzier.convert_tokens_to_ids(['[SEP]']))
        self.word_to_wordpieces = np.array(self.word_to_wordpieces, dtype=int)
        print("Successfully generate word pieces.")

    def index_dataset(self, *datasets):
        """
        使用bert的tokenizer将word_pieces与word_pieces_seq_len这两列加入到datasets中，并将他们设置为input。加入的word_piece
            已经包含了[CLS]与[SEP], 且将word_pieces这一列的pad value设置为了bert的pad value。

        :param datasets: DataSet对象
        :return:
        """
        def convert_words_to_word_pieces(words):
            word_pieces = list(chain(*self.word_to_wordpieces[words].tolist()))
            word_pieces = [self._cls_index] + word_pieces + [self._sep_index]
            return word_pieces

        for index, dataset in enumerate(datasets):
            try:
                dataset.apply_field(convert_words_to_word_pieces, field_name='words', new_field_name='word_pieces',
                                    is_input=True)
                dataset.set_pad_val('word_pieces', self._wordpiece_pad_index)
            except Exception as e:
                print(f"Exception happens when processing the {index} dataset.")
                raise e

    def forward(self, word_pieces, token_type_ids=None):
        """

        :param word_pieces: torch.LongTensor, batch_size x max_len
        :param token_type_ids: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        batch_size, max_len = word_pieces.size()

        attn_masks = word_pieces.ne(self._pad_index)
        bert_outputs, _ = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                           output_all_encoded_layers=True)
        # output_layers = [self.layers]  # len(self.layers) x batch_size x max_word_piece_length x hidden_size
        outputs = bert_outputs[0].new_zeros((len(self.layers), batch_size, max_len, bert_outputs[0].size(-1)))
        for l_index, l in enumerate(self.layers):
            outputs[l_index] = bert_outputs[l]
        return outputs

class BertWordPieceEncoder(nn.Module):
    """
    可以通过读取vocabulary使用的Bert的Encoder。传入vocab，然后调用index_datasets方法在vocabulary中生成word piece的表示。

    :param vocab: Vocabulary.
    :param model_dir_or_name:
    :param layers:
    :param requires_grad:
    """
    def __init__(self, vocab:Vocabulary, model_dir_or_name:str='en-base', layers:str='-1',
                 requires_grad:bool=False):
        super().__init__()
        PRETRAIN_URL = _get_base_url('bert')
        # TODO 修改
        PRETRAINED_BERT_MODEL_DIR = {'en-base': 'bert_en-80f95ea7.tar.gz',
                                     'cn': 'elmo_cn.zip'}

        if model_dir_or_name in PRETRAINED_BERT_MODEL_DIR:
            model_name = PRETRAINED_BERT_MODEL_DIR[model_dir_or_name]
            model_url = PRETRAIN_URL + model_name
            model_dir = cached_path(model_url)
            # 检查是否存在
        elif os.path.isdir(model_dir_or_name):
            model_dir = model_dir_or_name
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        self.model = _WordPieceBertModel(model_dir=model_dir, vocab=vocab, layers=layers)
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        """
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        requires_grads = set([param.requires_grad for name, param in self.named_parameters()])
        if len(requires_grads)==1:
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

    def index_datasets(self, *datasets):
        """
        对datasets进行word piece的index。

        Example::

        :param datasets:
        :return:
        """
        self.model.index_dataset(*datasets)

    def forward(self, words, token_type_ids=None):
        """
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个表示。

        :param words: batch_size x max_len
        :param token_type_ids: batch_size x max_len, 用于区分前一句和后一句话
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        outputs = self.model(words, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)

        return outputs