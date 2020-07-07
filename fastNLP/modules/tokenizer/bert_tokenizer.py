r"""

"""

__all__ = [
    'BertTokenizer'
]

import os
import collections
import unicodedata
from ...core import logger
from ...io.file_utils import _get_file_name_base_on_postfix
from ...io.file_utils import _get_bert_dir

VOCAB_NAME = 'vocab.txt'

PRETRAINED_INIT_CONFIGURATION = {
    "en": {"do_lower_case": False},
    "en-base-uncased": {'do_lower_case': True},
    'en-base-cased': {'do_lower_case':False},
    "en-large-cased-wwm": {"do_lower_case": False},
    'en-large-cased': {'do_lower_case':False},
    'en-large-uncased': {'do_lower_case':True},
    'en-large-uncased-wwm': {'do_lower_case':True},
    'cn': {'do_lower_case':True},
    'cn-base': {'do_lower_case': True},
    'cn-wwm-ext': {'do_lower_case': True},
    'multi-base-cased': {'do_lower_case': False},
    'multi-base-uncased': {'do_lower_case': True},
}

def _is_control(char):
    r"""Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    r"""Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (((cp >= 33) and (cp <= 47)) or ((cp >= 58) and (cp <= 64)) or
       ((cp >= 91) and (cp <= 96)) or ((cp >= 123) and (cp <= 126))):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_whitespace(char):
    r"""Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def whitespace_tokenize(text):
    r"""Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    r"""Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        r"""Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        r"""Tokenizes a piece of text."""
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
        r"""Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        r"""Splits punctuation on a piece of text."""
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
        r"""Adds whitespace around any CJK character."""
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
        r"""Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (((cp >= 0x4E00) and (cp <= 0x9FFF)) or  #
            ((cp >= 0x3400) and (cp <= 0x4DBF)) or  #
            ((cp >= 0x20000) and (cp <= 0x2A6DF)) or  #
            ((cp >= 0x2A700) and (cp <= 0x2B73F)) or  #
            ((cp >= 0x2B740) and (cp <= 0x2B81F)) or  #
            ((cp >= 0x2B820) and (cp <= 0x2CEAF)) or
            ((cp >= 0xF900) and (cp <= 0xFAFF)) or  #
            ((cp >= 0x2F800) and (cp <= 0x2FA1F))):  #
            return True

        return False

    def _clean_text(self, text):
        r"""Performs invalid character removal and whitespace cleanup on text."""
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


def load_vocab(vocab_file):
    r"""Loads a vocabulary file into a dictionary."""
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


class WordpieceTokenizer(object):
    r"""Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        r"""Tokenizes a piece of text into its word pieces.

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
        if len(output_tokens) == 0:  # 防止里面全是空格或者回车符号
            return [self.unk_token]
        return output_tokens


class BertTokenizer(object):
    r"""Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        r"""Constructs a BertTokenizer.

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

    @property
    def unk_index(self):
        return self.vocab['[UNK]']

    @property
    def pad_index(self):
        return self.vocab['[PAD]']

    @property
    def cls_index(self):
        return self.vocab['[CLS]']

    @property
    def sep_index(self):
        return self.vocab['[SEP]']

    def _reinit_on_new_vocab(self, vocab):
        r"""
        在load bert之后，可能会对vocab进行重新排列。重新排列之后调用这个函数重新初始化与vocab相关的性质

        :param vocab:
        :return:
        """
        self.vocab = vocab
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

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
        r"""Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        r"""将token ids转换为一句话"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return self._convert_tokens_to_string(tokens)

    def _convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def save_vocabulary(self, vocab_path):
        r"""Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    def save_pretrained(self, save_directory):
        self.save_vocabulary(save_directory)

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        r"""
        给定模型的名字或者路径，直接读取vocab.
        """
        model_dir = _get_bert_dir(model_dir_or_name)
        pretrained_model_name_or_path = _get_file_name_base_on_postfix(model_dir, '.txt')
        logger.info("loading vocabulary file {}".format(pretrained_model_name_or_path))
        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_position_embeddings', int(1e12)), max_len)
        # Instantiate tokenizer.
        if 'do_lower_case' not in kwargs:
            if model_dir_or_name in PRETRAINED_INIT_CONFIGURATION:
                kwargs['do_lower_case'] = PRETRAINED_INIT_CONFIGURATION[model_dir_or_name]['do_lower_case']
            else:
                if 'case' in model_dir_or_name:
                    kwargs['do_lower_case'] = False
                elif 'uncase' in model_dir_or_name:
                    kwargs['do_lower_case'] = True

        tokenizer = cls(pretrained_model_name_or_path, *inputs, **kwargs)
        return tokenizer

    def encode(self, text, add_special_tokens=True):
        """
        给定text输入将数据encode为index的形式。

        Example::

            >>> from fastNLP.modules import BertTokenizer
            >>> bert_tokenizer = BertTokenizer.from_pretrained('en')
            >>> print(bert_tokenizer.encode('from'))
            >>> print(bert_tokenizer.encode("This is a demo sentence"))
            >>> print(bert_tokenizer.encode(["This", "is", 'a']))


        :param List[str],str text: 输入的一条认为是一句话。
        :param bool add_special_tokens: 是否保证句首和句尾是cls和sep。
        :return:
        """

        word_pieces = []
        if isinstance(text, str):
            words = text.split()
        elif isinstance(text, list):
            words = text
        else:
            raise TypeError("Only support str or List[str]")
        for word in words:
            _words = self.basic_tokenizer._tokenize_chinese_chars(word).split()
            tokens = []
            for word in _words:
                tokens.extend(self.wordpiece_tokenizer.tokenize(word))
            word_piece_ids = self.convert_tokens_to_ids(tokens)
            word_pieces.extend(word_piece_ids)
        if add_special_tokens:
            if word_pieces[0] != self.cls_index:
                word_pieces.insert(0, self.cls_index)
            if word_pieces[-1] != self.sep_index:
                word_pieces.append(self.sep_index)
        return word_pieces
