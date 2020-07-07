r"""undocumented
这个页面的代码很大程度上参考(复制粘贴)了https://github.com/huggingface/pytorch-pretrained-BERT的代码， 如果你发现该代码对你
    有用，也请引用一下他们。
"""

__all__ = [
    'GPT2Tokenizer'
]

from functools import lru_cache
import json
import regex as re
import itertools


from ...io.file_utils import _get_gpt2_dir
from ...core import logger
from fastNLP.io.file_utils import _get_file_name_base_on_postfix

import os

PRETRAINED_GPT2_MODEL_DIR = PRETRAINED_BERT_MODEL_DIR = {
    'en-small': 'gpt2-small.zip',
    'en-median': 'gpt2-medium.zip',
    'en': 'gpt2-medium.zip'
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "en-small": 1024,
    'en': 1024,
    "en-medium": 1024,
    "en-large": 1024,
    "en-xl": 1024,
    "en-distilgpt2": 1024,
}

PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def gpt2_tokenize(text, add_prefix_space=True):
    """

    :param str text:
    :param bool add_prefix_space: 是否在句子前面加上space，如果加上才能保证与GPT2训练时一致
    :return: []
    """
    if text is '':
        return []
    if add_prefix_space:
        text = ' ' + text
    tokens = []
    for token in re.findall(PATTERN, text):
        tokens.append(token)
    return tokens


class GPT2Tokenizer:
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding and tokenize methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer's ``encode``, ``decode``, and ``tokenize`` methods will not conserve
          the spaces at the beginning of a string: `tokenizer.decode(tokenizer.encode(" Hello")) = "Hello"`
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "sep_token",
    ]

    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs
    ):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.max_len = int(1e12)
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}
        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                else:
                    assert isinstance(value, str)
                setattr(self, key, value)

        self.max_len_single_sentence = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

    def _reinit_on_new_vocab(self, vocab):
        self.encoder = {k:v for k,v in vocab.items()}
        self.decoder = {v:k for k,v in vocab.items()}
        self.cache = {}

    @property
    def bos_token(self):
        """ Beginning of sentence token (string). Log an error if used while not having been set. """
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token (string). Log an error if used while not having been set. """
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def pad_token(self):
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        """ Classification token (string). E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def sep_token(self):
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @property
    def bos_index(self):
        """ Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def sep_index(self):
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def eos_index(self):
        """ Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_index(self):
        """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def pad_index(self):
        """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_index(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_index(self):
        """ Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        # 如果token没有找到，会被拆分成字母返回
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)  # 如果word是abcd，则((a,b), (b,c), (c, d), (e,f))

        if not pairs:
            return token

        while True:
            # 首先找到最常的pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])  #最先找的
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        bpe_tokens = []
        for token in gpt2_tokenize(text, add_prefix_space=add_prefix_space):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_pretrained(self, save_directory):
        return self.save_vocabulary(save_directory)

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        r"""
        """
        return cls._from_pretrained(model_dir_or_name)

    # 将它修改一定传入文件夹
    @classmethod
    def _from_pretrained(cls, model_dir_or_name):
        """

        :param str model_dir_or_name: 目录或者缩写名
        :param init_inputs:
        :param kwargs:
        :return:
        """
        # 它需要两个文件，第一个是vocab.json，第二个是merge_file?
        model_dir = _get_gpt2_dir(model_dir_or_name)
        # 里面会包含四个文件vocab.json, merge.txt, config.json, model.bin

        tokenizer_config_file = _get_file_name_base_on_postfix(model_dir, 'config.json')
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
        if 'max_len' not in init_kwargs:
            init_kwargs['max_len'] = 1024
        # Set max length if needed
        if model_dir_or_name in PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[model_dir_or_name]
            if max_len is not None and isinstance(max_len, (int, float)):
                init_kwargs["max_len"] = min(init_kwargs.get("max_len", int(1e12)), max_len)

        # 将vocab, merge加入到init_kwargs中
        init_kwargs['vocab_file'] = _get_file_name_base_on_postfix(model_dir, 'vocab.json')
        init_kwargs['merges_file'] = _get_file_name_base_on_postfix(model_dir, 'merges.txt')

        init_inputs = init_kwargs.pop("init_inputs", ())
        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        return tokenizer

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)

    def tokenize(self, text, add_prefix_space=True):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
            Args:
                - text: The sequence to be encoded.
                - add_prefix_space (boolean, default True):
                    Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        all_special_tokens = self.all_special_tokens

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r'(' + r'|'.join(escaped_special_toks) + r')|' + \
                      r'(.+?)'
            return re.sub(
                pattern,
                lambda m: m.groups()[0] or m.groups()[1].lower(),
                t)

        if self.init_kwargs.get('do_lower_case', False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text, add_prefix_space=add_prefix_space)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.added_tokens_encoder \
                            and sub_text not in all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(itertools.chain.from_iterable((self._tokenize(token, add_prefix_space=add_prefix_space) if token not \
                                                                                          in self.added_tokens_encoder and token not in all_special_tokens \
                                                           else [token] for token in tokenized_text)))

        added_tokens = list(self.added_tokens_encoder.keys()) + all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_id_to_tokens(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    @property
    def special_tokens_map(self):
        """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string):
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def encode(self, text, add_special_tokens=False, add_prefix_space=True):
        """
        给定text输入将数据encode为index的形式。

        Example::

            >>> from fastNLP.modules import GPT2Tokenizer
            >>> gpt2_tokenizer = GPT2Tokenizer.from_pretrained('en')
            >>> print(gpt2_tokenizer.encode('from'))
            >>> print(gpt2_tokenizer.encode("This is a demo sentence"))
            >>> print(gpt2_tokenizer.encode(["This", "is", 'a']))


        :param List[str],str text: 输入的一条认为是一句话。
        :param bool add_special_tokens: 是否保证句首和句尾是cls和sep。GPT2没有cls和sep这一说
        :return:
        """
        if isinstance(text, str):
            words = text.split()
        elif isinstance(text, list):
            words = text
        else:
            raise TypeError("Only support str or List[str]")

        word_pieces = []
        for word in words:
            tokens = self.tokenize(word, add_prefix_space=add_prefix_space)
            word_piece_ids = self.convert_tokens_to_ids(tokens)
            word_pieces.extend(word_piece_ids)
        if add_special_tokens:
            if self._cls_token is not None and word_pieces[0] != self.cls_index:
                word_pieces.insert(0, self.cls_index)
            if self._sep_token is not None and word_pieces[-1] != self.sep_index:
                word_pieces.append(self.eos_index)
        return word_pieces

    def get_used_merge_pair_vocab(self, token):
        # 如果token没有找到，会被拆分成字母返回  TODO need comment
        used_pairs = {}
        word = tuple(token)
        pairs = get_pairs(word)  # 如果word是abcd，则((a,b), (b,c), (c, d), (e,f))

        if not pairs:
            return token, used_pairs

        while True:
            # 首先找到最常的pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            used_pairs[bigram] = self.bpe_ranks[bigram]
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])  #最先找的
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        return word, used_pairs