
from typing import List, Optional
import json

import torch
import torch.nn as nn

from .bert import BertEmbeddings, BertModel, BertConfig, _get_bert_dir
from .gpt2 import GPT2Tokenizer
from ..utils import create_position_ids_from_input_ids, _get_file_name_base_on_postfix
from ...core import logger

PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, words_embeddings=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(words_embeddings)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, words_embeddings=words_embeddings
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaModel(BertModel):
    r"""
    undocumented
    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.apply(self.init_bert_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        kwargs.pop('cache_dir', None)
        kwargs.pop('from_tf', None)

        # get model dir from name or dir
        pretrained_model_dir = _get_bert_dir(model_dir_or_name)

        # Load config
        config_file = _get_file_name_base_on_postfix(pretrained_model_dir, 'config.json')
        config = BertConfig.from_json_file(config_file)

        # Load model
        if state_dict is None:
            weights_path = _get_file_name_base_on_postfix(pretrained_model_dir, '.bin')
            state_dict = torch.load(weights_path, map_location='cpu')
        else:
            logger.error(f'Cannot load parameters through `state_dict` variable.')
            raise RuntimeError(f'Cannot load parameters through `state_dict` variable.')

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, 'roberta') and any(
            s.startswith('roberta') for s in state_dict.keys()
        ):
            start_prefix = 'roberta.'
        if hasattr(model, 'roberta') and not any(
            s.startswith('roberta') for s in state_dict.keys()
        ):
            model_to_load = getattr(model, 'roberta')

        load(model_to_load, prefix=start_prefix)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split('roberta.')[-1] for key in model.state_dict().keys()
            ]

            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        logger.info(f"Load pre-trained RoBERTa parameters from file {weights_path}.")

        return model


class RobertaTokenizer(GPT2Tokenizer):

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        RoBERTa does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.

        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, add_special_tokens=False, **kwargs):
        if "add_prefix_space" in kwargs:
            add_prefix_space = kwargs["add_prefix_space"]
        else:
            add_prefix_space = add_special_tokens
        if add_prefix_space and not text[0].isspace():
            text = " " + text
        return text

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        """

        :param str model_dir_or_name: 目录或者缩写名
        :param kwargs:
        :return:
        """
        # 它需要两个文件，第一个是vocab.json，第二个是merge_file?
        model_dir = _get_bert_dir(model_dir_or_name)
        # 里面会包含四个文件vocab.json, merge.txt, config.json, model.bin

        tokenizer_config_file = _get_file_name_base_on_postfix(model_dir, 'config.json')
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
        # Set max length if needed
        if model_dir_or_name in PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES[model_dir_or_name]
            if max_len is not None and isinstance(max_len, (int, float)):
                init_kwargs["max_len"] = min(init_kwargs.get("max_len", int(1e12)), max_len)

        # 将vocab, merge加入到init_kwargs中
        if 'vocab_file' in kwargs:  # 如果指定了词表则用指定词表
            init_kwargs['vocab_file'] = kwargs['vocab_file']
        else:
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


