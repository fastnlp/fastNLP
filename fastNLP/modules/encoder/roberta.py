
r"""undocumented
这个页面的代码很大程度上参考(复制粘贴)了https://github.com/huggingface/pytorch-pretrained-BERT的代码， 如果你发现该代码对你
    有用，也请引用一下他们。
"""

__all__ = [
    'RobertaModel'
]

import torch
import torch.nn as nn

from .bert import BertEmbeddings, BertModel, BertConfig
from fastNLP.io.file_utils import _get_file_name_base_on_postfix
from ...io.file_utils import _get_roberta_dir
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

    def forward(self, input_ids, token_type_ids, words_embeddings=None):
        position_ids = self.create_position_ids_from_input_ids(input_ids)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, words_embeddings=words_embeddings
        )

    def create_position_ids_from_input_ids(self, x):
        """ Replace non-padding symbols with their position numbers. Position numbers begin at
        padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
        `utils.make_positions`.

        :param torch.Tensor x:
        :return torch.Tensor:
        """
        mask = x.ne(self.padding_idx).long()
        incremental_indicies = torch.cumsum(mask, dim=1) * mask
        return incremental_indicies + self.padding_idx


class RobertaModel(BertModel):
    r"""
    undocumented
    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.apply(self.init_bert_weights)

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        kwargs.pop('cache_dir', None)
        kwargs.pop('from_tf', None)

        # get model dir from name or dir
        pretrained_model_dir = _get_roberta_dir(model_dir_or_name)

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


