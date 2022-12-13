# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ElasticBERT model configuration """


from fastNLP.core.log import logger
from fastNLP.transformers.torch.configuration_utils import PretrainedConfig


__all__ = [
    "ELASTICBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "ElasticBertConfig",
]

ELASTICBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "elasticbert-base": "https://huggingface.co/fnlp/elasticbert-base/resolve/main/config.json",
    "elasticbert-large": "https://huggingface.co/fnlp/elasticbert-large/resolve/main/config.json",
    "elasticbert-base-chinese": "https://huggingface.co/fnlp/elasticbert-chinese-base/resolve/main/config.json"
}


class ElasticBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`ElasticBertModel`

    Args:
        max_output_layers (:obj: `int`, default to 12):
            The maximum number of classification layers.
        num_output_layers (:obj: `int`, default to 1):
            The number of classification layers. Used to specify how many classification layers there are. 
            It is 1 in static usage, and equal to num_hidden_layers in dynamic usage.
    """

    model_type = "elasticbert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_output_layers=12,
        num_output_layers=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_output_layers = max_output_layers
        self.num_output_layers = num_output_layers
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache