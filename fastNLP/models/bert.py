"""undocumented
bert.py is modified from huggingface/pytorch-pretrained-BERT, which is licensed under the Apache License 2.0.

"""

__all__ = []

import warnings

import torch
from torch import nn

from .base_model import BaseModel
from ..core.const import Const
from ..core._logger import logger
from ..modules.encoder import BertModel
from ..modules.encoder.bert import BertConfig, CONFIG_FILE
from ..embeddings.bert_embedding import BertEmbedding


class BertForSequenceClassification(BaseModel):
    """BERT model for classification.
    """
    def __init__(self, init_embed: BertEmbedding, num_labels: int=2):
        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = init_embed
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            warn_msg = "Bert for sequence classification excepts BertEmbedding `include_cls_sep` True, but got False."
            logger.warn(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        hidden = self.dropout(self.bert(words))
        cls_hidden = hidden[:, 0]
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForSentenceMatching(BaseModel):

    """BERT model for matching.
    """
    def __init__(self, init_embed: BertEmbedding, num_labels: int=2):
        super(BertForSentenceMatching, self).__init__()
        self.num_labels = num_labels
        self.bert = init_embed
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            error_msg = "Bert for sentence matching excepts BertEmbedding `include_cls_sep` True, but got False."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def forward(self, words):
        hidden = self.dropout(self.bert(words))
        cls_hidden = hidden[:, 0]
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForMultipleChoice(BaseModel):
    """BERT model for multiple choice tasks.
    """
    def __init__(self, init_embed: BertEmbedding, num_choices=2):
        super(BertForMultipleChoice, self).__init__()

        self.num_choices = num_choices
        self.bert = init_embed
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.embedding_dim, 1)
        self.include_cls_sep = init_embed.model.include_cls_sep

        if not self.bert.model.include_cls_sep:
            error_msg = "Bert for multiple choice excepts BertEmbedding `include_cls_sep` True, but got False."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def forward(self, words):
        """
        :param torch.Tensor words: [batch_size, num_choices, seq_len]
        :return: [batch_size, num_labels]
        """
        batch_size, num_choices, seq_len = words.size()

        input_ids = words.view(batch_size * num_choices, seq_len)
        hidden = self.bert(input_ids)
        pooled_output = hidden[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        return {Const.OUTPUT: reshaped_logits}

    def predict(self, words):
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForTokenClassification(BaseModel):
    """BERT model for token-level classification.
    """
    def __init__(self, init_embed: BertEmbedding, num_labels):
        super(BertForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = init_embed
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)
        self.include_cls_sep = init_embed.model.include_cls_sep

        if self.include_cls_sep:
            warn_msg = "Bert for token classification excepts BertEmbedding `include_cls_sep` False, but got True."
            warnings.warn(warn_msg)
            logger.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.Tensor words: [batch_size, seq_len]
        :return: [batch_size, seq_len, num_labels]
        """
        sequence_output = self.bert(words)
        if self.include_cls_sep:
            sequence_output = sequence_output[:, 1: -1]  # [batch_size, seq_len, embed_dim]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForQuestionAnswering(BaseModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `bert_dir`: a dir which contains the bert parameters within file `pytorch_model.bin`
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    bert_dir = 'your-bert-file-dir'
    model = BertForQuestionAnswering(config, bert_dir)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, init_embed: BertEmbedding, num_labels=2):
        super(BertForQuestionAnswering, self).__init__()

        self.bert = init_embed
        self.num_labels = num_labels
        self.qa_outputs = nn.Linear(self.bert.embedding_dim, self.num_labels)

        if not self.bert.model.include_cls_sep:
            error_msg = "Bert for multiple choice excepts BertEmbedding `include_cls_sep` True, but got False."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def forward(self, words):
        sequence_output = self.bert(words)
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]

        return {Const.OUTPUTS(i): logits[:, :, i] for i in range(self.num_labels)}

    def predict(self, words):
        logits = self.forward(words)
        return {Const.OUTPUTS(i): torch.argmax(logits[Const.OUTPUTS(i)], dim=-1) for i in range(self.num_labels)}
