"""undocumented
bert.py is modified from huggingface/pytorch-pretrained-BERT, which is licensed under the Apache License 2.0.

"""

__all__ = []

import os

import torch
from torch import nn

from .base_model import BaseModel
from ..core.const import Const
from ..core.utils import seq_len_to_mask
from ..modules.encoder import BertModel
from ..modules.encoder.bert import BertConfig, CONFIG_FILE


class BertForSequenceClassification(BaseModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(num_labels, config)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels, config=None, bert_dir=None):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        if bert_dir is not None:
            self.bert = BertModel.from_pretrained(bert_dir)
            config = BertConfig(os.path.join(bert_dir, CONFIG_FILE))
        else:
            if config is None:
                config = BertConfig(30522)
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    @classmethod
    def from_pretrained(cls, num_labels, pretrained_model_dir):
        config = BertConfig(pretrained_model_dir)
        model = cls(num_labels=num_labels, config=config, bert_dir=pretrained_model_dir)
        return model

    def forward(self, words, seq_len=None, target=None):
        if seq_len is None:
            seq_len = torch.ones_like(words, dtype=words.dtype, device=words.device)
        if len(seq_len.size()) + 1 == len(words.size()):
            seq_len = seq_len_to_mask(seq_len, max_len=words.size(-1))
        _, pooled_output = self.bert(words, attention_mask=seq_len, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if target is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, target)
            return {Const.OUTPUT: logits, Const.LOSS: loss}
        else:
            return {Const.OUTPUT: logits}

    def predict(self, words, seq_len=None):
        logits = self.forward(words, seq_len=seq_len)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForMultipleChoice(BaseModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_choices = 2
    model = BertForMultipleChoice(num_choices, config, bert_dir)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_choices, config=None, bert_dir=None):
        super(BertForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        if bert_dir is not None:
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            if config is None:
                config = BertConfig(30522)
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, num_choices, pretrained_model_dir):
        config = BertConfig(pretrained_model_dir)
        model = cls(num_choices=num_choices, config=config, bert_dir=pretrained_model_dir)
        return model

    def forward(self, words, seq_len1=None, seq_len2=None, target=None):
        input_ids, token_type_ids, attention_mask = words, seq_len1, seq_len2
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if target is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, target)
            return {Const.OUTPUT: reshaped_logits, Const.LOSS: loss}
        else:
            return {Const.OUTPUT: reshaped_logits}

    def predict(self, words, seq_len1=None, seq_len2=None,):
        logits = self.forward(words, seq_len1=seq_len1, seq_len2=seq_len2)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForTokenClassification(BaseModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
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
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    bert_dir = 'your-bert-file-dir'
    model = BertForTokenClassification(num_labels, config, bert_dir)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels, config=None, bert_dir=None):
        super(BertForTokenClassification, self).__init__()
        self.num_labels = num_labels
        if bert_dir is not None:
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            if config is None:
                config = BertConfig(30522)
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    @classmethod
    def from_pretrained(cls, num_labels, pretrained_model_dir):
        config = BertConfig(pretrained_model_dir)
        model = cls(num_labels=num_labels, config=config, bert_dir=pretrained_model_dir)
        return model

    def forward(self, words, seq_len1=None, seq_len2=None, target=None):
        sequence_output, _ = self.bert(words, seq_len1, seq_len2, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if target is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if seq_len2 is not None:
                active_loss = seq_len2.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = target.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), target.view(-1))
            return {Const.OUTPUT: logits, Const.LOSS: loss}
        else:
            return {Const.OUTPUT: logits}

    def predict(self, words, seq_len1=None, seq_len2=None):
        logits = self.forward(words, seq_len1, seq_len2)[Const.OUTPUT]
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
    def __init__(self, config=None, bert_dir=None):
        super(BertForQuestionAnswering, self).__init__()
        if bert_dir is not None:
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            if config is None:
                config = BertConfig(30522)
            self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    @classmethod
    def from_pretrained(cls, pretrained_model_dir):
        config = BertConfig(pretrained_model_dir)
        model = cls(config=config, bert_dir=pretrained_model_dir)
        return model

    def forward(self, words, seq_len1=None, seq_len2=None, target1=None, target2=None):
        sequence_output, _ = self.bert(words, seq_len1, seq_len2, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if target1 is not None and target2 is not None:
            # If we are on multi-GPU, split add a dimension
            if len(target1.size()) > 1:
                target1 = target1.squeeze(-1)
            if len(target2.size()) > 1:
                target2 = target2.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            target1.clamp_(0, ignored_index)
            target2.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, target1)
            end_loss = loss_fct(end_logits, target2)
            total_loss = (start_loss + end_loss) / 2
            return {Const.OUTPUTS(0): start_logits, Const.OUTPUTS(1): end_logits, Const.LOSS: total_loss}
        else:
            return {Const.OUTPUTS(0): start_logits, Const.OUTPUTS(1): end_logits}

    def predict(self, words, seq_len1=None, seq_len2=None):
        logits = self.forward(words, seq_len1, seq_len2)
        start_logits = logits[Const.OUTPUTS(0)]
        end_logits = logits[Const.OUTPUTS(1)]
        return {Const.OUTPUTS(0): torch.argmax(start_logits, dim=-1),
                Const.OUTPUTS(1): torch.argmax(end_logits, dim=-1)}
