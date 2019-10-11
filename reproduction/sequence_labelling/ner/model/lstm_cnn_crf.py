
from torch import nn
from fastNLP import seq_len_to_mask
from fastNLP.modules import LSTM
from fastNLP.modules import ConditionalRandomField, allowed_transitions
import torch.nn.functional as F
from fastNLP import Const

class CNNBiLSTMCRF(nn.Module):
    def __init__(self, embed, hidden_size, num_layers, tag_vocab, dropout=0.5, encoding_type='bioes'):
        super().__init__()
        self.embedding = embed
        self.lstm = LSTM(input_size=self.embedding.embedding_dim,
                             hidden_size=hidden_size//2, num_layers=num_layers,
                             bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(tag_vocab))

        transitions = allowed_transitions(tag_vocab.idx2word, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=transitions)

        self.dropout = nn.Dropout(dropout, inplace=True)

        for name, param in self.named_parameters():
            if 'fc' in name:
                if param.data.dim()>1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0)
            if 'crf' in name:
                nn.init.zeros_(param)

    def _forward(self, words, seq_len, target=None):
        words = self.embedding(words)
        outputs, _ = self.lstm(words, seq_len)
        self.dropout(outputs)

        logits = F.log_softmax(self.fc(outputs), dim=-1)

        if target is not None:
            loss = self.crf(logits, target, seq_len_to_mask(seq_len, max_len=logits.size(1))).mean()
            return {Const.LOSS: loss}
        else:
            pred, _ = self.crf.viterbi_decode(logits, seq_len_to_mask(seq_len, max_len=logits.size(1)))
            return {Const.OUTPUT: pred}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len, None)
