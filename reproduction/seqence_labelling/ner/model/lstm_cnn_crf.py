
import torch
from torch import nn
from fastNLP import seq_len_to_mask
from fastNLP.modules import Embedding
from fastNLP.modules import LSTM
from fastNLP.modules import ConditionalRandomField, allowed_transitions, TimestepDropout
import torch.nn.functional as F
from fastNLP import Const

class CNNBiLSTMCRF(nn.Module):
    def __init__(self, embed, char_embed, hidden_size, num_layers, tag_vocab, dropout=0.5, encoding_type='bioes'):
        super().__init__()

        self.embedding = Embedding(embed, dropout=0.5)
        self.char_embedding = Embedding(char_embed, dropout=0.5)
        self.lstm = LSTM(input_size=self.embedding.embedding_dim+self.char_embedding.embedding_dim,
                         hidden_size=hidden_size//2, num_layers=num_layers,
                         bidirectional=True, batch_first=True, dropout=dropout)
        self.forward_fc = nn.Linear(hidden_size//2, len(tag_vocab))
        self.backward_fc = nn.Linear(hidden_size//2, len(tag_vocab))

        transitions = allowed_transitions(tag_vocab.idx2word, encoding_type=encoding_type, include_start_end=False)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=False, allowed_transitions=transitions)

        self.dropout = TimestepDropout(dropout, inplace=True)

        for name, param in self.named_parameters():
            if 'ward_fc' in name:
                if param.data.dim()>1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0)
            if 'crf' in name:
                nn.init.zeros_(param)

    def _forward(self, words, cap_words, seq_len, target=None):
        words = self.embedding(words)
        chars = self.char_embedding(cap_words)
        words = torch.cat([words, chars], dim=-1)
        outputs, _ = self.lstm(words, seq_len)
        self.dropout(outputs)
        forwards, backwards = outputs.chunk(2, dim=-1)

        # forward_logits = F.log_softmax(self.forward_fc(forwards), dim=-1)
        # backward_logits = F.log_softmax(self.backward_fc(backwards), dim=-1)

        logits = self.forward_fc(forwards) + self.backward_fc(backwards)
        self.dropout(logits)

        if target is not None:
            loss = self.crf(logits, target, seq_len_to_mask(seq_len))
            return {Const.LOSS: loss}
        else:
            pred, _ = self.crf.viterbi_decode(logits, seq_len_to_mask(seq_len))
            return {Const.OUTPUT: pred}

    def forward(self, words, cap_words, seq_len, target):
        return self._forward(words, cap_words, seq_len, target)

    def predict(self, words, cap_words, seq_len):
        return self._forward(words, cap_words, seq_len, None)
