
import torch
from fastNLP.modules import LSTM
from fastNLP.modules import allowed_transitions, ConditionalRandomField
from fastNLP import seq_len_to_mask
from torch import nn
from fastNLP import Const
import torch.nn.functional as F

class BiLSTMCRF(nn.Module):
    def __init__(self, char_embed, hidden_size, num_layers, target_vocab=None, bigram_embed=None, trigram_embed=None,
                 dropout=0.5):
        super().__init__()

        embed_size = char_embed.embed_size
        self.char_embed = char_embed
        if bigram_embed:
            embed_size += bigram_embed.embed_size
        self.bigram_embed = bigram_embed
        if trigram_embed:
            embed_size += trigram_embed.embed_size
        self.trigram_embed = trigram_embed

        self.lstm = LSTM(embed_size, hidden_size=hidden_size//2, bidirectional=True, batch_first=True,
                         num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, len(target_vocab))

        transitions = None
        if target_vocab:
            transitions = allowed_transitions(target_vocab, include_start_end=True, encoding_type='bmes')

        self.crf = ConditionalRandomField(num_tags=len(target_vocab), allowed_transitions=transitions)

    def _forward(self, chars, bigrams, trigrams, seq_len, target=None):
        chars = self.char_embed(chars)
        if bigrams is not None:
            bigrams = self.bigram_embed(bigrams)
            chars = torch.cat([chars, bigrams], dim=-1)
        if trigrams is not None:
            trigrams = self.trigram_embed(trigrams)
            chars = torch.cat([chars, trigrams], dim=-1)

        output, _ = self.lstm(chars, seq_len)
        output = self.dropout(output)
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(output, mask)
            return {Const.OUTPUT:pred}
        else:
            loss = self.crf.forward(output, tags=target, mask=mask)
            return {Const.LOSS:loss}

    def forward(self, chars, seq_len, target, bigrams=None, trigrams=None):
        return self._forward(chars, bigrams, trigrams, seq_len, target)

    def predict(self, chars, seq_len, bigrams=None, trigrams=None):
        return self._forward(chars, bigrams, trigrams, seq_len)