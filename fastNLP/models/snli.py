import torch
import torch.nn as nn

from fastNLP.models.base_model import BaseModel
from fastNLP.modules import decoder as Decoder
from fastNLP.modules import encoder as Encoder
from fastNLP.modules import aggregator as Aggregator


my_inf = 10e12


class ESIM(BaseModel):
    """
    PyTorch Network for SNLI task using ESIM model.
    """

    def __init__(self, **kwargs):
        super(ESIM, self).__init__()
        self.vocab_size = kwargs["vocab_size"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_size = kwargs["hidden_size"]
        self.batch_first = kwargs["batch_first"]
        self.dropout = kwargs["dropout"]
        self.n_labels = kwargs["num_classes"]
        self.gpu = kwargs["gpu"] and torch.cuda.is_available()

        self.drop = nn.Dropout(self.dropout)

        self.embedding = Encoder.Embedding(
            self.vocab_size, self.embed_dim, dropout=self.dropout,
            init_emb=kwargs["init_embedding"] if "inin_embedding" in kwargs.keys() else None,
        )

        self.embedding_layer = Encoder.Linear(self.embed_dim, self.hidden_size)

        self.encoder = Encoder.LSTM(
            input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, bias=True,
            batch_first=self.batch_first, bidirectional=True
        )

        self.bi_attention = Aggregator.BiAttention()
        self.mean_pooling = Aggregator.MeanPoolWithMask()
        self.max_pooling = Aggregator.MaxPoolWithMask()

        self.inference_layer = Encoder.Linear(self.hidden_size * 4, self.hidden_size)

        self.decoder = Encoder.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, bias=True,
            batch_first=self.batch_first, bidirectional=True
        )

        self.output = Decoder.MLP([4 * self.hidden_size, self.hidden_size, self.n_labels], 'tanh', dropout=self.dropout)

    def forward(self, words1, words2, seq_len1, seq_len2):
        """ Forward function

        :param words1: A Tensor represents premise: [batch size(B), premise seq len(PL)].
        :param words2: A Tensor represents hypothesis: [B, hypothesis seq len(HL)].
        :param seq_len1: A Tensor record which is a real word and which is a padding word in premise: [B].
        :param seq_len2: A Tensor record which is a real word and which is a padding word in hypothesis: [B].
        :return: prediction: A Dict with Tensor of classification result: [B, n_labels(N)].
        """

        premise0 = self.embedding_layer(self.embedding(words1))
        hypothesis0 = self.embedding_layer(self.embedding(words2))

        _BP, _PSL, _HP = premise0.size()
        _BH, _HSL, _HH = hypothesis0.size()
        _BPL, _PLL = seq_len1.size()
        _HPL, _HLL = seq_len2.size()

        assert _BP == _BH and _BPL == _HPL and _BP == _BPL
        assert _HP == _HH
        assert _PSL == _PLL and _HSL == _HLL

        B, PL, H = premise0.size()
        B, HL, H = hypothesis0.size()

        a0 = self.encoder(self.drop(premise0))  # a0: [B, PL, H * 2]
        b0 = self.encoder(self.drop(hypothesis0))  # b0: [B, HL, H * 2]

        a = torch.mean(a0.view(B, PL, -1, H), dim=2)  # a: [B, PL, H]
        b = torch.mean(b0.view(B, HL, -1, H), dim=2)  # b: [B, HL, H]

        ai, bi = self.bi_attention(a, b, seq_len1, seq_len2)

        ma = torch.cat((a, ai, a - ai, a * ai), dim=2)  # ma: [B, PL, 4 * H]
        mb = torch.cat((b, bi, b - bi, b * bi), dim=2)  # mb: [B, HL, 4 * H]

        f_ma = self.inference_layer(ma)
        f_mb = self.inference_layer(mb)

        vat = self.decoder(self.drop(f_ma))
        vbt = self.decoder(self.drop(f_mb))

        va = torch.mean(vat.view(B, PL, -1, H), dim=2)  # va: [B, PL, H]
        vb = torch.mean(vbt.view(B, HL, -1, H), dim=2)  # vb: [B, HL, H]

        va_ave = self.mean_pooling(va, seq_len1, dim=1)  # va_ave: [B, H]
        va_max, va_arg_max = self.max_pooling(va, seq_len1, dim=1)  # va_max: [B, H]
        vb_ave = self.mean_pooling(vb, seq_len2, dim=1)  # vb_ave: [B, H]
        vb_max, vb_arg_max = self.max_pooling(vb, seq_len2, dim=1)  # vb_max: [B, H]

        v = torch.cat((va_ave, va_max, vb_ave, vb_max), dim=1)  # v: [B, 4 * H]

        prediction = torch.tanh(self.output(v))  # prediction: [B, N]

        return {'pred': prediction}

    def predict(self, words1, words2, seq_len1, seq_len2):
        prediction = self.forward(words1, words2, seq_len1, seq_len2)['pred']
        return torch.argmax(prediction, dim=-1)

