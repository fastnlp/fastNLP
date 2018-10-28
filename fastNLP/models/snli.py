import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP.models.base_model import BaseModel
from fastNLP.modules import decoder as Decoder, encoder as Encoder


my_inf = 10e12


class SNLI(BaseModel):
    """
    PyTorch Network for SNLI.
    """

    def __init__(self, args, init_embedding=None):
        super(SNLI, self).__init__()
        self.vocab_size = args["vocab_size"]
        self.embed_dim = args["embed_dim"]
        self.hidden_size = args["hidden_size"]
        self.batch_first = args["batch_first"]
        self.dropout = args["dropout"]
        self.n_labels = args["num_classes"]
        self.gpu = args["gpu"] and torch.cuda.is_available()

        self.embedding = Encoder.embedding.Embedding(self.vocab_size, self.embed_dim, init_emb=init_embedding,
                                                     dropout=self.dropout)

        self.embedding_layer = Encoder.Linear(self.embed_dim, self.hidden_size)

        self.encoder = Encoder.LSTM(
            input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, bias=True,
            batch_first=self.batch_first, bidirectional=True
        )

        self.inference_layer = Encoder.Linear(self.hidden_size * 4, self.hidden_size)

        self.decoder = Encoder.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, bias=True,
            batch_first=self.batch_first, bidirectional=True
        )

        self.output = Decoder.MLP([4 * self.hidden_size, self.hidden_size, self.n_labels], 'tanh')

    def forward(self, premise, hypothesis, premise_len, hypothesis_len):
        """ Forward function

        :param premise: A Tensor represents premise: [batch size(B), premise seq len(PL), hidden size(H)].
        :param hypothesis: A Tensor represents hypothesis: [B, hypothesis seq len(HL), H].
        :param premise_len: A Tensor record which is a real word and which is a padding word in premise: [B, PL].
        :param hypothesis_len: A Tensor record which is a real word and which is a padding word in hypothesis: [B, HL].
        :return: prediction: A Tensor of classification result: [B, n_labels(N)].
        """

        premise0 = self.embedding_layer(self.embedding(premise))
        hypothesis0 = self.embedding_layer(self.embedding(hypothesis))

        _BP, _PSL, _HP = premise0.size()
        _BH, _HSL, _HH = hypothesis0.size()
        _BPL, _PLL = premise_len.size()
        _HPL, _HLL = hypothesis_len.size()

        assert _BP == _BH and _BPL == _HPL and _BP == _BPL
        assert _HP == _HH
        assert _PSL == _PLL and _HSL == _HLL

        B, PL, H = premise0.size()
        B, HL, H = hypothesis0.size()

        # a0, (ah0, ac0) = self.encoder(premise)  # a0: [B, PL, H * 2], ah0: [2, B, H]
        # b0, (bh0, bc0) = self.encoder(hypothesis)  # b0: [B, HL, H * 2]

        a0 = self.encoder(premise0)  # a0: [B, PL, H * 2]
        b0 = self.encoder(hypothesis0)  # b0: [B, HL, H * 2]

        a = torch.mean(a0.view(B, PL, -1, H), dim=2)  # a: [B, PL, H]
        b = torch.mean(b0.view(B, HL, -1, H), dim=2)  # b: [B, HL, H]

        ai, bi = self.calc_bi_attention(a, b, premise_len, hypothesis_len)

        ma = torch.cat((a, ai, a - ai, a * ai), dim=2)  # ma: [B, PL, 4 * H]
        mb = torch.cat((b, bi, b - bi, b * bi), dim=2)  # mb: [B, HL, 4 * H]

        f_ma = self.inference_layer(ma)
        f_mb = self.inference_layer(mb)

        vat = self.decoder(f_ma)
        vbt = self.decoder(f_mb)

        va = torch.mean(vat.view(B, PL, -1, H), dim=2)  # va: [B, PL, H]
        vb = torch.mean(vbt.view(B, HL, -1, H), dim=2)  # vb: [B, HL, H]

        # va_ave = torch.mean(va, dim=1)  # va_ave: [B, H]
        # va_max, va_arg_max = torch.max(va, dim=1)  # va_max: [B, H]
        # vb_ave = torch.mean(vb, dim=1)  # vb_ave: [B, H]
        # vb_max, vb_arg_max = torch.max(vb, dim=1)  # vb_max: [B, H]

        va_ave = self.mean_pooling(va, premise_len, dim=1)  # va_ave: [B, H]
        va_max, va_arg_max = self.max_pooling(va, premise_len, dim=1)  # va_max: [B, H]
        vb_ave = self.mean_pooling(vb, hypothesis_len, dim=1)  # vb_ave: [B, H]
        vb_max, vb_arg_max = self.max_pooling(vb, hypothesis_len, dim=1)  # vb_max: [B, H]

        v = torch.cat((va_ave, va_max, vb_ave, vb_max), dim=1)  # v: [B, 4 * H]

        # v_mlp = F.tanh(self.mlp_layer1(v))  # v_mlp: [B, H]
        # prediction = self.mlp_layer2(v_mlp)  # prediction: [B, N]

        prediction = F.tanh(self.output(v))  # prediction: [B, N]

        return prediction

    @staticmethod
    def calc_bi_attention(in_x1, in_x2, x1_len, x2_len):

        # in_x1: [batch_size, x1_seq_len, hidden_size]
        # in_x2: [batch_size, x2_seq_len, hidden_size]
        # x1_len: [batch_size, x1_seq_len]
        # x2_len: [batch_size, x2_seq_len]

        assert in_x1.size()[0] == in_x2.size()[0]
        assert in_x1.size()[2] == in_x2.size()[2]
        # The batch size and hidden size must be equal.
        assert in_x1.size()[1] == x1_len.size()[1] and in_x2.size()[1] == x2_len.size()[1]
        # The seq len in in_x and x_len must be equal.
        assert in_x1.size()[0] == x1_len.size()[0] and x1_len.size()[0] == x2_len.size()[0]

        batch_size = in_x1.size()[0]
        x1_max_len = in_x1.size()[1]
        x2_max_len = in_x2.size()[1]

        in_x2_t = torch.transpose(in_x2, 1, 2)  # [batch_size, hidden_size, x2_seq_len]

        attention_matrix = torch.bmm(in_x1, in_x2_t)  # [batch_size, x1_seq_len, x2_seq_len]

        a_mask = x1_len.le(0.5).float() * -my_inf  # [batch_size, x1_seq_len]
        a_mask = a_mask.view(batch_size, x1_max_len, -1)
        a_mask = a_mask.expand(-1, -1, x2_max_len)  # [batch_size, x1_seq_len, x2_seq_len]
        b_mask = x2_len.le(0.5).float() * -my_inf
        b_mask = b_mask.view(batch_size, -1, x2_max_len)
        b_mask = b_mask.expand(-1, x1_max_len, -1)  # [batch_size, x1_seq_len, x2_seq_len]

        attention_a = F.softmax(attention_matrix + a_mask, dim=2)  # [batch_size, x1_seq_len, x2_seq_len]
        attention_b = F.softmax(attention_matrix + b_mask, dim=1)  # [batch_size, x1_seq_len, x2_seq_len]

        out_x1 = torch.bmm(attention_a, in_x2)  # [batch_size, x1_seq_len, hidden_size]
        attention_b_t = torch.transpose(attention_b, 1, 2)
        out_x2 = torch.bmm(attention_b_t, in_x1)  # [batch_size, x2_seq_len, hidden_size]

        return out_x1, out_x2

    @staticmethod
    def mean_pooling(tensor, mask, dim=0):
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(tensor * masks, dim=dim) / torch.sum(masks, dim=1)

    @staticmethod
    def max_pooling(tensor, mask, dim=0):
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, tensor.size(2)).float()
        return torch.max(tensor + masks.le(0.5).float() * -my_inf, dim=dim)
