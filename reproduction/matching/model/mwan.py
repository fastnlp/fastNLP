import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
from fastNLP.core.const import Const

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidrect, dropout):
        super(RNNModel, self).__init__()

        if num_layers <= 1:
            dropout = 0.0
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidrect)

        self.number = (2 if bidrect else 1) * num_layers

    def forward(self, x, mask):
        '''
            mask: (batch_size, seq_len) 
            x: (batch_size, seq_len, input_size)
        '''
        lens = (mask).long().sum(dim=1)
        lens, idx_sort = tc.sort(lens, descending=True)
        _, idx_unsort = tc.sort(idx_sort)

        x = x[idx_sort]

        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        self.rnn.flatten_parameters()
        y, h = self.rnn(x)
        y, lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)

        h = h.transpose(0,1).contiguous()   #make batch size first

        y = y[idx_unsort]                   #(batch_size, seq_len, bid * hid_size)
        h = h[idx_unsort]                   #(batch_size, number, hid_size)

        return y, h

class Contexualizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super(Contexualizer, self).__init__()

        self.rnn = RNNModel(input_size, hidden_size, num_layers, True, dropout)
        self.output_size = hidden_size * 2

        self.reset_parameters()

    def reset_parameters(self):
        weights = self.rnn.rnn.all_weights
        for w1 in weights:
            for w2 in w1:
                if len(list(w2.size())) <= 1:
                    w2.data.fill_(0)
                else: nn.init.xavier_normal_(w2.data, gain=1.414)

    def forward(self, s, mask):
        y = self.rnn(s, mask)[0]            # (batch_size, seq_len, 2 * hidden_size)

        return y

class ConcatAttention_Param(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(ConcatAttention_Param, self).__init__()
        self.ln = nn.Linear(input_size + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.vq = nn.Parameter(tc.rand(hidden_size))
        self.drop = nn.Dropout(dropout)

        self.output_size = input_size
        
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.v.weight.data)
        nn.init.xavier_uniform_(self.ln.weight.data)
        self.ln.bias.data.fill_(0)

    def forward(self, h, mask):
        '''
            h: (batch_size, len, input_size)
            mask: (batch_size, len)
        '''

        vq = self.vq.view(1,1,-1).expand(h.size(0), h.size(1), self.vq.size(0))

        s = self.v(tc.tanh(self.ln(tc.cat([h,vq],-1)))).squeeze(-1)    # (batch_size, len)
        
        s = s - ((mask.eq(False)).float() * 10000)
        a = tc.softmax(s, dim=1)

        r = a.unsqueeze(-1) * h       # (batch_size, len, input_size)
        r = tc.sum(r, dim=1)          # (batch_size, input_size)

        return self.drop(r)

 
def get_2dmask(mask_hq, mask_hp, siz=None):

    if siz is None:
        siz = (mask_hq.size(0), mask_hq.size(1), mask_hp.size(1))

    mask_mat = 1
    if mask_hq is not None:
        mask_mat = mask_mat * mask_hq.unsqueeze(2).expand(siz)
    if mask_hp is not None:
        mask_mat = mask_mat * mask_hp.unsqueeze(1).expand(siz)
    return mask_mat

def Attention(hq, hp, mask_hq, mask_hp, my_method):
    standard_size = (hq.size(0), hq.size(1), hp.size(1), hq.size(-1))
    mask_mat = get_2dmask(mask_hq, mask_hp, standard_size[:-1])

    hq_mat = hq.unsqueeze(2).expand(standard_size)
    hp_mat = hp.unsqueeze(1).expand(standard_size)

    s = my_method(hq_mat, hp_mat)           # (batch_size, len_q, len_p)

    s = s - ((mask_mat.eq(False)).float() * 10000)
    a = tc.softmax(s, dim=1)

    q = a.unsqueeze(-1) * hq_mat            #(batch_size, len_q, len_p, input_size)
    q = tc.sum(q, dim=1)                    #(batch_size, len_p, input_size)

    return q

class ConcatAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2, input_size_2=-1):
        super(ConcatAttention, self).__init__()

        if input_size_2 < 0:
            input_size_2 = input_size
        self.ln = nn.Linear(input_size + input_size_2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.drop = nn.Dropout(dropout)

        self.output_size = input_size

        
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.v.weight.data)
        nn.init.xavier_uniform_(self.ln.weight.data)
        self.ln.bias.data.fill_(0)

    def my_method(self, hq_mat, hp_mat):
        s = tc.cat([hq_mat, hp_mat], dim=-1)
        s = self.v(tc.tanh(self.ln(s))).squeeze(-1)    #(batch_size, len_q, len_p)
        return s

    def forward(self, hq, hp, mask_hq=None, mask_hp=None):
        '''
            hq: (batch_size, len_q, input_size)
            mask_hq: (batch_size, len_q)
        '''
        return self.drop(Attention(hq, hp, mask_hq, mask_hp, self.my_method))

class MinusAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(MinusAttention, self).__init__()
        self.ln = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        self.drop = nn.Dropout(dropout)
        self.output_size = input_size
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.v.weight.data)
        nn.init.xavier_uniform_(self.ln.weight.data)
        self.ln.bias.data.fill_(0)

    def my_method(self, hq_mat, hp_mat):
        s = hq_mat - hp_mat
        s = self.v(tc.tanh(self.ln(s))).squeeze(-1)    #(batch_size, len_q, len_p) s[j,t]
        return s

    def forward(self, hq, hp, mask_hq=None, mask_hp=None):
        return self.drop(Attention(hq, hp, mask_hq, mask_hp, self.my_method))

class DotProductAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(DotProductAttention, self).__init__()
        self.ln = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        self.drop = nn.Dropout(dropout)
        self.output_size = input_size
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.v.weight.data)
        nn.init.xavier_uniform_(self.ln.weight.data)
        self.ln.bias.data.fill_(0)

    def my_method(self, hq_mat, hp_mat):
        s = hq_mat * hp_mat
        s = self.v(tc.tanh(self.ln(s))).squeeze(-1)    #(batch_size, len_q, len_p) s[j,t]
        return s

    def forward(self, hq, hp, mask_hq=None, mask_hp=None):
        return self.drop(Attention(hq, hp, mask_hq, mask_hp, self.my_method))

class BiLinearAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2, input_size_2=-1):
        super(BiLinearAttention, self).__init__()

        input_size_2 = input_size if input_size_2 < 0 else input_size_2

        self.ln = nn.Linear(input_size_2, input_size)
        self.drop = nn.Dropout(dropout)
        self.output_size = input_size
        
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.ln.weight.data)
        self.ln.bias.data.fill_(0)
        
    def my_method(self, hq, hp, mask_p):
        # (bs, len, input_size)

        hp = self.ln(hp)
        hp = hp * mask_p.unsqueeze(-1)
        s = tc.matmul(hq, hp.transpose(-1,-2))

        return s

    def forward(self, hq, hp, mask_hq=None, mask_hp=None):
        standard_size = (hq.size(0), hq.size(1), hp.size(1), hq.size(-1))
        mask_mat = get_2dmask(mask_hq, mask_hp, standard_size[:-1])

        s = self.my_method(hq, hp, mask_hp)         # (batch_size, len_q, len_p)

        s = s - ((mask_mat.eq(False)).float() * 10000)
        a = tc.softmax(s, dim=1)

        hq_mat = hq.unsqueeze(2).expand(standard_size)
        q = a.unsqueeze(-1) * hq_mat                #(batch_size, len_q, len_p, input_size)
        q = tc.sum(q, dim=1)                      #(batch_size, len_p, input_size)

        return self.drop(q)


class AggAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(AggAttention, self).__init__()
        self.ln = nn.Linear(input_size + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.vq = nn.Parameter(tc.rand(hidden_size, 1))
        self.drop = nn.Dropout(dropout)

        self.output_size = input_size
        
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.vq.data)
        nn.init.xavier_uniform_(self.v.weight.data)
        nn.init.xavier_uniform_(self.ln.weight.data)
        self.ln.bias.data.fill_(0)
        self.vq.data = self.vq.data[:,0]


    def forward(self, hs, mask):
        '''
            hs: [(batch_size, len_q, input_size), ...]
            mask: (batch_size, len_q)
        '''
        
        hs = tc.cat([h.unsqueeze(0) for h in hs], dim=0)# (4, batch_size, len_q, input_size)

        vq = self.vq.view(1,1,1,-1).expand(hs.size(0), hs.size(1), hs.size(2), self.vq.size(0))

        s = self.v(tc.tanh(self.ln(tc.cat([hs,vq],-1)))).squeeze(-1)# (4, batch_size, len_q)

        s = s - ((mask.unsqueeze(0).eq(False)).float() * 10000)
        a = tc.softmax(s, dim=0)

        x = a.unsqueeze(-1) * hs
        x = tc.sum(x, dim=0)#(batch_size, len_q, input_size)

        return self.drop(x)

class Aggragator(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(Aggragator, self).__init__()

        now_size = input_size
        self.ln = nn.Linear(2 * input_size, 2 * input_size)

        now_size = 2 * input_size
        self.rnn = Contexualizer(now_size, hidden_size, 2, dropout)

        now_size = self.rnn.output_size
        self.agg_att = AggAttention(now_size, now_size, dropout)

        now_size = self.agg_att.output_size
        self.agg_rnn = Contexualizer(now_size, hidden_size, 2, dropout)

        self.drop = nn.Dropout(dropout)

        self.output_size = self.agg_rnn.output_size

    def forward(self, qs, hp, mask):
        '''
            qs: [ (batch_size, len_p, input_size), ...]
            hp: (batch_size, len_p, input_size)
            mask if the same of hp's mask
        '''

        hs = [0 for _ in range(len(qs))]

        for i in range(len(qs)):
            q = qs[i]
            x = tc.cat([q, hp], dim=-1)
            g = tc.sigmoid(self.ln(x))
            x_star = x * g
            h = self.rnn(x_star, mask)

            hs[i] = h

        x = self.agg_att(hs, mask)      #(batch_size, len_p, output_size)
        h = self.agg_rnn(x, mask)       #(batch_size, len_p, output_size)
        return self.drop(h)


class Mwan_Imm(nn.Module):
    def __init__(self, input_size, hidden_size, num_class=3, dropout=0.2, use_allennlp=False):
        super(Mwan_Imm, self).__init__()

        now_size = input_size
        self.enc_s1 = Contexualizer(now_size, hidden_size, 2, dropout)
        self.enc_s2 = Contexualizer(now_size, hidden_size, 2, dropout)

        now_size = self.enc_s1.output_size
        self.att_c = ConcatAttention(now_size, hidden_size, dropout)
        self.att_b = BiLinearAttention(now_size, hidden_size, dropout)
        self.att_d = DotProductAttention(now_size, hidden_size, dropout)
        self.att_m = MinusAttention(now_size, hidden_size, dropout)

        now_size = self.att_c.output_size
        self.agg = Aggragator(now_size, hidden_size, dropout)

        now_size = self.enc_s1.output_size
        self.pred_1 = ConcatAttention_Param(now_size, hidden_size, dropout)
        now_size = self.agg.output_size
        self.pred_2 = ConcatAttention(now_size, hidden_size, dropout, 
                                                            input_size_2=self.pred_1.output_size)

        now_size = self.pred_2.output_size
        self.ln1 = nn.Linear(now_size, hidden_size)
        self.ln2 = nn.Linear(hidden_size, num_class)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ln1.weight.data)
        nn.init.xavier_uniform_(self.ln2.weight.data)
        self.ln1.bias.data.fill_(0)
        self.ln2.bias.data.fill_(0)

    def forward(self, s1, s2, mas_s1, mas_s2):
        hq = self.enc_s1(s1, mas_s1)                #(batch_size, len_q, output_size)
        hp = self.enc_s1(s2, mas_s2)

        mas_s1 = mas_s1[:,:hq.size(1)]
        mas_s2 = mas_s2[:,:hp.size(1)]
        mas_q, mas_p = mas_s1, mas_s2

        qc = self.att_c(hq, hp, mas_s1, mas_s2)     #(batch_size, len_p, output_size)
        qb = self.att_b(hq, hp, mas_s1, mas_s2)
        qd = self.att_d(hq, hp, mas_s1, mas_s2)
        qm = self.att_m(hq, hp, mas_s1, mas_s2)

        ho = self.agg([qc,qb,qd,qm], hp, mas_s2)    #(batch_size, len_p, output_size)

        rq = self.pred_1(hq, mas_q)                 #(batch_size, output_size)
        rp = self.pred_2(ho, rq.unsqueeze(1), mas_p)#(batch_size, 1, output_size)
        rp = rp.squeeze(1)                          #(batch_size, output_size)

        rp = F.relu(self.ln1(rp))
        rp = self.ln2(rp)

        return rp

class MwanModel(nn.Module):
    def __init__(self, num_class, EmbLayer, args_of_imm={}, ElmoLayer=None):
        super(MwanModel, self).__init__()

        self.emb = EmbLayer

        if ElmoLayer is not None:
            self.elmo = ElmoLayer
            self.elmo_preln = nn.Linear(3 * self.elmo.emb_size, self.elmo.emb_size)
            self.elmo_ln = nn.Linear(args_of_imm["input_size"] + 
                                                    self.elmo.emb_size, args_of_imm["input_size"])

        else:
            self.elmo = None


        self.imm = Mwan_Imm(num_class=num_class, **args_of_imm)
        self.drop = nn.Dropout(args_of_imm["dropout"])


    def forward(self, words1, words2, str_s1=None, str_s2=None, *pargs, **kwargs):
        '''
            str_s is for elmo use , however we don't use elmo
            str_s: (batch_size, seq_len, word_len)
        '''

        s1, s2 = words1, words2

        mas_s1 = (s1 != 0).float()    # mas: (batch_size, seq_len)
        mas_s2 = (s2 != 0).float()    # mas: (batch_size, seq_len)

        mas_s1.requires_grad = False
        mas_s2.requires_grad = False

        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)

        if self.elmo is not None:
            s1_elmo = self.elmo(str_s1)
            s2_elmo = self.elmo(str_s2)

            s1_elmo = tc.tanh(self.elmo_preln(tc.cat(s1_elmo, dim=-1)))
            s2_elmo = tc.tanh(self.elmo_preln(tc.cat(s2_elmo, dim=-1)))

            s1_emb = tc.cat([s1_emb, s1_elmo], dim=-1)
            s2_emb = tc.cat([s2_emb, s2_elmo], dim=-1)

            s1_emb = tc.tanh(self.elmo_ln(s1_emb))
            s2_emb = tc.tanh(self.elmo_ln(s2_emb))

        s1_emb = self.drop(s1_emb)
        s2_emb = self.drop(s2_emb)

        y = self.imm(s1_emb, s2_emb, mas_s1, mas_s2)

        return {
            Const.OUTPUT: y, 
        }
