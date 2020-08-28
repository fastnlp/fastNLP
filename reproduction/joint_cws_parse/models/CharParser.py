


from fastNLP.models.biaffine_parser import BiaffineParser
from fastNLP.models.biaffine_parser import ArcBiaffine, LabelBilinear

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fastNLP.modules.dropout import TimestepDropout
from fastNLP.modules.encoder.variational_rnn import VarLSTM
from fastNLP import seq_len_to_mask
from fastNLP.embeddings import Embedding


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings


class CharBiaffineParser(BiaffineParser):
    def __init__(self, char_vocab_size,
                        emb_dim,
                         bigram_vocab_size,
                        trigram_vocab_size,
                        num_label,
                        rnn_layers=3,
                        rnn_hidden_size=800, #单向的数量
                        arc_mlp_size=500,
                        label_mlp_size=100,
                        dropout=0.3,
                        encoder='lstm',
                        use_greedy_infer=False,
                         app_index = 0,
                         pre_chars_embed=None,
                         pre_bigrams_embed=None,
                         pre_trigrams_embed=None):


        super(BiaffineParser, self).__init__()
        rnn_out_size = 2 * rnn_hidden_size
        self.char_embed = Embedding((char_vocab_size, emb_dim))
        self.bigram_embed = Embedding((bigram_vocab_size, emb_dim))
        self.trigram_embed = Embedding((trigram_vocab_size, emb_dim))
        if pre_chars_embed:
            self.pre_char_embed = Embedding(pre_chars_embed)
            self.pre_char_embed.requires_grad = False
        if pre_bigrams_embed:
            self.pre_bigram_embed = Embedding(pre_bigrams_embed)
            self.pre_bigram_embed.requires_grad = False
        if pre_trigrams_embed:
            self.pre_trigram_embed = Embedding(pre_trigrams_embed)
            self.pre_trigram_embed.requires_grad = False
        self.timestep_drop = TimestepDropout(dropout)
        self.encoder_name = encoder

        if encoder == 'var-lstm':
            self.encoder = VarLSTM(input_size=emb_dim*3,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=rnn_layers,
                                   bias=True,
                                   batch_first=True,
                                   input_dropout=dropout,
                                   hidden_dropout=dropout,
                                   bidirectional=True)
        elif encoder == 'lstm':
            self.encoder = nn.LSTM(input_size=emb_dim*3,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=rnn_layers,
                                   bias=True,
                                   batch_first=True,
                                   dropout=dropout,
                                   bidirectional=True)

        else:
            raise ValueError('unsupported encoder type: {}'.format(encoder))

        self.mlp = nn.Sequential(nn.Linear(rnn_out_size, arc_mlp_size * 2 + label_mlp_size * 2),
                                          nn.LeakyReLU(0.1),
                                          TimestepDropout(p=dropout),)
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size, num_label, bias=True)
        self.use_greedy_infer = use_greedy_infer
        self.reset_parameters()
        self.dropout = dropout

        self.app_index = app_index
        self.num_label = num_label
        if self.app_index != 0:
            raise ValueError("现在app_index必须等于0")

    def reset_parameters(self):
        for name, m in self.named_modules():
            if 'embed' in name:
                pass
            elif hasattr(m, 'reset_parameters') or hasattr(m, 'init_param'):
                pass
            else:
                for p in m.parameters():
                    if len(p.size())>1:
                        nn.init.xavier_normal_(p, gain=0.1)
                    else:
                        nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, chars, bigrams, trigrams, seq_lens, gold_heads=None, pre_chars=None, pre_bigrams=None,
                pre_trigrams=None):
        """
        max_len是包含root的
        :param chars: batch_size x max_len
        :param ngrams: batch_size x max_len*ngram_per_char
        :param seq_lens: batch_size
        :param gold_heads: batch_size x max_len
        :param pre_chars: batch_size x max_len
        :param pre_ngrams: batch_size x max_len*ngram_per_char
        :return dict: parsing results
            arc_pred: [batch_size, seq_len, seq_len]
            label_pred: [batch_size, seq_len, seq_len]
            mask: [batch_size, seq_len]
            head_pred: [batch_size, seq_len] if gold_heads is not provided, predicting the heads
        """
        # prepare embeddings
        batch_size, seq_len = chars.shape
        # print('forward {} {}'.format(batch_size, seq_len))

        # get sequence mask
        mask = seq_len_to_mask(seq_lens).long()

        chars = self.char_embed(chars) # [N,L] -> [N,L,C_0]
        bigrams = self.bigram_embed(bigrams) # [N,L] -> [N,L,C_1]
        trigrams = self.trigram_embed(trigrams)

        if pre_chars is not None:
            pre_chars = self.pre_char_embed(pre_chars)
            # pre_chars = self.pre_char_fc(pre_chars)
            chars = pre_chars + chars
        if pre_bigrams is not None:
            pre_bigrams = self.pre_bigram_embed(pre_bigrams)
            # pre_bigrams = self.pre_bigram_fc(pre_bigrams)
            bigrams = bigrams + pre_bigrams
        if pre_trigrams is not None:
            pre_trigrams = self.pre_trigram_embed(pre_trigrams)
            # pre_trigrams = self.pre_trigram_fc(pre_trigrams)
            trigrams = trigrams + pre_trigrams

        x = torch.cat([chars, bigrams, trigrams], dim=2) # -> [N,L,C]

        # encoder, extract features
        if self.training:
            x = drop_input_independent(x, self.dropout)
        sort_lens, sort_idx = torch.sort(seq_lens, dim=0, descending=True)
        x = x[sort_idx]
        x = nn.utils.rnn.pack_padded_sequence(x, sort_lens, batch_first=True)
        feat, _ = self.encoder(x)  # -> [N,L,C]
        feat, _ = nn.utils.rnn.pad_packed_sequence(feat, batch_first=True)
        _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
        feat = feat[unsort_idx]
        feat = self.timestep_drop(feat)

        # for arc biaffine
        # mlp, reduce dim
        feat = self.mlp(feat)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feat[:,:,:arc_sz], feat[:,:,arc_sz:2*arc_sz]
        label_dep, label_head = feat[:,:,2*arc_sz:2*arc_sz+label_sz], feat[:,:,2*arc_sz+label_sz:]

        # biaffine arc classifier
        arc_pred = self.arc_predictor(arc_head, arc_dep) # [N, L, L]

        # use gold or predicted arc to predict label
        if gold_heads is None or not self.training:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_pred, mask)
            else:
                heads = self.mst_decoder(arc_pred, mask)
            head_pred = heads
        else:
            assert self.training # must be training mode
            if gold_heads is None:
                heads = self.greedy_decoder(arc_pred, mask)
                head_pred = heads
            else:
                head_pred = None
                heads = gold_heads
        # heads: batch_size x max_len

        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=chars.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep) # [N, max_len, num_label]
        # 这里限制一下，只有当head为下一个时，才能预测app这个label
        arange_index = torch.arange(1, seq_len+1, dtype=torch.long, device=chars.device).unsqueeze(0)\
            .repeat(batch_size, 1) # batch_size x max_len
        app_masks = heads.ne(arange_index) #  batch_size x max_len, 为1的位置不可以预测app
        app_masks = app_masks.unsqueeze(2).repeat(1, 1, self.num_label)
        app_masks[:, :, 1:] = 0
        label_pred = label_pred.masked_fill(app_masks, -np.inf)

        res_dict = {'arc_pred': arc_pred, 'label_pred': label_pred, 'mask': mask}
        if head_pred is not None:
            res_dict['head_pred'] = head_pred
        return res_dict

    @staticmethod
    def loss(arc_pred, label_pred, arc_true, label_true, mask):
        """
        Compute loss.

        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, n_tags]
        :param arc_true: [batch_size, seq_len]
        :param label_true: [batch_size, seq_len]
        :param mask: [batch_size, seq_len]
        :return: loss value
        """

        batch_size, seq_len, _ = arc_pred.shape
        flip_mask = (mask.eq(False))
        # _arc_pred = arc_pred.clone()
        _arc_pred = arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))

        arc_true.data[:, 0].fill_(-1)
        label_true.data[:, 0].fill_(-1)

        arc_nll = F.cross_entropy(_arc_pred.view(-1, seq_len), arc_true.view(-1), ignore_index=-1)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)), label_true.view(-1), ignore_index=-1)

        return arc_nll + label_nll

    def predict(self, chars, bigrams, trigrams, seq_lens, pre_chars, pre_bigrams, pre_trigrams):
        """

        max_len是包含root的

        :param chars: batch_size x max_len
        :param ngrams: batch_size x max_len*ngram_per_char
        :param seq_lens: batch_size
        :param pre_chars: batch_size x max_len
        :param pre_ngrams: batch_size x max_len*ngram_per_cha
        :return:
        """
        res = self(chars, bigrams, trigrams, seq_lens, pre_chars=pre_chars, pre_bigrams=pre_bigrams,
                   pre_trigrams=pre_trigrams, gold_heads=None)
        output = {}
        output['arc_pred'] = res.pop('head_pred')
        _, label_pred = res.pop('label_pred').max(2)
        output['label_pred'] = label_pred
        return output

class CharParser(nn.Module):
    def __init__(self, char_vocab_size,
                        emb_dim,
                         bigram_vocab_size,
                        trigram_vocab_size,
                        num_label,
                        rnn_layers=3,
                        rnn_hidden_size=400, #单向的数量
                        arc_mlp_size=500,
                        label_mlp_size=100,
                        dropout=0.3,
                        encoder='var-lstm',
                        use_greedy_infer=False,
                         app_index = 0,
                         pre_chars_embed=None,
                         pre_bigrams_embed=None,
                         pre_trigrams_embed=None):
        super().__init__()

        self.parser = CharBiaffineParser(char_vocab_size,
                                         emb_dim,
                         bigram_vocab_size,
                        trigram_vocab_size,
                        num_label,
                        rnn_layers,
                        rnn_hidden_size, #单向的数量
                        arc_mlp_size,
                        label_mlp_size,
                        dropout,
                        encoder,
                        use_greedy_infer,
                         app_index,
                         pre_chars_embed=pre_chars_embed,
                         pre_bigrams_embed=pre_bigrams_embed,
                         pre_trigrams_embed=pre_trigrams_embed)

    def forward(self, chars, bigrams, trigrams, seq_lens, char_heads, char_labels, pre_chars=None, pre_bigrams=None,
                pre_trigrams=None):
        res_dict = self.parser(chars, bigrams, trigrams, seq_lens, gold_heads=char_heads, pre_chars=pre_chars,
                               pre_bigrams=pre_bigrams, pre_trigrams=pre_trigrams)
        arc_pred = res_dict['arc_pred']
        label_pred = res_dict['label_pred']
        masks = res_dict['mask']
        loss = self.parser.loss(arc_pred, label_pred, char_heads, char_labels, masks)
        return {'loss': loss}

    def predict(self, chars, bigrams, trigrams, seq_lens, pre_chars=None, pre_bigrams=None, pre_trigrams=None):
        res = self.parser(chars, bigrams, trigrams, seq_lens, gold_heads=None, pre_chars=pre_chars,
                               pre_bigrams=pre_bigrams, pre_trigrams=pre_trigrams)
        output = {}
        output['head_preds'] = res.pop('head_pred')
        _, label_pred = res.pop('label_pred').max(2)
        output['label_preds'] = label_pred
        return output
