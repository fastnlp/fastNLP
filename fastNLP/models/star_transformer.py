from fastNLP.modules.encoder.star_transformer import StarTransformer
from fastNLP.core.utils import seq_lens_to_masks

import torch
from torch import nn
import torch.nn.functional as F


class StarTransEnc(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 hidden_size,
                 num_layers,
                 num_head,
                 head_dim,
                 max_len,
                 emb_dropout,
                 dropout):
        super(StarTransEnc, self).__init__()
        self.emb_fc = nn.Linear(emb_dim, hidden_size)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = StarTransformer(hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       num_head=num_head,
                                       head_dim=head_dim,
                                       dropout=dropout,
                                       max_len=max_len)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.emb_fc(self.emb_drop(x))
        nodes, relay = self.encoder(x, mask)
        return nodes, relay


class Cls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(Cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls),
        )

    def forward(self, x):
        h = self.fc(x)
        return h


class NLICls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(NLICls, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim*4, hid_dim),  #4
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2, torch.abs(x1-x2), x1*x2], 1)
        h = self.fc(x)
        return h

class STSeqLabel(nn.Module):
    """star-transformer model for sequence labeling
    """
    def __init__(self, vocab_size, emb_dim, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1,):
        super(STSeqLabel, self).__init__()
        self.enc = StarTransEnc(vocab_size=vocab_size,
                                emb_dim=emb_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = Cls(hidden_size, num_cls, cls_hidden_size)

    def forward(self, word_seq, seq_lens):
        mask = seq_lens_to_masks(seq_lens)
        nodes, _ = self.enc(word_seq, mask)
        output = self.cls(nodes)
        output = output.transpose(1,2) # make hidden to be dim 1
        return {'output': output} # [bsz, n_cls, seq_len]

    def predict(self, word_seq, seq_lens):
        y = self.forward(word_seq, seq_lens)
        _, pred = y['output'].max(1)
        return {'output': pred, 'seq_lens': seq_lens}


class STSeqCls(nn.Module):
    """star-transformer model for sequence classification
    """

    def __init__(self, vocab_size, emb_dim, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1,):
        super(STSeqCls, self).__init__()
        self.enc = StarTransEnc(vocab_size=vocab_size,
                                emb_dim=emb_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = Cls(hidden_size, num_cls, cls_hidden_size)

    def forward(self, word_seq, seq_lens):
        mask = seq_lens_to_masks(seq_lens)
        nodes, relay = self.enc(word_seq, mask)
        y = 0.5 * (relay + nodes.max(1)[0])
        output = self.cls(y) # [bsz, n_cls]
        return {'output': output}

    def predict(self, word_seq, seq_lens):
        y = self.forward(word_seq, seq_lens)
        _, pred = y['output'].max(1)
        return {'output': pred}


class STNLICls(nn.Module):
    """star-transformer model for NLI
    """

    def __init__(self, vocab_size, emb_dim, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1,):
        super(STNLICls, self).__init__()
        self.enc = StarTransEnc(vocab_size=vocab_size,
                                emb_dim=emb_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = NLICls(hidden_size, num_cls, cls_hidden_size)

    def forward(self, word_seq1, word_seq2, seq_lens1, seq_lens2):
        mask1 = seq_lens_to_masks(seq_lens1)
        mask2 = seq_lens_to_masks(seq_lens2)
        def enc(seq, mask):
            nodes, relay = self.enc(seq, mask)
            return 0.5 * (relay + nodes.max(1)[0])
        y1 = enc(word_seq1, mask1)
        y2 = enc(word_seq2, mask2)
        output = self.cls(y1, y2) # [bsz, n_cls]
        return {'output': output}

    def predict(self, word_seq1, word_seq2, seq_lens1, seq_lens2):
        y = self.forward(word_seq1, word_seq2, seq_lens1, seq_lens2)
        _, pred = y['output'].max(1)
        return {'output': pred}
