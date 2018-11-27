import torch
import numpy as np

from fastNLP.models.base_model import BaseModel
from fastNLP.modules import decoder, encoder
from fastNLP.modules.utils import seq_mask


class SeqLabeling(BaseModel):
    """
    PyTorch Network for sequence labeling
    """

    def __init__(self, args):
        super(SeqLabeling, self).__init__()
        vocab_size = args["vocab_size"]
        word_emb_dim = args["word_emb_dim"]
        hidden_dim = args["rnn_hidden_units"]
        num_classes = args["num_classes"]

        self.Embedding = encoder.embedding.Embedding(vocab_size, word_emb_dim)
        self.Rnn = encoder.lstm.LSTM(word_emb_dim, hidden_dim)
        self.Linear = encoder.linear.Linear(hidden_dim, num_classes)
        self.Crf = decoder.CRF.ConditionalRandomField(num_classes)
        self.mask = None

    def forward(self, word_seq, word_seq_origin_len, truth=None):
        """
        :param word_seq: LongTensor, [batch_size, mex_len]
        :param word_seq_origin_len: LongTensor, [batch_size,], the origin lengths of the sequences.
        :param truth: LongTensor, [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                    If truth is not None, return loss, a scalar. Used in training.
        """
        assert word_seq.shape[0] == word_seq_origin_len.shape[0]
        if truth is not None:
            assert truth.shape == word_seq.shape
        self.mask = self.make_mask(word_seq, word_seq_origin_len)

        x = self.Embedding(word_seq)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        return {"loss": self._internal_loss(x, truth) if truth is not None else None,
                "predict": self.decode(x)}

    def loss(self, x, y):
        """ Since the loss has been computed in forward(), this function simply returns x."""
        return x

    def _internal_loss(self, x, y):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        assert x.shape[:2] == y.shape
        assert y.shape == self.mask.shape
        total_loss = self.Crf(x, y, self.mask)
        return torch.mean(total_loss)

    def make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_mask(seq_len, max_len)
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask

    def decode(self, x, pad=True):
        """
        :param x: FloatTensor, [batch_size, max_len, tag_size]
        :param pad: pad the output sequence to equal lengths
        :return prediction: list of [decode path(list)]
        """
        max_len = x.shape[1]
        tag_seq = self.Crf.viterbi_decode(x, self.mask)
        # pad prediction to equal length
        if pad is True:
            for pred in tag_seq:
                if len(pred) < max_len:
                    pred += [0] * (max_len - len(pred))
        return tag_seq


class AdvSeqLabel(SeqLabeling):
    """
    Advanced Sequence Labeling Model
    """

    def __init__(self, args, emb=None):
        super(AdvSeqLabel, self).__init__(args)

        vocab_size = args["vocab_size"]
        word_emb_dim = args["word_emb_dim"]
        hidden_dim = args["rnn_hidden_units"]
        num_classes = args["num_classes"]
        dropout = args['dropout']

        self.Embedding = encoder.embedding.Embedding(vocab_size, word_emb_dim, init_emb=emb)
        self.norm1 = torch.nn.LayerNorm(word_emb_dim)
        # self.Rnn = encoder.lstm.LSTM(word_emb_dim, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True)
        self.Rnn = torch.nn.LSTM(input_size=word_emb_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
        self.Linear1 = encoder.Linear(hidden_dim * 2, hidden_dim * 2 // 3)
        self.norm2 = torch.nn.LayerNorm(hidden_dim * 2 // 3)
        # self.batch_norm = torch.nn.BatchNorm1d(hidden_dim * 2 // 3)
        self.relu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.Linear2 = encoder.Linear(hidden_dim * 2 // 3, num_classes)

        self.Crf = decoder.CRF.ConditionalRandomField(num_classes, include_start_end_trans=False)

    def forward(self, word_seq, word_seq_origin_len, truth=None):
        """
        :param word_seq: LongTensor, [batch_size, mex_len]
        :param word_seq_origin_len: LongTensor, [batch_size, ]
        :param truth: LongTensor, [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                   If truth is not None, return loss, a scalar. Used in training.
        """

        word_seq = word_seq.long()
        word_seq_origin_len = word_seq_origin_len.long()
        self.mask = self.make_mask(word_seq, word_seq_origin_len)
        sent_len, idx_sort = torch.sort(word_seq_origin_len, descending=True)
        _, idx_unsort = torch.sort(idx_sort, descending=False)

        # word_seq_origin_len = word_seq_origin_len.long()
        truth = truth.long() if truth is not None else None

        batch_size = word_seq.size(0)
        max_len = word_seq.size(1)
        if next(self.parameters()).is_cuda:
            word_seq = word_seq.cuda()
            idx_sort = idx_sort.cuda()
            idx_unsort = idx_unsort.cuda()
            self.mask = self.mask.cuda()

        x = self.Embedding(word_seq)
        x = self.norm1(x)
        # [batch_size, max_len, word_emb_dim]

        sent_variable = x[idx_sort]
        sent_packed = torch.nn.utils.rnn.pack_padded_sequence(sent_variable, sent_len, batch_first=True)

        x, _ = self.Rnn(sent_packed)
        # print(x)
        # [batch_size, max_len, hidden_size * direction]

        sent_output = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
        x = sent_output[idx_unsort]

        x = x.contiguous()
        # x = x.view(batch_size * max_len, -1)
        x = self.Linear1(x)
        # x = self.batch_norm(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        # x = x.view(batch_size, max_len, -1)
        # [batch_size, max_len, num_classes]
        # TODO seq_lens的key这样做不合理
        return {"loss": self._internal_loss(x, truth) if truth is not None else None,
                "predict": self.decode(x),
                'word_seq_origin_len': word_seq_origin_len}

    def predict(self, **x):
        out = self.forward(**x)
        return {"predict": out["predict"]}

    def loss(self, **kwargs):
        assert 'loss' in kwargs
        return kwargs['loss']

if __name__ == '__main__':
    args = {
        'vocab_size': 20,
        'word_emb_dim': 100,
        'rnn_hidden_units': 100,
        'num_classes': 10,
    }
    model = AdvSeqLabel(args)
    data = []
    for i in range(20):
        word_seq = torch.randint(20, (15,)).long()
        word_seq_len = torch.LongTensor([15])
        truth = torch.randint(10, (15,)).long()
        data.append((word_seq, word_seq_len, truth))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)
    curidx = 0
    for i in range(1000):
        endidx = min(len(data), curidx + 5)
        b_word, b_len, b_truth = [], [], []
        for word_seq, word_seq_len, truth in data[curidx: endidx]:
            b_word.append(word_seq)
            b_len.append(word_seq_len)
            b_truth.append(truth)
        word_seq = torch.stack(b_word, dim=0)
        word_seq_len = torch.cat(b_len, dim=0)
        truth = torch.stack(b_truth, dim=0)
        res = model(word_seq, word_seq_len, truth)
        loss = res['loss']
        pred = res['predict']
        print('loss: {} acc {}'.format(loss.item(), ((pred.data == truth).long().sum().float() / word_seq_len.sum().float())))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        curidx = endidx
        if curidx == len(data):
            curidx = 0

