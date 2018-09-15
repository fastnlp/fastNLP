import torch

from fastNLP.models.base_model import BaseModel
from fastNLP.modules import decoder, encoder


def seq_mask(seq_len, max_len):
    """Create a mask for the sequences.

    :param seq_len: list or torch.LongTensor
    :param max_len: int
    :return mask: torch.LongTensor
    """
    if isinstance(seq_len, list):
        seq_len = torch.LongTensor(seq_len)
    mask = [torch.ge(seq_len, i + 1) for i in range(max_len)]
    mask = torch.stack(mask, 1)
    return mask


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
        self.Rnn = encoder.lstm.Lstm(word_emb_dim, hidden_dim)
        self.Linear = encoder.linear.Linear(hidden_dim, num_classes)
        self.Crf = decoder.CRF.ConditionalRandomField(num_classes)
        self.mask = None

    def forward(self, word_seq, word_seq_origin_len):
        """
        :param word_seq: LongTensor, [batch_size, mex_len]
        :param word_seq_origin_len: LongTensor, [batch_size,], the origin lengths of the sequences.
        :return y: [batch_size, mex_len, tag_size]
        """
        self.mask = self.make_mask(word_seq, word_seq_origin_len)

        x = self.Embedding(word_seq)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        return x

    def loss(self, x, y):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        total_loss = self.Crf(x, y, self.mask)
        return torch.mean(total_loss)

    def make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)
        mask = mask.to(x)
        return mask

    def prediction(self, x):
        """
        :param x: FloatTensor, [batch_size, max_len, tag_size]
        :return prediction: list of [decode path(list)]
        """
        tag_seq = self.Crf.viterbi_decode(x, self.mask)
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

        self.Embedding = encoder.embedding.Embedding(vocab_size, word_emb_dim, init_emb=emb)
        self.Rnn = encoder.lstm.Lstm(word_emb_dim, hidden_dim, num_layers=3, dropout=0.3, bidirectional=True)
        self.Linear1 = encoder.Linear(hidden_dim * 2, hidden_dim * 2 // 3)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim * 2 // 3)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.3)
        self.Linear2 = encoder.Linear(hidden_dim * 2 // 3, num_classes)

        self.Crf = decoder.CRF.ConditionalRandomField(num_classes)

    def forward(self, x, seq_len):
        """
        :param x: LongTensor, [batch_size, mex_len]
        :param seq_len: list of int.
        :return y: [batch_size, mex_len, tag_size]
        """
        self.mask = self.make_mask(x, seq_len)

        batch_size = x.size(0)
        max_len = x.size(1)
        x = self.Embedding(x)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        x = x.contiguous()
        x = x.view(batch_size * max_len, -1)
        x = self.Linear1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        x = x.view(batch_size, max_len, -1)
        # [batch_size, max_len, num_classes]
        return x
