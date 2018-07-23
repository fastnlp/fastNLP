import torch

from fastNLP.models.base_model import BaseModel
from fastNLP.modules import decoder, encoder, utils


class SeqLabeling(BaseModel):
    """
    PyTorch Network for sequence labeling
    """

    def __init__(self, hidden_dim,
                 rnn_num_layer,
                 num_classes,
                 vocab_size,
                 word_emb_dim=100,
                 init_emb=None,
                 rnn_mode="gru",
                 bi_direction=False,
                 dropout=0.5,
                 use_crf=True):
        super(SeqLabeling, self).__init__()

        self.Embedding = encoder.embedding.Embedding(vocab_size, word_emb_dim)
        self.Rnn = encoder.lstm.Lstm(word_emb_dim, hidden_dim)
        self.Linear = encoder.linear.Linear(hidden_dim, num_classes)
        self.Crf = decoder.CRF.ConditionalRandomField(num_classes)

    def forward(self, x):
        """
        :param x: LongTensor, [batch_size, mex_len]
        :return y: [batch_size, tag_size, tag_size]
        """
        x = self.Embedding(x)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        return x

    def loss(self, x, y, seq_length):
        """
        Negative log likelihood loss.
        :param x: FloatTensor, [batch_size, max_len, tag_size]
        :param y: LongTensor, [batch_size, max_len]
        :param seq_length: list of int. [batch_size]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()

        batch_size = x.size(0)
        max_len = x.size(1)

        mask = utils.seq_mask(seq_length, max_len)
        mask = mask.byte().view(batch_size, max_len)
        # mask = x.new(batch_size, max_len)

        total_loss = self.Crf(x, y, mask)

        return torch.mean(total_loss)

    def prediction(self, x, seq_length):
        """
        :param x: FloatTensor, [batch_size, tag_size, tag_size]
        :param seq_length: int
        :return prediction: list of tuple of (decode path(list), best score)
        """
        x = x.float()
        batch_size = x.size(0)
        max_len = x.size(1)

        mask = utils.seq_mask(seq_length, max_len)
        mask = mask.byte()
        # mask = x.new(batch_size, max_len)

        tag_seq = self.Crf.viterbi_decode(x, mask)

        return tag_seq
