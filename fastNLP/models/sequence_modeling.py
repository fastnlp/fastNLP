import torch

from fastNLP.models.base_model import BaseModel
from fastNLP.modules import decoder, encoder


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

    def forward(self, x):
        """
        :param x: LongTensor, [batch_size, mex_len]
        :return y: [batch_size, mex_len, tag_size]
        """
        x = self.Embedding(x)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(x)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        return x

    def loss(self, x, y, mask):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :param mask: ByteTensor, [batch_size, ,max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        total_loss = self.Crf(x, y, mask)
        return torch.mean(total_loss)

    def prediction(self, x, mask):
        """
        :param x: FloatTensor, [batch_size, max_len, tag_size]
        :param mask: ByteTensor, [batch_size, max_len]
        :return prediction: list of [decode path(list)]
        """
        tag_seq = self.Crf.viterbi_decode(x, mask)
        return tag_seq
