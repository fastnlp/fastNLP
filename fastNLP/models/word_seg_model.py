import torch.nn as nn

from fastNLP.models.base_model import BaseModel


class WordSeg(BaseModel):
    """
        PyTorch Network for word segmentation
    """

    def __init__(self, hidden_dim, lstm_num_layers, vocab_size, word_emb_dim=100):
        super(WordSeg, self).__init__()

        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim
        self.lstm_num_layers = lstm_num_layers
        self.hidden_dim = hidden_dim

        self.word_emb = nn.Embedding(self.vocab_size, self.word_emb_dim)

        self.lstm = nn.LSTM(input_size=self.word_emb_dim,
                            hidden_size=self.word_emb_dim,
                            num_layers=self.lstm_num_layers,
                            bias=True,
                            dropout=0.5,
                            batch_first=True)

        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

    def forward(self, x, hidden):
        """
        :param x: tensor of shape [batch_size, seq_len], vocabulary index
        :param hidden:
        :return x: probability of vocabulary entries
                hidden: (memory cell, hidden state) from LSTM
        """
        # [batch_size, seq_len]
        x = self.word_emb(x)
        # [batch_size, seq_len, word_emb_size]
        x, hidden = self.lstm(x, hidden)
        # [batch_size, seq_len, word_emb_size]
        x = x.contiguous().view(x.shape[0] * x.shape[1], -1)
        # [batch_size*seq_len, word_emb_size]
        x = self.linear(x)
        # [batch_size*seq_len, vocab_size]
        return x, hidden
