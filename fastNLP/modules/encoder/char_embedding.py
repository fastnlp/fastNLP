import torch
from torch import nn

from fastNLP.modules.utils import initial_parameter


# from torch.nn.init import xavier_uniform
class ConvCharEmbedding(nn.Module):
    """Character-level Embedding with CNN.

    :param int char_emb_size: the size of character level embedding. Default: 50
        say 26 characters, each embedded to 50 dim vector, then the input_size is 50.
    :param tuple feature_maps: tuple of int. The length of the tuple is the number of convolution operations
        over characters. The i-th integer is the number of filters (dim of out channels) for the i-th
        convolution.
    :param tuple kernels: tuple of int. The width of each kernel.
    """

    def __init__(self, char_emb_size=50, feature_maps=(40, 30, 30), kernels=(3, 4, 5), initial_method=None):
        super(ConvCharEmbedding, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, feature_maps[i], kernel_size=(char_emb_size, kernels[i]), bias=True, padding=(0, 4))
            for i in range(len(kernels))])

        initial_parameter(self, initial_method)

    def forward(self, x):
        """
        :param x: ``[batch_size * sent_length, word_length, char_emb_size]``
        :return: feature map of shape [batch_size * sent_length, sum(feature_maps), 1]
        """
        x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        # [batch_size*sent_length, channel, width, height]
        x = x.transpose(2, 3)
        # [batch_size*sent_length, channel, height, width]
        return self.convolute(x).unsqueeze(2)

    def convolute(self, x):
        feats = []
        for conv in self.convs:
            y = conv(x)
            # [batch_size*sent_length, feature_maps[i], 1, width - kernels[i] + 1]
            y = torch.squeeze(y, 2)
            # [batch_size*sent_length, feature_maps[i], width - kernels[i] + 1]
            y = torch.tanh(y)
            y, __ = torch.max(y, 2)
            # [batch_size*sent_length, feature_maps[i]]
            feats.append(y)
        return torch.cat(feats, 1)  # [batch_size*sent_length, sum(feature_maps)]


class LSTMCharEmbedding(nn.Module):
    """Character-level Embedding with LSTM.

    :param int char_emb_size: the size of character level embedding. Default: 50
        say 26 characters, each embedded to 50 dim vector, then the input_size is 50.
    :param int hidden_size: the number of hidden units. Default:  equal to char_emb_size.
    """
    def __init__(self, char_emb_size=50, hidden_size=None, initial_method=None):
        super(LSTMCharEmbedding, self).__init__()
        self.hidden_size = char_emb_size if hidden_size is None else hidden_size

        self.lstm = nn.LSTM(input_size=char_emb_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)
        initial_parameter(self, initial_method)

    def forward(self, x):
        """
        :param x: ``[ n_batch*n_word, word_length, char_emb_size]``
        :return: [ n_batch*n_word, char_emb_size]
        """
        batch_size = x.shape[0]
        h0 = torch.empty(1, batch_size, self.hidden_size)
        h0 = nn.init.orthogonal_(h0)
        c0 = torch.empty(1, batch_size, self.hidden_size)
        c0 = nn.init.orthogonal_(c0)

        _, hidden = self.lstm(x, (h0, c0))
        return hidden[0].squeeze().unsqueeze(2)
