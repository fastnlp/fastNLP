r"""undocumented"""

__all__ = [
    "ConvolutionCharEncoder",
    "LSTMCharEncoder"
]
import torch
import torch.nn as nn

from ..utils import initial_parameter


# from torch.nn.init import xavier_uniform
class ConvolutionCharEncoder(nn.Module):
    r"""
    char级别的卷积编码器.
    
    """

    def __init__(self, char_emb_size=50, feature_maps=(40, 30, 30), kernels=(1, 3, 5), initial_method=None):
        r"""
        
        :param int char_emb_size: char级别embedding的维度. Default: 50
            :例: 有26个字符, 每一个的embedding是一个50维的向量, 所以输入的向量维度为50.
        :param tuple feature_maps: 一个由int组成的tuple. tuple的长度是char级别卷积操作的数目, 第`i`个int表示第`i`个卷积操作的filter.
        :param tuple kernels: 一个由int组成的tuple. tuple的长度是char级别卷积操作的数目, 第`i`个int表示第`i`个卷积操作的卷积核.
        :param initial_method: 初始化参数的方式, 默认为`xavier normal`
        """
        super(ConvolutionCharEncoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, feature_maps[i], kernel_size=(char_emb_size, kernels[i]), bias=True,
                      padding=(0, kernels[i] // 2))
            for i in range(len(kernels))])

        initial_parameter(self, initial_method)

    def forward(self, x):
        r"""
        :param torch.Tensor x: ``[batch_size * sent_length, word_length, char_emb_size]`` 输入字符的embedding
        :return: torch.Tensor : 卷积计算的结果, 维度为[batch_size * sent_length, sum(feature_maps), 1]
        """
        x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        # [batch_size*sent_length, channel, width, height]
        x = x.transpose(2, 3)
        # [batch_size*sent_length, channel, height, width]
        return self._convolute(x).unsqueeze(2)

    def _convolute(self, x):
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


class LSTMCharEncoder(nn.Module):
    r"""
    char级别基于LSTM的encoder.
    """

    def __init__(self, char_emb_size=50, hidden_size=None, initial_method=None):
        r"""
        :param int char_emb_size: char级别embedding的维度. Default: 50
                例: 有26个字符, 每一个的embedding是一个50维的向量, 所以输入的向量维度为50.
        :param int hidden_size: LSTM隐层的大小, 默认为char的embedding维度
        :param initial_method: 初始化参数的方式, 默认为`xavier normal`
        """
        super(LSTMCharEncoder, self).__init__()
        self.hidden_size = char_emb_size if hidden_size is None else hidden_size

        self.lstm = nn.LSTM(input_size=char_emb_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)
        initial_parameter(self, initial_method)

    def forward(self, x):
        r"""
        :param torch.Tensor x: ``[ n_batch*n_word, word_length, char_emb_size]`` 输入字符的embedding
        :return: torch.Tensor : [ n_batch*n_word, char_emb_size]经过LSTM编码的结果
        """
        batch_size = x.shape[0]
        h0 = torch.empty(1, batch_size, self.hidden_size)
        h0 = nn.init.orthogonal_(h0)
        c0 = torch.empty(1, batch_size, self.hidden_size)
        c0 = nn.init.orthogonal_(c0)

        _, hidden = self.lstm(x, (h0, c0))
        return hidden[0].squeeze().unsqueeze(2)
