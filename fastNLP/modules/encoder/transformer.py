import torch
from torch import nn

from ..aggregator.attention import MultiHeadAtte
from ..dropout import TimestepDropout


class TransformerEncoder(nn.Module):
    class SubLayer(nn.Module):
        def __init__(self, model_size, inner_size, key_size, value_size, num_head, dropout=0.1):
            super(TransformerEncoder.SubLayer, self).__init__()
            self.atte = MultiHeadAtte(model_size, key_size, value_size, num_head, dropout)
            self.norm1 = nn.LayerNorm(model_size)
            self.ffn = nn.Sequential(nn.Linear(model_size, inner_size),
                                     nn.ReLU(),
                                     nn.Linear(inner_size, model_size),
                                     TimestepDropout(dropout),)
            self.norm2 = nn.LayerNorm(model_size)

        def forward(self, input, seq_mask=None, atte_mask_out=None):
            """

            :param input: [batch, seq_len, model_size]
            :param seq_mask: [batch, seq_len]
            :return: [batch, seq_len, model_size]
            """
            attention = self.atte(input, input, input, atte_mask_out)
            norm_atte = self.norm1(attention + input)
            attention *= seq_mask
            output = self.ffn(norm_atte)
            output = self.norm2(output + norm_atte)
            output *= seq_mask
            return output

    def __init__(self, num_layers, **kargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([self.SubLayer(**kargs) for _ in range(num_layers)])

    def forward(self, x, seq_mask=None):
        output = x
        if seq_mask is None:
            atte_mask_out = None
        else:
            atte_mask_out = (seq_mask < 1)[:,None,:]
            seq_mask = seq_mask[:,:,None]
        for layer in self.layers:
            output = layer(output, seq_mask, atte_mask_out)
        return output
