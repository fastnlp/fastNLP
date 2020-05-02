r"""undocumented"""

__all__ = [
    "TimestepDropout"
]

import torch


class TimestepDropout(torch.nn.Dropout):
    r"""
    传入参数的shape为 ``(batch_size, num_timesteps, embedding_dim)``
    使用同一个shape为 ``(batch_size, embedding_dim)`` 的mask在每个timestamp上做dropout。
    """
    
    def forward(self, x):
        dropout_mask = x.new_ones(x.shape[0], x.shape[-1])
        torch.nn.functional.dropout(dropout_mask, self.p, self.training, inplace=True)
        dropout_mask = dropout_mask.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        if self.inplace:
            x *= dropout_mask
            return
        else:
            return x * dropout_mask
