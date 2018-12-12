import torch


class TimestepDropout(torch.nn.Dropout):
    """This module accepts a `[batch_size, num_timesteps, embedding_dim)]` and use a single
    dropout mask of shape `(batch_size, embedding_dim)` to apply on every time step.
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
