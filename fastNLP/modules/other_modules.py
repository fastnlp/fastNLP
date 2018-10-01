import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn import Parameter


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=20, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H)
        return x * self.weight + self.bias


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class BiLinear(nn.Module):
    def __init__(self, n_left, n_right, n_out, bias=True):
        """
        Args:
            n_left: size of left input
            n_right: size of right input
            n_out: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        """
        super(BiLinear, self).__init__()
        self.n_left = n_left
        self.n_right = n_right
        self.n_out = n_out

        self.U = Parameter(torch.Tensor(self.n_out, self.n_left, self.n_right))
        self.W_l = Parameter(torch.Tensor(self.n_out, self.n_left))
        self.W_r = Parameter(torch.Tensor(self.n_out, self.n_left))

        if bias:
            self.bias = Parameter(torch.Tensor(n_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """
        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]
        Returns:
        """
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], \
            "batch size of left and right inputs mis-match: (%s, %s)" % (left_size[:-1], right_size[:-1])
        batch = int(np.prod(left_size[:-1]))

        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(batch, self.n_left)
        input_right = input_right.view(batch, self.n_right)

        # output [batch, out_features]
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + \
                 F.linear(input_left, self.W_l, None) + \
                 F.linear(input_right, self.W_r, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output.view(left_size[:-1] + (self.n_out,))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self.n_left) \
               + ', in2_features=' + str(self.n_right) \
               + ', out_features=' + str(self.n_out) + ')'


class BiAffine(nn.Module):
    def __init__(self, n_enc, n_dec, n_labels, biaffine=True, **kwargs):
        """
        Args:
            n_enc: int
                the dimension of the encoder input.
            n_dec: int
                the dimension of the decoder input.
            n_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(BiAffine, self).__init__()
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.num_labels = n_labels
        self.biaffine = biaffine

        self.W_d = Parameter(torch.Tensor(self.num_labels, self.n_dec))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.n_enc))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.n_dec, self.n_enc))
        else:
            self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.constant_(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        """
        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]
        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output
