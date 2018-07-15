import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .emd import WlossLayer
from ..utils import orthogonal


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


class OrthEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(OrthEmbedding, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        self.weight = orthogonal(self.weight)
        nn.init.constant_(self.bias, 0.)


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


class Transpose(nn.Module):
    def __init__(self, x, y):
        super(Transpose, self).__init__()
        self.x = x
        self.y = y

    def forward(self, x):
        return x.transpose(self.x, self.y)


class WordDropout(nn.Module):
    def __init__(self, dropout_rate, drop_to_token):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.drop_to_token = drop_to_token

    def forward(self, word_idx):
        if not self.training:
            return word_idx
        drop_mask = torch.rand(word_idx.shape) < self.dropout_rate
        if word_idx.device.type == 'cuda':
            drop_mask = drop_mask.cuda()
        drop_mask = drop_mask.long()
        output = drop_mask * self.drop_to_token + (1 - drop_mask) * word_idx
        return output


import torch
import torch.utils.data
import numpy
from torch.autograd import Function, Variable
from torch import optim


class WlossLayer(torch.nn.Module):
    def __init__(self, lam=100, sinkhorn_iter=50):
        super(WlossLayer, self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        # self.register_buffer("K", torch.exp(-self.cost / self.lam).double())
        # self.register_buffer("KM", (self.cost * self.K).double())

    def forward(self, pred, target, cost):
        return WassersteinLossStab.apply(pred, target,
                                         cost, self.lam, self.sinkhorn_iter)


class WassersteinLossStab(Function):
    @staticmethod
    def forward(ctx, pred, target, cost,
                lam=1e-3, sinkhorn_iter=4):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        # import pdb
        # pdb.set_trace()
        eps = 1e-8

        # pred = pred.gather(dim=1, index=)
        na = pred.size(1)
        nb = target.size(1)

        cost = cost.double()
        pred = pred.double()
        target = target.double()

        cost = cost[:na, :nb].double()
        K = torch.exp(-cost / lam).double()
        KM = (cost * K).double()

        batch_size = pred.size(0)

        # pdb.set_trace()
        log_a, log_b = torch.log(pred + eps), torch.log(target + eps)
        log_u = cost.new(batch_size, na).fill_(-numpy.log(na))
        log_v = cost.new(batch_size, nb).fill_(-numpy.log(nb))
        # import pdb
        # pdb.set_trace()
        for i in range(int(sinkhorn_iter)):
            log_u_max = torch.max(log_u, dim=1)[0]
            u_stab = torch.exp(log_u - log_u_max.unsqueeze(1) + eps)
            log_v = log_b - torch.log(torch.mm(K.t(), u_stab.t()).t()) - log_u_max.unsqueeze(1)
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v - log_v_max.unsqueeze(1))
            tmp = log_u
            log_u = log_a - torch.log(torch.mm(K, v_stab.t()).t() + eps) - log_v_max.unsqueeze(1)
            # print(log_u.sum())
            if torch.norm(tmp - log_u) / torch.norm(log_u) < eps:
                break

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v - log_v_max.unsqueeze(1))
        logcostpart1 = torch.log(torch.mm(KM, v_stab.t()).t() + eps) + log_v_max.unsqueeze(1)
        wnorm = torch.exp(log_u + logcostpart1).mean(0).sum()  # sum(1) for per item pair loss...
        grad_input = log_u * lam
        # print("log_u", log_u)
        grad_input = grad_input - torch.mean(grad_input, dim=1).unsqueeze(1)
        grad_input = grad_input - torch.mean(grad_input, dim=1).unsqueeze(1)
        grad_input = grad_input / batch_size

        ctx.save_for_backward(grad_input)
        # print("grad type", type(grad_input))

        return pred.new((wnorm,)), grad_input

    @staticmethod
    def backward(ctx, grad_output, _):
        grad_input = ctx.saved_variables
        # print(grad)
        res = grad_output.clone()
        res.data.resize_(grad_input[0].size()).copy_(grad_input[0].data)
        res = res.mul_(grad_output[0]).float()
        # print("in backward func:\n\n", res)
        return res, None, None, None, None, None, None


class Sinkhorn(Function):
    def __init__(self):
        super(Sinkhorn, self).__init__()

    def forward(ctx, a, b, M, reg, tau, warmstart, numItermax, stop):
        a = a.double()
        b = b.double()
        M = M.double()

        nbb = b.size(1)

        # init data
        na = len(a)
        nb = len(b)

        cpt = 0

        # we assume that no distances are null except those of the diagonal of
        # distances
        if warmstart is None:
            alpha, beta = np.zeros(na), np.zeros(nb)
        else:
            alpha, beta = warmstart

        if nbb:
            u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
        else:
            u, v = np.ones(na) / na, np.ones(nb) / nb

        def get_K(alpha, beta):
            """log space computation"""
            return np.exp(-(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg)

        def get_Gamma(alpha, beta, u, v):
            """log space gamma computation"""
            return np.exp(
                -(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg + np.log(u.reshape((na, 1))) + np.log(
                    v.reshape((1, nb))))

        # print(np.min(K))

        K = get_K(alpha, beta)
        transp = K
        cpt = 0
        err = 1
        while 1:

            uprev = u
            vprev = v

            # sinkhorn update
            v = b / (np.dot(K.T, u) + 1e-16)
            u = a / (np.dot(K, v) + 1e-16)

            # remove numerical problems and store them in K
            if np.abs(u).max() > tau or np.abs(v).max() > tau:
                if nbb:
                    alpha, beta = alpha + reg * \
                                  np.max(np.log(u), 1), beta + reg * np.max(np.log(v))
                else:
                    alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
                    if nbb:
                        u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
                    else:
                        u, v = np.ones(na) / na, np.ones(nb) / nb
                K = get_K(alpha, beta)

            if cpt % print_period == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                if nbb:
                    err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + \
                          np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
                else:
                    transp = get_Gamma(alpha, beta, u, v)
                    err = np.linalg.norm((np.sum(transp, axis=0) - b)) ** 2
                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % (print_period * 20) == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))

            if err <= stopThr:
                loop = False

            if cpt >= numItermax:
                loop = False

            if np.any(np.isnan(u)) or np.any(np.isnan(v)):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break

            cpt = cpt + 1

        # print('err=',err,' cpt=',cpt)
        if log:
            log['logu'] = alpha / reg + np.log(u)
            log['logv'] = beta / reg + np.log(v)
            log['alpha'] = alpha + reg * np.log(u)
            log['beta'] = beta + reg * np.log(v)
            log['warmstart'] = (log['alpha'], log['beta'])
            if nbb:
                res = np.zeros((nbb))
                for i in range(nbb):
                    res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                return res, log

            else:
                return get_Gamma(alpha, beta, u, v), log
        else:
            if nbb:
                res = np.zeros((nbb))
                for i in range(nbb):
                    res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                return res
            else:
                return get_Gamma(alpha, beta, u, v)


if __name__ == "__main__":
    cost = (torch.Tensor(2, 2).fill_(1) - torch.diag(torch.Tensor(2).fill_(1)))  # .cuda()
    mylayer = WlossLayer(cost)  # .cuda()
    inp = Variable(torch.Tensor([[1, 0], [0.5, 0.5]]), requires_grad=True)  # .cuda()
    ground_true = Variable(torch.Tensor([[0, 1], [0.5, 0.5]]))  # .cuda()

    res, _ = mylayer(inp, ground_true)
    # print(inp.requires_grad, res.requires_grad)
    # print(res, inp)
    mylayer.zero_grad()
    res.backward()
    print("inp's gradient is good:")
    print(inp.grad)

    print("convert to gpu:\n", inp.cuda().grad)
    print("=============================================="
          "\n However, this does not work on pytorch when GPU is enabled")

    cost = (torch.Tensor(2, 2).fill_(1) - torch.diag(torch.Tensor(2).fill_(1))).cuda()
    mylayer = WlossLayer(cost).cuda()
    inp = Variable(torch.Tensor([[1, 0], [0.5, 0.5]]), requires_grad=True).cuda()
    ground_true = Variable(torch.Tensor([[0, 1], [0.5, 0.5]])).cuda()

    opt = optim.SGD([
        {'params': mylayer.parameters()},
    ], lr=1e-2, momentum=0.9)

    res, _ = mylayer(inp, ground_true)
    # print(inp.requires_grad, res.requires_grad)
    # print(res, inp)
    mylayer.zero_grad()
    res.backward()
    print("input's gradient is None!!!!!!!!!!!!!!!!")
    print(inp.grad)
