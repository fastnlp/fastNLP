import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
from torch.nn.parameter import Parameter

from fastNLP.modules.utils import initial_parameter

def default_initializer(hidden_size):
    stdv = 1.0 / math.sqrt(hidden_size)

    def forward(tensor):
        nn.init.uniform_(tensor, -stdv, stdv)

    return forward


def VarMaskedRecurrent(reverse=False):
    def forward(input, hidden, cell, mask):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            if mask is None or mask[i].data.min() > 0.5:
                hidden = cell(input[i], hidden)
            elif mask[i].data.max() > 0.5:
                hidden_next = cell(input[i], hidden)
                # hack to handle LSTM
                if isinstance(hidden, tuple):
                    hx, cx = hidden
                    hp1, cp1 = hidden_next
                    hidden = (hx + (hp1 - hx) * mask[i], cx + (cp1 - cx) * mask[i])
                else:
                    hidden = hidden + (hidden_next - hidden) * mask[i]
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def StackedRNN(inners, num_layers, lstm=False):
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, cells, mask):
        assert (len(cells) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                hy, output = inner(input, hidden[l], cells[l], mask)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def AutogradVarMaskedRNN(num_layers=1, batch_first=False, bidirectional=False, lstm=False):
    rec_factory = VarMaskedRecurrent

    if bidirectional:
        layer = (rec_factory(), rec_factory(reverse=True))
    else:
        layer = (rec_factory(),)

    func = StackedRNN(layer,
                      num_layers,
                      lstm=lstm)

    def forward(input, cells, hidden, mask):
        if batch_first:
            input = input.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        nexth, output = func(input, hidden, cells, mask)

        if batch_first:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


def VarMaskedStep():
    def forward(input, hidden, cell, mask):
        if mask is None or mask.data.min() > 0.5:
            hidden = cell(input, hidden)
        elif mask.data.max() > 0.5:
            hidden_next = cell(input, hidden)
            # hack to handle LSTM
            if isinstance(hidden, tuple):
                hx, cx = hidden
                hp1, cp1 = hidden_next
                hidden = (hx + (hp1 - hx) * mask, cx + (cp1 - cx) * mask)
            else:
                hidden = hidden + (hidden_next - hidden) * mask
        # hack to handle LSTM
        output = hidden[0] if isinstance(hidden, tuple) else hidden

        return hidden, output

    return forward


def StackedStep(layer, num_layers, lstm=False):
    def forward(input, hidden, cells, mask):
        assert (len(cells) == num_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for l in range(num_layers):
            hy, output = layer(input, hidden[l], cells[l], mask)
            next_hidden.append(hy)
            input = output

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(num_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(num_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(num_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def AutogradVarMaskedStep(num_layers=1, lstm=False):
    layer = VarMaskedStep()

    func = StackedStep(layer,
                       num_layers,
                       lstm=lstm)

    def forward(input, cells, hidden, mask):
        nexth, output = func(input, hidden, cells, mask)
        return output, nexth

    return forward


class VarMaskedRNNBase(nn.Module):
    def __init__(self, Cell, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=(0, 0), bidirectional=False, initializer=None,initial_method = None, **kwargs):

        super(VarMaskedRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = False
        num_directions = 2 if bidirectional else 1

        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                cell = self.Cell(layer_input_size, hidden_size, self.bias, p=dropout, initializer=initializer, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions + direction), cell)
        initial_parameter(self, initial_method)
    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def reset_noise(self, batch_size):
        for cell in self.all_cells:
            cell.reset_noise(batch_size)

    def forward(self, input, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.tensor(input.data.new(self.num_layers * num_directions, batch_size, self.hidden_size).zero_(),
                              requires_grad=True)
            if self.lstm:
                hx = (hx, hx)

        func = AutogradVarMaskedRNN(num_layers=self.num_layers,
                                    batch_first=self.batch_first,
                                    bidirectional=self.bidirectional,
                                    lstm=self.lstm)

        self.reset_noise(batch_size)

        output, hidden = func(input, self.all_cells, hx, None if mask is None else mask.view(mask.size() + (1,)))
        return output, hidden

    def step(self, input, hx=None, mask=None):
        '''
        execute one step forward (only for one-directional RNN).
        Args:
            input (batch, input_size): input tensor of this step.
            hx (num_layers, batch, hidden_size): the hidden state of last step.
            mask (batch): the mask tensor of this step.
        Returns:
            output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
            hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
        '''
        assert not self.bidirectional, "step only cannot be applied to bidirectional RNN."
        batch_size = input.size(0)
        if hx is None:
            hx = torch.tensor(input.data.new(self.num_layers, batch_size, self.hidden_size).zero_(), requires_grad=True)
            if self.lstm:
                hx = (hx, hx)

        func = AutogradVarMaskedStep(num_layers=self.num_layers, lstm=self.lstm)

        output, hidden = func(input, self.all_cells, hx, mask)
        return output, hidden


class VarMaskedFastLSTM(VarMaskedRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarMaskedFastLSTM, self).__init__(VarFastLSTMCell, *args, **kwargs)
        self.lstm = True


class VarRNNCellBase(nn.Module):
    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_noise(self, batch_size):
        """
        Should be overriden by all subclasses.
        Args:
            batch_size: (int) batch size of input.
        """
        raise NotImplementedError


class VarFastLSTMCell(VarRNNCellBase):
    """
    A long short-term memory (LSTM) cell with variational dropout.
    .. math::
        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5), initializer=None,initial_method =None):
        super(VarFastLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None
        initial_parameter(self, initial_method)
    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() == 1:
                weight.data.zero_()
            else:
                self.initializer(weight.data)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.data.new(batch_size, self.input_size)
                self.noise_in = torch.tensor(noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in))
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.data.new(batch_size, self.hidden_size)
                self.noise_hidden = torch.tensor(noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden))
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return self.__forward(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.noise_in, self.noise_hidden,
        )

    @staticmethod
    def __forward(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
        if noise_in is not None:
            if input.is_cuda:
                input = input * noise_in.cuda(input.get_device())
            else:
                input = input * noise_in

        if input.is_cuda:
            w_ih = w_ih.cuda(input.get_device())
            w_hh = w_hh.cuda(input.get_device())
            hidden = [h.cuda(input.get_device()) for h in hidden]
            b_ih = b_ih.cuda(input.get_device())
            b_hh = b_hh.cuda(input.get_device())
            igates = F.linear(input, w_ih.cuda(input.get_device()))
            hgates = F.linear(hidden[0], w_hh) if noise_hidden is None \
                else F.linear(hidden[0] * noise_hidden.cuda(input.get_device()), w_hh)
            state = fusedBackend.LSTMFused.apply
            # print("use backend")
            # use some magic function
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

        hx, cx = hidden
        if noise_hidden is not None:
            hx = hx * noise_hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy
