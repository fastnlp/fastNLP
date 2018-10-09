import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

# from fastNLP.modules.utils import initial_parameter

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


class VarRnnCellWrapper(nn.Module):
    def __init__(self, cell, hidden_size, input_p, hidden_p):
        super(VarRnnCellWrapper, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.input_p = input_p
        self.hidden_p = hidden_p

    def forward(self, input, hidden):
        """
        :param input: [seq_len, batch_size, input_size]
        :param hidden: for LSTM, tuple of (h_0, c_0), [batch_size, hidden_size]
                       for other RNN, h_0, [batch_size, hidden_size]

        :return output: [seq_len, bacth_size, hidden_size]
                hidden: for LSTM, tuple of (h_n, c_n), [batch_size, hidden_size]
                        for other RNN, h_n, [batch_size, hidden_size]
        """
        is_lstm = isinstance(hidden, tuple)
        _, batch_size, input_size = input.shape
        mask_x = input.new_ones((batch_size, input_size))
        mask_h = input.new_ones((batch_size, self.hidden_size))
        nn.functional.dropout(mask_x, p=self.input_p, training=self.training, inplace=True)
        nn.functional.dropout(mask_h, p=self.hidden_p, training=self.training, inplace=True)

        input_x = input * mask_x.unsqueeze(0)
        output_list = []
        for x in input_x:
            if is_lstm:
                hx, cx = hidden
                hidden = (hx * mask_h, cx)
            else:
                hidden *= mask_h
            hidden = self.cell(x, hidden)
            output_list.append(hidden[0] if is_lstm else hidden)
        output = torch.stack(output_list, dim=0)
        return output, hidden


class VarRNNBase(nn.Module):
    def __init__(self, mode, Cell, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False,
                 input_dropout=0, hidden_dropout=0, bidirectional=False):
        super(VarRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self._all_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
                cell = Cell(input_size, self.hidden_size, bias)
                self._all_cells.append(VarRnnCellWrapper(cell, self.hidden_size, input_dropout, hidden_dropout))

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        is_lstm = (self.mode == "LSTM")
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = input.new_zeros(self.num_layers * self.num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
            if is_lstm:
                hx = (hx, hx)

        if self.batch_first:
            input = input.transpose(0, 1)

        hidden_list = []
        for layer in range(self.num_layers):
            output_list = []
            for direction in range(self.num_directions):
                input_x = input if direction == 0 else input.flip(0)
                idx = self.num_directions * layer + direction
                cell = self._all_cells[idx]
                output_x, hidden_x = cell(input_x, (hx[0][idx], hx[1][idx]) if is_lstm else hx[idx])
                output_list.append(output_x if direction == 0 else output_x.flip(0))
                hidden_list.append(hidden_x)
            input = torch.cat(output_list, dim=-1)
            
        output = input.transpose(0, 1) if self.batch_first else input
        if is_lstm:
            h_list, c_list = zip(*hidden_list)
            hn = torch.stack(h_list, dim=0)
            cn = torch.stack(c_list, dim=0)
            hidden = (hn, cn)
        else:
            hidden = torch.stack(hidden_list, dim=0)
        
        if is_packed:
            output = PackedSequence(output, batch_sizes)

        return output, hidden 


class VarLSTM(VarRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarLSTM, self).__init__(mode="LSTM", Cell=nn.LSTMCell, *args, **kwargs)


if __name__ == '__main__':
    net = VarLSTM(input_size=10, hidden_size=20, num_layers=3, batch_first=True, bidirectional=True, input_dropout=0.33, hidden_dropout=0.33)
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=3, batch_first=True, bidirectional=True)
    x = torch.randn(2, 8, 10)
    y, hidden = net(x)
    y0, h0 = lstm(x)
    print(y.shape)
    print(y0.shape)
    print(y)
    print(hidden[0])
    print(hidden[0].shape)
    print(y0)
    print(h0[0])
    print(h0[0].shape)