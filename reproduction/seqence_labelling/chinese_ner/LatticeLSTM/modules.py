import torch.nn as nn
import torch
from fastNLP.core.utils import seq_len_to_mask
from utils import better_init_rnn
import numpy as np


class WordLSTMCell_yangjie(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,debug=False, left2right=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super().__init__()
        self.left2right = left2right
        self.debug = debug
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        nn.init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            nn.init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx



        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)

        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell_V0(nn.Module):
    def __init__(self, char_input_size, hidden_size, use_bias=True,debug=False):
        super().__init__()
        self.char_input_size = char_input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(char_input_size, 3 * hidden_size)
        )

        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size)
        )

        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(char_input_size, hidden_size)
        )

        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size)
        )

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)

        self.debug = debug
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        nn.init.orthogonal(self.weight_ih.data)
        nn.init.orthogonal(self.alpha_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        with torch.no_grad():
            self.alpha_weight_hh.set_(alpha_weight_hh_data)

        # The bias is just set to zero vectors.
        if self.use_bias:
            nn.init.constant_(self.bias.data, val=0)
            nn.init.constant_(self.alpha_bias.data, val=0)

    def forward(self, inp, skip_c, skip_count, hx):
        '''

        :param inp: chars B * hidden
        :param skip_c: 由跳边得到的c, B * X * hidden
        :param skip_count: 这个batch中每个example中当前位置的跳边的数量，用于mask
        :param hx:
        :return:
        '''
        max_skip_count = torch.max(skip_count).item()



        if True:
            h_0, c_0 = hx
            batch_size = h_0.size(0)

            bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

            wi = torch.matmul(inp, self.weight_ih)
            wh = torch.matmul(h_0, self.weight_hh)



            i, o, g = torch.split(wh + wi + bias_batch, split_size_or_sections=self.hidden_size, dim=1)

            i = torch.sigmoid(i).unsqueeze(1)
            o = torch.sigmoid(o).unsqueeze(1)
            g = torch.tanh(g).unsqueeze(1)



            alpha_wi = torch.matmul(inp, self.alpha_weight_ih)
            alpha_wi.unsqueeze_(1)

            # alpha_wi = alpha_wi.expand(1,skip_count,self.hidden_size)
            alpha_wh = torch.matmul(skip_c, self.alpha_weight_hh)

            alpha_bias_batch = self.alpha_bias.unsqueeze(0)

            alpha = torch.sigmoid(alpha_wi + alpha_wh + alpha_bias_batch)

            skip_mask = seq_len_to_mask(skip_count,max_len=skip_c.size()[1])

            skip_mask = 1 - skip_mask


            skip_mask = skip_mask.unsqueeze(-1).expand(*skip_mask.size(), self.hidden_size)

            skip_mask = (skip_mask).float()*1e20

            alpha = alpha - skip_mask

            alpha = torch.exp(torch.cat([i, alpha], dim=1))



            alpha_sum = torch.sum(alpha, dim=1, keepdim=True)

            alpha = torch.div(alpha, alpha_sum)

            merge_i_c = torch.cat([g, skip_c], dim=1)

            c_1 = merge_i_c * alpha

            c_1 = c_1.sum(1, keepdim=True)
            # h_1 = o * c_1
            h_1 = o * torch.tanh(c_1)

            return h_1.squeeze(1), c_1.squeeze(1)

        else:

            h_0, c_0 = hx
            batch_size = h_0.size(0)

            bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

            wi = torch.matmul(inp, self.weight_ih)
            wh = torch.matmul(h_0, self.weight_hh)

            i, o, g = torch.split(wh + wi + bias_batch, split_size_or_sections=self.hidden_size, dim=1)

            i = torch.sigmoid(i).unsqueeze(1)
            o = torch.sigmoid(o).unsqueeze(1)
            g = torch.tanh(g).unsqueeze(1)

            c_1 = g
            h_1 = o * c_1

            return h_1,c_1

class MultiInputLSTMCell_V1(nn.Module):
    def __init__(self, char_input_size, hidden_size, use_bias=True,debug=False):
        super().__init__()
        self.char_input_size = char_input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(char_input_size, 3 * hidden_size)
        )

        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size)
        )

        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(char_input_size, hidden_size)
        )

        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size)
        )

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)

        self.debug = debug
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        nn.init.orthogonal(self.weight_ih.data)
        nn.init.orthogonal(self.alpha_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        with torch.no_grad():
            self.alpha_weight_hh.set_(alpha_weight_hh_data)

        # The bias is just set to zero vectors.
        if self.use_bias:
            nn.init.constant_(self.bias.data, val=0)
            nn.init.constant_(self.alpha_bias.data, val=0)

    def forward(self, inp, skip_c, skip_count, hx):
        '''

        :param inp: chars B * hidden
        :param skip_c: 由跳边得到的c, B * X * hidden
        :param skip_count: 这个batch中每个example中当前位置的跳边的数量，用于mask
        :param hx:
        :return:
        '''
        max_skip_count = torch.max(skip_count).item()



        if True:
            h_0, c_0 = hx
            batch_size = h_0.size(0)

            bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

            wi = torch.matmul(inp, self.weight_ih)
            wh = torch.matmul(h_0, self.weight_hh)


            i, o, g = torch.split(wh + wi + bias_batch, split_size_or_sections=self.hidden_size, dim=1)

            i = torch.sigmoid(i).unsqueeze(1)
            o = torch.sigmoid(o).unsqueeze(1)
            g = torch.tanh(g).unsqueeze(1)



            ##basic lstm start

            f = 1 - i
            c_1_basic = f*c_0.unsqueeze(1) + i*g
            c_1_basic = c_1_basic.squeeze(1)





            alpha_wi = torch.matmul(inp, self.alpha_weight_ih)
            alpha_wi.unsqueeze_(1)


            alpha_wh = torch.matmul(skip_c, self.alpha_weight_hh)

            alpha_bias_batch = self.alpha_bias.unsqueeze(0)

            alpha = torch.sigmoid(alpha_wi + alpha_wh + alpha_bias_batch)

            skip_mask = seq_len_to_mask(skip_count,max_len=skip_c.size()[1])

            skip_mask = 1 - skip_mask


            skip_mask = skip_mask.unsqueeze(-1).expand(*skip_mask.size(), self.hidden_size)

            skip_mask = (skip_mask).float()*1e20

            alpha = alpha - skip_mask

            alpha = torch.exp(torch.cat([i, alpha], dim=1))



            alpha_sum = torch.sum(alpha, dim=1, keepdim=True)

            alpha = torch.div(alpha, alpha_sum)

            merge_i_c = torch.cat([g, skip_c], dim=1)

            c_1 = merge_i_c * alpha

            c_1 = c_1.sum(1, keepdim=True)
            # h_1 = o * c_1
            c_1 = c_1.squeeze(1)
            count_select = (skip_count != 0).float().unsqueeze(-1)




            c_1 = c_1*count_select + c_1_basic*(1-count_select)


            o = o.squeeze(1)
            h_1 = o * torch.tanh(c_1)

            return h_1, c_1

class LatticeLSTMLayer_sup_back_V0(nn.Module):
    def __init__(self, char_input_size, word_input_size, hidden_size, left2right,
                 bias=True,device=None,debug=False,skip_before_head=False):
        super().__init__()

        self.skip_before_head = skip_before_head

        self.hidden_size = hidden_size

        self.char_cell = MultiInputLSTMCell_V0(char_input_size, hidden_size, bias,debug)

        self.word_cell = WordLSTMCell_yangjie(word_input_size,hidden_size,bias,debug=self.debug)

        self.word_input_size = word_input_size
        self.left2right = left2right
        self.bias = bias
        self.device = device
        self.debug = debug

    def forward(self, inp, seq_len, skip_sources, skip_words, skip_count, init_state=None):
        '''

        :param inp: batch * seq_len * embedding, chars
        :param seq_len: batch, length of chars
        :param skip_sources: batch * seq_len * X, 跳边的起点
        :param skip_words: batch * seq_len * X * embedding, 跳边的词
        :param lexicon_count: batch * seq_len, count of lexicon per example per position
        :param init_state: the hx of rnn
        :return:
        '''


        if self.left2right:

            max_seq_len = max(seq_len)
            batch_size = inp.size(0)
            c_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)
            h_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)

            for i in range(max_seq_len):
                max_lexicon_count = max(torch.max(skip_count[:, i]).item(), 1)
                h_0, c_0 = h_[:, i, :], c_[:, i, :]

                skip_word_flat = skip_words[:, i, :max_lexicon_count].contiguous()

                skip_word_flat = skip_word_flat.view(batch_size*max_lexicon_count,self.word_input_size)
                skip_source_flat = skip_sources[:, i, :max_lexicon_count].contiguous().view(batch_size, max_lexicon_count)


                index_0 = torch.tensor(range(batch_size)).unsqueeze(1).expand(batch_size,max_lexicon_count)
                index_1 = skip_source_flat

                if not self.skip_before_head:
                    c_x = c_[[index_0, index_1+1]]
                    h_x = h_[[index_0, index_1+1]]
                else:
                    c_x = c_[[index_0,index_1]]
                    h_x = h_[[index_0,index_1]]

                c_x_flat = c_x.view(batch_size*max_lexicon_count,self.hidden_size)
                h_x_flat = h_x.view(batch_size*max_lexicon_count,self.hidden_size)




                c_1_flat = self.word_cell(skip_word_flat,(h_x_flat,c_x_flat))

                c_1_skip = c_1_flat.view(batch_size,max_lexicon_count,self.hidden_size)

                h_1,c_1 = self.char_cell(inp[:,i,:],c_1_skip,skip_count[:,i],(h_0,c_0))


                h_ = torch.cat([h_,h_1.unsqueeze(1)],dim=1)
                c_ = torch.cat([c_, c_1.unsqueeze(1)], dim=1)

            return h_[:,1:],c_[:,1:]
        else:
            mask_for_seq_len = seq_len_to_mask(seq_len)

            max_seq_len = max(seq_len)
            batch_size = inp.size(0)
            c_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)
            h_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)

            for i in reversed(range(max_seq_len)):
                max_lexicon_count = max(torch.max(skip_count[:, i]).item(), 1)



                h_0, c_0 = h_[:, 0, :], c_[:, 0, :]

                skip_word_flat = skip_words[:, i, :max_lexicon_count].contiguous()

                skip_word_flat = skip_word_flat.view(batch_size*max_lexicon_count,self.word_input_size)
                skip_source_flat = skip_sources[:, i, :max_lexicon_count].contiguous().view(batch_size, max_lexicon_count)


                index_0 = torch.tensor(range(batch_size)).unsqueeze(1).expand(batch_size,max_lexicon_count)
                index_1 = skip_source_flat-i

                if not self.skip_before_head:
                    c_x = c_[[index_0, index_1-1]]
                    h_x = h_[[index_0, index_1-1]]
                else:
                    c_x = c_[[index_0,index_1]]
                    h_x = h_[[index_0,index_1]]

                c_x_flat = c_x.view(batch_size*max_lexicon_count,self.hidden_size)
                h_x_flat = h_x.view(batch_size*max_lexicon_count,self.hidden_size)




                c_1_flat = self.word_cell(skip_word_flat,(h_x_flat,c_x_flat))

                c_1_skip = c_1_flat.view(batch_size,max_lexicon_count,self.hidden_size)

                h_1,c_1 = self.char_cell(inp[:,i,:],c_1_skip,skip_count[:,i],(h_0,c_0))


                h_1_mask = h_1.masked_fill(1-mask_for_seq_len[:,i].unsqueeze(-1),0)
                c_1_mask = c_1.masked_fill(1 - mask_for_seq_len[:, i].unsqueeze(-1), 0)


                h_ = torch.cat([h_1_mask.unsqueeze(1),h_],dim=1)
                c_ = torch.cat([c_1_mask.unsqueeze(1),c_], dim=1)

            return h_[:,:-1],c_[:,:-1]

class LatticeLSTMLayer_sup_back_V1(nn.Module):
    # V1与V0的不同在于，V1在当前位置完全无lexicon匹配时，会采用普通的lstm计算公式，
    # 普通的lstm计算公式与杨杰实现的lattice lstm在lexicon数量为0时不同
    def __init__(self, char_input_size, word_input_size, hidden_size, left2right,
                 bias=True,device=None,debug=False,skip_before_head=False):
        super().__init__()

        self.debug = debug

        self.skip_before_head = skip_before_head

        self.hidden_size = hidden_size

        self.char_cell = MultiInputLSTMCell_V1(char_input_size, hidden_size, bias,debug)

        self.word_cell = WordLSTMCell_yangjie(word_input_size,hidden_size,bias,debug=self.debug)

        self.word_input_size = word_input_size
        self.left2right = left2right
        self.bias = bias
        self.device = device

    def forward(self, inp, seq_len, skip_sources, skip_words, skip_count, init_state=None):
        '''

        :param inp: batch * seq_len * embedding, chars
        :param seq_len: batch, length of chars
        :param skip_sources: batch * seq_len * X, 跳边的起点
        :param skip_words: batch * seq_len * X * embedding_size, 跳边的词
        :param lexicon_count: batch * seq_len,
        lexicon_count[i,j]为第i个例子以第j个位子为结尾匹配到的词的数量
        :param init_state: the hx of rnn
        :return:
        '''


        if self.left2right:

            max_seq_len = max(seq_len)
            batch_size = inp.size(0)
            c_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)
            h_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)

            for i in range(max_seq_len):
                max_lexicon_count = max(torch.max(skip_count[:, i]).item(), 1)
                h_0, c_0 = h_[:, i, :], c_[:, i, :]

                #为了使rnn能够计算B*lexicon_count*embedding_size的张量，需要将其reshape成二维张量
                #为了匹配pytorch的[]取址方式，需要将reshape成二维张量

                skip_word_flat = skip_words[:, i, :max_lexicon_count].contiguous()

                skip_word_flat = skip_word_flat.view(batch_size*max_lexicon_count,self.word_input_size)
                skip_source_flat = skip_sources[:, i, :max_lexicon_count].contiguous().view(batch_size, max_lexicon_count)


                index_0 = torch.tensor(range(batch_size)).unsqueeze(1).expand(batch_size,max_lexicon_count)
                index_1 = skip_source_flat


                if not self.skip_before_head:
                    c_x = c_[[index_0, index_1+1]]
                    h_x = h_[[index_0, index_1+1]]
                else:
                    c_x = c_[[index_0,index_1]]
                    h_x = h_[[index_0,index_1]]

                c_x_flat = c_x.view(batch_size*max_lexicon_count,self.hidden_size)
                h_x_flat = h_x.view(batch_size*max_lexicon_count,self.hidden_size)



                c_1_flat = self.word_cell(skip_word_flat,(h_x_flat,c_x_flat))

                c_1_skip = c_1_flat.view(batch_size,max_lexicon_count,self.hidden_size)

                h_1,c_1 = self.char_cell(inp[:,i,:],c_1_skip,skip_count[:,i],(h_0,c_0))


                h_ = torch.cat([h_,h_1.unsqueeze(1)],dim=1)
                c_ = torch.cat([c_, c_1.unsqueeze(1)], dim=1)

            return h_[:,1:],c_[:,1:]
        else:
            mask_for_seq_len = seq_len_to_mask(seq_len)

            max_seq_len = max(seq_len)
            batch_size = inp.size(0)
            c_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)
            h_ = torch.zeros(size=[batch_size, 1, self.hidden_size], requires_grad=True).to(self.device)

            for i in reversed(range(max_seq_len)):
                max_lexicon_count = max(torch.max(skip_count[:, i]).item(), 1)


                h_0, c_0 = h_[:, 0, :], c_[:, 0, :]

                skip_word_flat = skip_words[:, i, :max_lexicon_count].contiguous()

                skip_word_flat = skip_word_flat.view(batch_size*max_lexicon_count,self.word_input_size)
                skip_source_flat = skip_sources[:, i, :max_lexicon_count].contiguous().view(batch_size, max_lexicon_count)


                index_0 = torch.tensor(range(batch_size)).unsqueeze(1).expand(batch_size,max_lexicon_count)
                index_1 = skip_source_flat-i

                if not self.skip_before_head:
                    c_x = c_[[index_0, index_1-1]]
                    h_x = h_[[index_0, index_1-1]]
                else:
                    c_x = c_[[index_0,index_1]]
                    h_x = h_[[index_0,index_1]]

                c_x_flat = c_x.view(batch_size*max_lexicon_count,self.hidden_size)
                h_x_flat = h_x.view(batch_size*max_lexicon_count,self.hidden_size)




                c_1_flat = self.word_cell(skip_word_flat,(h_x_flat,c_x_flat))



                c_1_skip = c_1_flat.view(batch_size,max_lexicon_count,self.hidden_size)

                h_1,c_1 = self.char_cell(inp[:,i,:],c_1_skip,skip_count[:,i],(h_0,c_0))


                h_1_mask = h_1.masked_fill(1-mask_for_seq_len[:,i].unsqueeze(-1),0)
                c_1_mask = c_1.masked_fill(1 - mask_for_seq_len[:, i].unsqueeze(-1), 0)


                h_ = torch.cat([h_1_mask.unsqueeze(1),h_],dim=1)
                c_ = torch.cat([c_1_mask.unsqueeze(1),c_], dim=1)



            return h_[:,:-1],c_[:,:-1]




