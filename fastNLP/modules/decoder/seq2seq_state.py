r"""
每个Decoder都有对应的State用来记录encoder的输出以及Decode的历史记录

"""

__all__ = [
    'State',
    "LSTMState",
    "TransformerState"
]

from typing import Union
import torch


class State:
    def __init__(self, encoder_output=None, encoder_mask=None, **kwargs):
        """
        每个Decoder都有对应的State对象用来承载encoder的输出以及当前时刻之前的decode状态。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor, 默认其中第一维是batch
            维度
        :param Union[torch.Tensor, list, tuple] encoder_mask: 如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch
            维度
        :param kwargs:
        """
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):
        """
        返回的State中包含的是多少个sample的encoder状态，主要用于Generate的时候确定batch的大小。

        :return:
        """
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        """
        当前Decode到哪个token了，decoder只会从decode_length之后的token开始decode, 为0说明还没开始decode。

        :return:
        """
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)


class LSTMState(State):
    def __init__(self, encoder_output, encoder_mask, hidden, cell):
        """
        LSTMDecoder对应的State，保存encoder的输出以及LSTM解码过程中的一些中间状态

        :param torch.FloatTensor encoder_output: bsz x src_seq_len x encode_output_size，encoder的输出
        :param torch.BoolTensor encoder_mask: bsz x src_seq_len, 为0的地方是padding
        :param torch.FloatTensor hidden: num_layers x bsz x hidden_size, 上个时刻的hidden状态
        :param torch.FloatTensor cell: num_layers x bsz x hidden_size, 上个时刻的cell状态
        """
        super().__init__(encoder_output, encoder_mask)
        self.hidden = hidden
        self.cell = cell
        self._input_feed = hidden[0]  # 默认是上一个时刻的输出

    @property
    def input_feed(self):
        """
        LSTMDecoder中每个时刻的输入会把上个token的embedding和input_feed拼接起来输入到下个时刻，在LSTMDecoder不使用attention时，
            input_feed即上个时刻的hidden state, 否则是attention layer的输出。
        :return: torch.FloatTensor, bsz x hidden_size
        """
        return self._input_feed

    @input_feed.setter
    def input_feed(self, value):
        self._input_feed = value

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.hidden = self._reorder_state(self.hidden, indices, dim=1)
        self.cell = self._reorder_state(self.cell, indices, dim=1)
        if self.input_feed is not None:
            self.input_feed = self._reorder_state(self.input_feed, indices, dim=0)


class TransformerState(State):
    def __init__(self, encoder_output, encoder_mask, num_decoder_layer):
        """
        与TransformerSeq2SeqDecoder对应的State，

        :param torch.FloatTensor encoder_output: bsz x encode_max_len x encoder_output_size, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x encode_max_len 为1的地方需要attend
        :param int num_decoder_layer: decode有多少层
        """
        super().__init__(encoder_output, encoder_mask)
        self.encoder_key = [None] * num_decoder_layer  # 每一个元素 bsz x encoder_max_len x key_dim
        self.encoder_value = [None] * num_decoder_layer  # 每一个元素 bsz x encoder_max_len x value_dim
        self.decoder_prev_key = [None] * num_decoder_layer  # 每一个元素 bsz x decode_length x key_dim
        self.decoder_prev_value = [None] * num_decoder_layer  # 每一个元素 bsz x decode_length x key_dim

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_prev_key = self._reorder_state(self.decoder_prev_key, indices)
        self.decoder_prev_value = self._reorder_state(self.decoder_prev_value, indices)

    @property
    def decode_length(self):
        if self.decoder_prev_key[0] is not None:
            return self.decoder_prev_key[0].size(1)
        return 0


