r"""
每个Decoder都有对应的State用来记录encoder的输出以及Decode的历史记录

"""

__all__ = [
    'State',
    "LSTMState",
    "TransformerState"
]

from typing import Union, List, Tuple
import torch


class State:
    """
    每个 ``Decoder`` 都有对应的 :class:`State` 对象用来承载 ``encoder`` 的输出以及当前时刻之前的 ``decode`` 状态。

    :param encoder_output: 如果不为 ``None`` ，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 ``batch_size``
        维度
    :param encoder_mask: 如果部位 ``None``，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 ``batch_size``
        维度
    :param kwargs:
    """
    def __init__(self, encoder_output: Union[torch.Tensor, List, Tuple]=None, 
                encoder_mask: Union[torch.Tensor, List, Tuple]=None, **kwargs):
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):
        """
        返回的 State 中包含的是多少个 sample 的 encoder 状态，主要用于 Generate 的时候确定 batch_size 的大小。
        """
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        """
        当前 Decode 到哪个 token 了，decoder 只会从 decode_length 之后的 token 开始 decode, 为 **0** 说明还没开始 decode。
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
    """
    :class:`~fastNLP.modules.torch.decoder.LSTMSeq2SeqDecoder` 对应的 :class:`State`，保存 ``encoder`` 的输出以及 ``LSTM`` 解码过程中的一些中间状态

    :param encoder_output: ``encoder`` 的输出，形状为 ``[batch_size, src_seq_len, encode_output_size]``
    :param encoder_mask: 掩码，形状为 ``[batch_size, src_seq_len]``，为 **1** 的地方表示需要 attend
    :param hidden: 上个时刻的 ``hidden`` 状态，形状为 ``[num_layers, batch_size, hidden_size]``
    :param cell: 上个时刻的 ``cell`` 状态，形状为 ``[num_layers, batch_size, hidden_size]``
    """
    def __init__(self, encoder_output: torch.FloatTensor, encoder_mask: torch.BoolTensor, hidden: torch.FloatTensor, cell: torch.FloatTensor):
        super().__init__(encoder_output, encoder_mask)
        self.hidden = hidden
        self.cell = cell
        self._input_feed = hidden[0]  # 默认是上一个时刻的输出

    @property
    def input_feed(self) -> torch.FloatTensor:
        """
        :class:`~fastNLP.modules.torch.decoder.LSTMSeq2SeqDecoder` 中每个时刻的输入会把上个 token 的 embedding 和 ``input_feed`` 拼接起来输入到下个时刻，在
        :class:`~fastNLP.modules.torch.decoder.LSTMSeq2SeqDecoder` 不使用 ``attention`` 时，``input_feed`` 即上个时刻的 ``hidden state``，否则是 ``attention layer`` 的输出。
        
        :return: ``[batch_size, hidden_size]``
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
    """
    与 :class:`~fastNLP.modules.torch.decoder.TransformerSeq2SeqDecoder` 对应的 :class:`State`。

    :param encoder_output: ``encoder`` 的输出，形状为 ``[batch_size, encode_max_len, encode_output_size]``，
    :param encoder_mask: 掩码，形状为 ``[batch_size, encode_max_len]``，为 **1** 的地方表示需要 attend
    :param num_decoder_layer: decoder 层数
    """
    def __init__(self, encoder_output: torch.FloatTensor, encoder_mask: torch.FloatTensor, num_decoder_layer: int):
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


