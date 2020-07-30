r"""undocumented"""

import torch
from torch import nn
from .seq2seq_model import Seq2SeqModel
from ..modules.generator.seq2seq_generator import SequenceGenerator


class SequenceGeneratorModel(nn.Module):
    """
    用于封装Seq2SeqModel使其可以做生成任务

    """

    def __init__(self, seq2seq_model: Seq2SeqModel, bos_token_id, eos_token_id=None, max_length=30, max_len_a=0.0,
                 num_beams=1, do_sample=True, temperature=1.0, top_k=50, top_p=1.0,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0):
        """

        :param Seq2SeqModel seq2seq_model: 序列到序列模型
        :param int,None bos_token_id: 句子开头的token id
        :param int,None eos_token_id: 句子结束的token id
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                           pad_token_id=pad_token_id)

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None):
        """
        透传调用seq2seq_model的forward

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.seq2seq_model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)

    def predict(self, src_tokens, src_seq_len=None):
        """
        给定source的内容，输出generate的内容

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        state = self.seq2seq_model.prepare_state(src_tokens, src_seq_len)
        result = self.generator.generate(state)
        return {'pred': result}
