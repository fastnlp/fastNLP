import torch
from torch import nn
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from .seq2seq_model import Seq2SeqModel
from ...modules.torch.generator.seq2seq_generator import SequenceGenerator


__all__ = ['SequenceGeneratorModel']


class SequenceGeneratorModel(nn.Module):
    """
    通过使用本模型封装 seq2seq_model 使得其既可以用于训练也可以用于生成。训练的时候，本模型的 :meth:`forward` 函数会被调用，
    生成的时候本模型的 :meth:`predict` 函数会被调用。

    :param seq2seq_model: 序列到序列模型
    :param bos_token_id: 句子开头的 token id
    :param eos_token_id: 句子结束的 token id
    :param max_length: 生成句子的最大长度, 每句话的 decode 长度为 ``max_length + max_len_a * src_len``
    :param max_len_a: 每句话的 decode 长度为 ``max_length + max_len_a*src_len``。如果不为 0，需要保证 State 中包含 encoder_mask
    :param num_beams: **beam search** 的大小
    :param do_sample: 是否通过采样的方式生成
    :param temperature: 只有在 do_sample 为 ``True`` 才有意义
    :param top_k: 只从 ``top_k`` 中采样
    :param top_p: 只从 ``top_p`` 的 token 中采样（ **nucleus sampling** ）
    :param repetition_penalty: 多大程度上惩罚重复的 token
    :param length_penalty: 对长度的惩罚，**小于 1** 鼓励长句，**大于 1** 鼓励短句
    :param pad_token_id: 当某句话生成结束之后，之后生成的内容用 ``pad_token_id`` 补充
    """

    def __init__(self, seq2seq_model: Seq2SeqModel, bos_token_id: int=None, eos_token_id: int=None, max_length: int=30,
                 max_len_a: float=0.0, num_beams: int=1, do_sample: bool=True, temperature: float=1.0, top_k: int=50,
                 top_p: float=1.0, repetition_penalty: float=1, length_penalty: float=1.0, pad_token_id: int=0):
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                           pad_token_id=pad_token_id)

    def forward(self, src_tokens: "torch.LongTensor", tgt_tokens: "torch.LongTensor",
                src_seq_len: "torch.LongTensor"=None, tgt_seq_len: "torch.LongTensor"=None):
        """
        调用 seq2seq_model 的 :meth:`forward` 。

        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param tgt_tokens: target 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :param tgt_seq_len: target的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'pred': torch.Tensor}``, 其中 ``pred`` 的形状为 ``[batch_size, max_len, vocab_size]``
        """
        return self.seq2seq_model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)

    def train_step(self, src_tokens: "torch.LongTensor", tgt_tokens: "torch.LongTensor",
                    src_seq_len: "torch.LongTensor"=None, tgt_seq_len: "torch.LongTensor"=None):
        """
        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param tgt_tokens: target 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :param tgt_seq_len: target的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'loss': torch.Tensor}``
        """
        res = self(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)
        pred = res['pred']
        if tgt_seq_len is not None:
            mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1))
            tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        loss = F.cross_entropy(pred[:, :-1].transpose(1, 2), tgt_tokens[:, 1:])
        return {'loss': loss}

    def evaluate_step(self, src_tokens: "torch.LongTensor", src_seq_len: "torch.LongTensor"=None):
        """
        给定 source 的内容，输出 generate 的内容。

        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'pred': torch.Tensor}`` ，表示生成结果
        """
        state = self.seq2seq_model.prepare_state(src_tokens, src_seq_len)
        result = self.generator.generate(state)
        return {'pred': result}
