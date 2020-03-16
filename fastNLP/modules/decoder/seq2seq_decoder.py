# coding=utf-8
import torch
from torch import nn
import abc
import torch.nn.functional as F
from fastNLP.embeddings import StaticEmbedding
import numpy as np
from typing import Union, Tuple
from fastNLP.embeddings import get_embeddings
from torch.nn import LayerNorm
import math
from reproduction.Summarization.Baseline.tools.PositionEmbedding import \
    get_sinusoid_encoding_table  # todo: 应该将position embedding移到core


class Past:
    def __init__(self):
        pass

    @abc.abstractmethod
    def num_samples(self):
        pass


class TransformerPast(Past):
    def __init__(self, encoder_outputs: torch.Tensor = None, encoder_mask: torch.Tensor = None,
                 encoder_key: torch.Tensor = None, encoder_value: torch.Tensor = None,
                 decoder_prev_key: torch.Tensor = None, decoder_prev_value: torch.Tensor = None):
        """

        :param encoder_outputs: (batch,src_seq_len,dim)
        :param encoder_mask: (batch,src_seq_len)
        :param encoder_key: list of (batch, src_seq_len, dim)
        :param encoder_value:
        :param decoder_prev_key:
        :param decoder_prev_value:
        """
        self.encoder_outputs = encoder_outputs
        self.encoder_mask = encoder_mask
        self.encoder_kv = encoder_key
        self.encoder_value = encoder_value
        self.decoder_prev_key = decoder_prev_key
        self.decoder_prev_value = decoder_prev_value

    def num_samples(self):
        if self.encoder_outputs is not None:
            return self.encoder_outputs.size(0)
        return None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def reorder_past(self, indices: torch.LongTensor, past: Past) -> Past:
        """
        根据indices中的index，将past的中状态置为正确的顺序

        :param torch.LongTensor indices:
        :param Past past:
        :return:
        """
        raise NotImplemented

    def decode_one(self, *args, **kwargs) -> Tuple[torch.Tensor, Past]:
        """
        当模型进行解码时，使用这个函数。只返回一个batch_size x vocab_size的结果。需要考虑一种特殊情况，即tokens长度不是1，即给定了
            解码句子开头的情况，这种情况需要查看Past中是否正确计算了decode的状态

        :return:
        """
        raise NotImplemented


class DecoderMultiheadAttention(nn.Module):
    """
    Transformer Decoder端的multihead layer
    相比原版的Multihead功能一致，但能够在inference时加速
    参考fairseq
    """

    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.0, layer_idx: int = None):
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_model // n_head
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.reset_parameters()

    def forward(self, query, key, value, self_attn_mask=None, encoder_attn_mask=None, past=None, inference=False):
        """

        :param query: (batch, seq_len, dim)
        :param key: (batch, seq_len, dim)
        :param value: (batch, seq_len, dim)
        :param self_attn_mask: None or ByteTensor (1, seq_len, seq_len)
        :param encoder_attn_mask: (batch, src_len) ByteTensor
        :param past: required for now
        :param inference:
        :return: x和attention weight
        """
        if encoder_attn_mask is not None:
            assert self_attn_mask is None
        assert past is not None, "Past is required for now"
        is_encoder_attn = True if encoder_attn_mask is not None else False

        q = self.q_proj(query)  # (batch,q_len,dim)
        q *= self.scaling
        k = v = None
        prev_k = prev_v = None

        if inference and is_encoder_attn and past.encoder_key[self.layer_idx] is not None:
            k = past.encoder_key[self.layer_idx]  # (batch,k_len,dim)
            v = past.encoder_value[self.layer_idx]  # (batch,v_len,dim)
        else:
            if inference and not is_encoder_attn and past.decoder_prev_key[self.layer_idx] is not None:
                prev_k = past.decoder_prev_key[self.layer_idx]  # (batch, seq_len, dim)
                prev_v = past.decoder_prev_value[self.layer_idx]

        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)
        if prev_k is not None:
            k = torch.cat((prev_k, k), dim=1)
            v = torch.cat((prev_v, v), dim=1)

        # 更新past
        if inference and is_encoder_attn and past.encoder_key[self.layer_idx] is None:
            past.encoder_key[self.layer_idx] = k
            past.encoder_value[self.layer_idx] = v
        if inference and not is_encoder_attn:
            past.decoder_prev_key[self.layer_idx] = prev_k
            past.decoder_prev_value[self.layer_idx] = prev_v

        batch_size, q_len, d_model = query.size()
        k_len, v_len = key.size(1), value.size(1)
        q = q.contiguous().view(batch_size, q_len, self.n_head, self.head_dim)
        k = k.contiguous().view(batch_size, k_len, self.n_head, self.head_dim)
        v = v.contiguous().view(batch_size, v_len, self.n_head, self.head_dim)

        attn_weights = torch.einsum('bqnh,bknh->bqkn', q, k)  # bs,q_len,k_len,n_head
        mask = encoder_attn_mask if is_encoder_attn else self_attn_mask
        if mask is not None:
            if len(mask.size()) == 2:  # 是encoder mask, batch,src_len/k_len
                mask = mask[:, None, :, None]
            else:  # (1, seq_len, seq_len)
                mask = mask[...:None]
            _mask = mask

            attn_weights = attn_weights.masked_fill(_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('bqkn,bknh->bqnh', attn_weights, v)  # batch,q_len,n_head,head_dim
        output = output.view(batch_size, q_len, -1)
        output = self.out_proj(output)  # batch,q_len,dim

        return output, attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj)
        nn.init.xavier_uniform_(self.k_proj)
        nn.init.xavier_uniform_(self.v_proj)
        nn.init.xavier_uniform_(self.out_proj)


class TransformerSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 layer_idx: int = None):
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx  # 记录layer的层索引，以方便获取past的信息

        self.self_attn = DecoderMultiheadAttention(d_model, n_head, dropout, layer_idx)
        self.self_attn_layer_norm = LayerNorm(d_model)

        self.encoder_attn = DecoderMultiheadAttention(d_model, n_head, dropout, layer_idx)
        self.encoder_attn_layer_norm = LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

        self.final_layer_norm = LayerNorm(self.d_model)

    def forward(self, x, encoder_outputs, self_attn_mask=None, encoder_attn_mask=None, past=None, inference=False):
        """

        :param x: (batch, seq_len, dim)
        :param encoder_outputs: (batch,src_seq_len,dim)
        :param self_attn_mask:
        :param encoder_attn_mask:
        :param past:
        :param inference:
        :return:
        """
        if inference:
            assert past is not None, "Past is required when inference"

        # self attention part
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              self_attn_mask=self_attn_mask,
                              past=past,
                              inference=inference)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # encoder attention part
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weight = self.encoder_attn(query=x,
                                           key=past.encoder_outputs,
                                           value=past.encoder_outputs,
                                           encoder_attn_mask=past.encoder_mask,
                                           past=past,
                                           inference=inference)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight


class TransformerSeq2SeqDecoder(Decoder):
    def __init__(self, embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray], num_layers: int = 6,
                 d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 output_embed: Union[Tuple[int, int], int, nn.Module, torch.Tensor, np.ndarray] = None,
                 bind_input_output_embed=False):
        """

        :param embed: decoder端输入的embedding
        :param num_layers: Transformer Decoder层数
        :param d_model: Transformer参数
        :param n_head: Transformer参数
        :param dim_ff: Transformer参数
        :param dropout:
        :param output_embed: 输出embedding
        :param bind_input_output_embed: 是否共享输入输出的embedding权重
        """
        super(TransformerSeq2SeqDecoder, self).__init__()
        self.token_embed = get_embeddings(embed)
        self.dropout = dropout

        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx)
                                           for layer_idx in range(num_layers)])

        if isinstance(output_embed, int):
            output_embed = (output_embed, d_model)
            output_embed = get_embeddings(output_embed)
        elif output_embed is not None:
            assert not bind_input_output_embed, "When `output_embed` is not None, " \
                                                "`bind_input_output_embed` must be False."
            if isinstance(output_embed, StaticEmbedding):
                for i in self.token_embed.words_to_words:
                    assert i == self.token_embed.words_to_words[i], "The index does not match."
                output_embed = self.token_embed.embedding.weight
            else:
                output_embed = get_embeddings(output_embed)
        else:
            if not bind_input_output_embed:
                raise RuntimeError("You have to specify output embedding.")

        # todo: 由于每个模型都有embedding的绑定或其他操作，建议挪到外部函数以减少冗余，可参考fairseq
        self.pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position=1024, d_hid=d_model, padding_idx=0),
            freeze=True
        )

        if bind_input_output_embed:
            assert output_embed is None, "When `bind_input_output_embed=True`, `output_embed` must be None"
            if isinstance(self.token_embed, StaticEmbedding):
                for i in self.token_embed.words_to_words:
                    assert i == self.token_embed.words_to_words[i], "The index does not match."
            self.output_embed = nn.Parameter(self.token_embed.weight.transpose(0, 1))
        else:
            if isinstance(output_embed, nn.Embedding):
                self.output_embed = nn.Parameter(output_embed.weight.transpose(0, 1))
            else:
                self.output_embed = output_embed.transpose(0, 1)
            self.output_hidden_size = self.output_embed.size(0)

        self.embed_scale = math.sqrt(d_model)

    def forward(self, tokens, past, return_attention=False, inference=False):
        """

        :param tokens: torch.LongTensor, tokens: batch_size x decode_len
        :param past: TransformerPast: 包含encoder输出及mask，在inference阶段保存了上一时刻的key和value以减少矩阵运算
        :param return_attention:
        :param inference: 是否在inference阶段
        :return:
        """
        assert past is not None
        batch_size, decode_len = tokens.size()
        device = tokens.device
        if not inference:
            self_attn_mask = self._get_triangle_mask(decode_len)
            self_attn_mask = self_attn_mask.to(device)[None, :, :]  # 1,seq,seq
        else:
            self_attn_mask = None
        tokens = self.token_embed(tokens) * self.embed_scale  # bs,decode_len,embed_dim
        pos = self.pos_embed(tokens)  # bs,decode_len,embed_dim
        tokens = pos + tokens
        if inference:
            tokens = tokens[:, -1, :]

        x = F.dropout(tokens, p=self.dropout, training=self.training)
        for layer in self.layer_stacks:
            x, attn_weight = layer(x, past.encoder_outputs, self_attn_mask=self_attn_mask,
                                   encoder_attn_mask=past.encoder_mask, past=past, inference=inference)

        output = torch.matmul(x, self.output_embed)

        if return_attention:
            return output, attn_weight
        return output

    @torch.no_grad()
    def decode_one(self, tokens, past) -> Tuple[torch.Tensor, Past]:
        """
        # todo: 对于transformer而言，因为position的原因，需要输入整个prefix序列，因此lstm的decode one和beam search需要改一下，以统一接口
        # todo: 是否不需要return past？ 因为past已经被改变了，不需要显式return？
        :param tokens: torch.LongTensor (batch_size,1)
        :param past: TransformerPast
        :return:
        """
        output = self.forward(tokens, past, inference=True)  # batch,1,vocab_size
        return output.squeeze(1), past

    def _get_triangle_mask(self, max_seq_len):
        tensor = torch.ones(max_seq_len, max_seq_len)
        return torch.tril(tensor).byte()
