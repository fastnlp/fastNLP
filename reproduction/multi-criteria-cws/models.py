import fastNLP
import torch
import math
from fastNLP.modules.encoder.transformer import TransformerEncoder
from fastNLP.modules.decoder.crf import ConditionalRandomField
from fastNLP import Const
import copy
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import transformer


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(
        self,
        task_size,
        d_model,
        word_embedding=None,
        bi_embedding=None,
        word_size=None,
        freeze=True,
    ):
        super(Embedding, self).__init__()
        self.task_size = task_size
        self.embed_dim = 0

        self.task_embed = nn.Embedding(task_size, d_model)
        if word_embedding is not None:
            # self.uni_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=freeze)
            # self.embed_dim+=word_embedding.shape[1]
            self.uni_embed = word_embedding
            self.embed_dim += word_embedding.embedding_dim
        else:
            if bi_embedding is not None:
                self.embed_dim += bi_embedding.shape[1]
            else:
                self.embed_dim = d_model
            assert word_size is not None
            self.uni_embed = Embedding(word_size, self.embed_dim)

        if bi_embedding is not None:
            # self.bi_embed = nn.Embedding.from_pretrained(torch.FloatTensor(bi_embedding), freeze=freeze)
            # self.embed_dim += bi_embedding.shape[1]*2
            self.bi_embed = bi_embedding
            self.embed_dim += bi_embedding.embedding_dim * 2

        print("Trans Freeze", freeze, self.embed_dim)

        if d_model != self.embed_dim:
            self.F = nn.Linear(self.embed_dim, d_model)
        else:
            self.F = None

        self.d_model = d_model

    def forward(self, task, uni, bi1=None, bi2=None):
        y_task = self.task_embed(task[:, 0:1])
        y = self.uni_embed(uni[:, 1:])
        if bi1 is not None:
            assert self.bi_embed is not None

            y = torch.cat([y, self.bi_embed(bi1), self.bi_embed(bi2)], dim=-1)
            # y2=self.bi_embed(bi)
            # y=torch.cat([y,y2[:,:-1,:],y2[:,1:,:]],dim=-1)

        # y=torch.cat([y_task,y],dim=1)
        if self.F is not None:
            y = self.F(y)
        y = torch.cat([y_task, y], dim=1)
        return y * math.sqrt(self.d_model)


def seq_len_to_mask(seq_len, max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert (
            len(np.shape(seq_len)) == 1
        ), f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert (
            seq_len.dim() == 1
        ), f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


class CWSModel(nn.Module):
    def __init__(self, encoder, src_embed, position, d_model, tag_size, crf=None):
        super(CWSModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.pos = copy.deepcopy(position)
        self.proj = nn.Linear(d_model, tag_size)
        self.tag_size = tag_size
        if crf is None:
            self.crf = None
            self.loss_f = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        else:
            print("crf")
            trans = fastNLP.modules.decoder.crf.allowed_transitions(
                crf, encoding_type="bmes"
            )
            self.crf = ConditionalRandomField(tag_size, allowed_transitions=trans)
        # self.norm=nn.LayerNorm(d_model)

    def forward(self, task, uni, seq_len, bi1=None, bi2=None, tags=None):
        # mask=fastNLP.core.utils.seq_len_to_mask(seq_len,uni.size(1)) # for dev 0.5.1
        mask = seq_len_to_mask(seq_len, uni.size(1))
        out = self.src_embed(task, uni, bi1, bi2)
        out = self.pos(out)
        # out=self.norm(out)
        out = self.proj(self.encoder(out, mask.float()))

        if self.crf is not None:
            if tags is not None:
                out = self.crf(out, tags, mask)
                return {"loss": out}
            else:
                out, _ = self.crf.viterbi_decode(out, mask)
                return {"pred": out}
        else:
            if tags is not None:
                out = out.contiguous().view(-1, self.tag_size)
                tags = tags.data.masked_fill_(mask.eq(False), -100).view(-1)
                loss = self.loss_f(out, tags)
                return {"loss": loss}
            else:
                out = torch.argmax(out, dim=-1)
                return {"pred": out}


def make_CWS(
    N=6,
    d_model=256,
    d_ff=1024,
    h=4,
    dropout=0.2,
    tag_size=4,
    task_size=8,
    bigram_embedding=None,
    word_embedding=None,
    word_size=None,
    crf=None,
    freeze=True,
):
    c = copy.deepcopy
    # encoder=TransformerEncoder(num_layers=N,model_size=d_model,inner_size=d_ff,key_size=d_model//h,value_size=d_model//h,num_head=h,dropout=dropout)
    encoder = transformer.make_encoder(
        N=N, d_model=d_model, h=h, dropout=dropout, d_ff=d_ff
    )

    position = PositionalEncoding(d_model, dropout)

    embed = Embedding(
        task_size, d_model, word_embedding, bigram_embedding, word_size, freeze
    )
    model = CWSModel(encoder, embed, position, d_model, tag_size, crf=crf)

    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)

    return model
