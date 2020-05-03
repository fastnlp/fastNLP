import torch
from torch import nn
import numpy as np
from ..embeddings import StaticEmbedding
from ..modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoder, Seq2SeqEncoder, LSTMSeq2SeqEncoder
from ..modules.decoder.seq2seq_decoder import TransformerSeq2SeqDecoder, LSTMSeq2SeqDecoder, Seq2SeqDecoder
from ..core import Vocabulary
import argparse


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def build_embedding(vocab, embed_dim, model_dir_or_name=None):
    """
    todo: 根据需求可丰富该函数的功能，目前只返回StaticEmbedding
    :param vocab: Vocabulary
    :param embed_dim:
    :param model_dir_or_name:
    :return:
    """
    assert isinstance(vocab, Vocabulary)
    embed = StaticEmbedding(vocab=vocab, embedding_dim=embed_dim, model_dir_or_name=model_dir_or_name)

    return embed


class BaseSeq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(BaseSeq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, Seq2SeqEncoder)
        assert isinstance(self.decoder, Seq2SeqDecoder)

    def forward(self, src_words, src_seq_len, tgt_prev_words):
        encoder_output, encoder_mask = self.encoder(src_words, src_seq_len)
        decoder_output = self.decoder(tgt_prev_words, encoder_output, encoder_mask)

        return {'tgt_output': decoder_output}


class LSTMSeq2SeqModel(BaseSeq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument('--embedding_dim', type=int, default=300)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--hidden_size', type=int, default=300)
        parser.add_argument('--bidirectional', action='store_true', default=True)
        args = parser.parse_args()

        return args

    @classmethod
    def build_model(cls, args, src_vocab, tgt_vocab):
        # 处理embedding
        src_embed = build_embedding(src_vocab, args.embedding_dim)
        if args.share_embedding:
            assert src_vocab == tgt_vocab, "share_embedding requires a joined vocab"
            tgt_embed = src_embed
        else:
            tgt_embed = build_embedding(tgt_vocab, args.embedding_dim)

        if args.bind_input_output_embed:
            output_embed = nn.Parameter(tgt_embed.embedding.weight)
        else:
            output_embed = nn.Parameter(torch.Tensor(len(tgt_vocab), args.embedding_dim), requires_grad=True)
            nn.init.normal_(output_embed, mean=0, std=args.embedding_dim ** -0.5)

        encoder = LSTMSeq2SeqEncoder(vocab=src_vocab, embed=src_embed, num_layers=args.num_layers,
                                     hidden_size=args.hidden_size, dropout=args.dropout,
                                     bidirectional=args.bidirectional)
        decoder = LSTMSeq2SeqDecoder(vocab=tgt_vocab, embed=tgt_embed, num_layers=args.num_layers,
                                     hidden_size=args.hidden_size, dropout=args.dropout, output_embed=output_embed,
                                     attention=True)

        return LSTMSeq2SeqModel(encoder, decoder)


class TransformerSeq2SeqModel(BaseSeq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=6)
        parser.add_argument('--n_head', type=int, default=8)
        parser.add_argument('--dim_ff', type=int, default=2048)
        parser.add_argument('--bind_input_output_embed', action='store_true', default=True)
        parser.add_argument('--share_embedding', action='store_true', default=True)

        args = parser.parse_args()

        return args

    @classmethod
    def build_model(cls, args, src_vocab, tgt_vocab):
        d_model = args.d_model
        args.max_positions = getattr(args, 'max_positions', 1024)  # 处理的最长长度

        # 处理embedding
        src_embed = build_embedding(src_vocab, d_model)
        if args.share_embedding:
            assert src_vocab == tgt_vocab, "share_embedding requires a joined vocab"
            tgt_embed = src_embed
        else:
            tgt_embed = build_embedding(tgt_vocab, d_model)

        if args.bind_input_output_embed:
            output_embed = nn.Parameter(tgt_embed.embedding.weight)
        else:
            output_embed = nn.Parameter(torch.Tensor(len(tgt_vocab), d_model), requires_grad=True)
            nn.init.normal_(output_embed, mean=0, std=d_model ** -0.5)

        pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(args.max_positions + 1, d_model, padding_idx=0),
            freeze=True)  # 这里规定0是padding

        encoder = TransformerSeq2SeqEncoder(vocab=src_vocab, embed=src_embed, pos_embed=pos_embed,
                                            num_layers=args.num_layers, d_model=args.d_model,
                                            n_head=args.n_head, dim_ff=args.dim_ff, dropout=args.dropout)
        decoder = TransformerSeq2SeqDecoder(vocab=tgt_vocab, embed=tgt_embed, pos_embed=pos_embed,
                                            num_layers=args.num_layers, d_model=args.d_model,
                                            n_head=args.n_head, dim_ff=args.dim_ff, dropout=args.dropout,
                                            output_embed=output_embed)

        return TransformerSeq2SeqModel(encoder, decoder)
