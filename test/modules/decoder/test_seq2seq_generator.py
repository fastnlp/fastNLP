import unittest

import torch

from fastNLP.modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoder, BiLSTMEncoder
from fastNLP.modules.decoder.seq2seq_decoder import TransformerSeq2SeqDecoder, TransformerPast, LSTMPast, LSTMDecoder
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.modules.decoder.seq2seq_generator import SequenceGenerator


class TestSequenceGenerator(unittest.TestCase):
    def test_case_for_transformer(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        vocab.add_word_lst("Another test !".split())
        embed = StaticEmbedding(vocab, embedding_dim=512)
        encoder = TransformerSeq2SeqEncoder(embed, num_layers=6)
        decoder = TransformerSeq2SeqDecoder(embed=embed, bind_input_output_embed=True, num_layers=6)

        src_words_idx = torch.LongTensor([[3, 1, 2], [1, 2, 0]])
        tgt_words_idx = torch.LongTensor([[1, 2, 3, 4], [2, 3, 0, 0]])
        src_seq_len = torch.LongTensor([3, 2])

        encoder_outputs, mask = encoder(src_words_idx, src_seq_len)
        past = TransformerPast(encoder_outputs=encoder_outputs, encoder_mask=mask, num_decoder_layer=6)

        generator = SequenceGenerator(decoder, bos_token_id=1, eos_token_id=2, num_beams=2)
        tokens_ids = generator.generate(past=past)

        print(tokens_ids)

    def test_case_for_lstm(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        vocab.add_word_lst("Another test !".split())
        embed = StaticEmbedding(vocab, embedding_dim=512)
        encoder = BiLSTMEncoder(embed)
        decoder = LSTMDecoder(embed, bind_input_output_embed=True)
        src_words_idx = torch.LongTensor([[3, 1, 2], [1, 2, 0]])
        tgt_words_idx = torch.LongTensor([[1, 2, 3, 4], [2, 3, 0, 0]])
        src_seq_len = torch.LongTensor([3, 2])

        words, hx = encoder(src_words_idx, src_seq_len)
        encode_mask = seq_len_to_mask(src_seq_len)
        hidden = torch.cat([hx[0][-2:-1], hx[0][-1:]], dim=-1).repeat(decoder.num_layers, 1, 1)
        cell = torch.cat([hx[1][-2:-1], hx[1][-1:]], dim=-1).repeat(decoder.num_layers, 1, 1)
        past = LSTMPast(encode_outputs=words, encode_mask=encode_mask, hx=(hidden, cell))

        generator = SequenceGenerator(decoder, bos_token_id=1, eos_token_id=2, num_beams=2)
        tokens_ids = generator.generate(past=past)

        print(tokens_ids)
