import unittest

import torch

from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.modules import TransformerSeq2SeqDecoder
from fastNLP.modules import LSTMSeq2SeqDecoder
from fastNLP import seq_len_to_mask


class TestTransformerSeq2SeqDecoder(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        vocab.add_word_lst("Another test !".split())
        embed = StaticEmbedding(vocab, embedding_dim=10)

        encoder_output = torch.randn(2, 3, 10)
        src_seq_len = torch.LongTensor([3, 2])
        encoder_mask = seq_len_to_mask(src_seq_len)

        for flag in [True, False]:
            with self.subTest(bind_decoder_input_output_embed=flag):
                decoder = TransformerSeq2SeqDecoder(embed=embed, pos_embed = None,
                         d_model = 10, num_layers=2, n_head = 5, dim_ff = 20, dropout = 0.1,
                         bind_decoder_input_output_embed = True)
                state = decoder.init_state(encoder_output, encoder_mask)
                output = decoder(tokens=torch.randint(0, len(vocab), size=(2, 4)), state=state)
                self.assertEqual(output.size(), (2, 4, len(vocab)))


class TestLSTMDecoder(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        vocab.add_word_lst("Another test !".split())
        embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=10)

        encoder_output = torch.randn(2, 3, 10)
        tgt_words_idx = torch.LongTensor([[1, 2, 3, 4], [2, 3, 0, 0]])
        src_seq_len = torch.LongTensor([3, 2])
        encoder_mask = seq_len_to_mask(src_seq_len)

        for flag in [True, False]:
            for attention in [True, False]:
                with self.subTest(bind_decoder_input_output_embed=flag, attention=attention):
                    decoder = LSTMSeq2SeqDecoder(embed=embed, num_layers = 2, hidden_size = 10,
                             dropout = 0.3, bind_decoder_input_output_embed=flag, attention=attention)
                    state = decoder.init_state(encoder_output, encoder_mask)
                    output = decoder(tgt_words_idx, state)
                    self.assertEqual(tuple(output.size()), (2, 4, len(vocab)))
