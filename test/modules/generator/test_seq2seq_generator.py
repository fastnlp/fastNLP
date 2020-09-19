import unittest

import torch
from fastNLP.modules.generator import SequenceGenerator
from fastNLP.modules import TransformerSeq2SeqDecoder, LSTMSeq2SeqDecoder, Seq2SeqDecoder, State
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from torch import nn
from fastNLP import seq_len_to_mask


def prepare_env():
    vocab = Vocabulary().add_word_lst("This is a test .".split())
    vocab.add_word_lst("Another test !".split())
    embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5)

    encoder_output = torch.randn(2, 3, 10)
    src_seq_len = torch.LongTensor([3, 2])
    encoder_mask = seq_len_to_mask(src_seq_len)

    return embed, encoder_output, encoder_mask


class TestSequenceGenerator(unittest.TestCase):
    def test_run(self):
        # 测试能否运行 (1) 初始化decoder，(2) decode一发
        embed, encoder_output, encoder_mask = prepare_env()

        for do_sample in [True, False]:
            for num_beams in [1, 3, 5]:
                with self.subTest(do_sample=do_sample, num_beams=num_beams):
                    decoder = LSTMSeq2SeqDecoder(embed=embed, num_layers=1, hidden_size=10,
                             dropout=0.3, bind_decoder_input_output_embed=True, attention=True)
                    state = decoder.init_state(encoder_output, encoder_mask)
                    generator = SequenceGenerator(decoder=decoder, max_length=20, num_beams=num_beams,
                             do_sample=do_sample, temperature=1.0, top_k=50, top_p=1.0, bos_token_id=1, eos_token_id=None,
                             repetition_penalty=1, length_penalty=1.0, pad_token_id=0)
                    generator.generate(state=state, tokens=None)

                    decoder = TransformerSeq2SeqDecoder(embed=embed, pos_embed=nn.Embedding(10, embed.embedding_dim),
                             d_model=encoder_output.size(-1), num_layers=2, n_head=2, dim_ff=10, dropout=0.1,
                             bind_decoder_input_output_embed=True)
                    state = decoder.init_state(encoder_output, encoder_mask)
                    generator = SequenceGenerator(decoder=decoder, max_length=5, num_beams=num_beams,
                             do_sample=do_sample, temperature=1.0, top_k=50, top_p=1.0, bos_token_id=1, eos_token_id=None,
                             repetition_penalty=1, length_penalty=1.0, pad_token_id=0)
                    generator.generate(state=state, tokens=None)

                    # 测试一下其它值
                    decoder = TransformerSeq2SeqDecoder(embed=embed, pos_embed=nn.Embedding(10, embed.embedding_dim),
                                                        d_model=encoder_output.size(-1), num_layers=2, n_head=2, dim_ff=10,
                                                        dropout=0.1,
                                                        bind_decoder_input_output_embed=True)
                    state = decoder.init_state(encoder_output, encoder_mask)
                    generator = SequenceGenerator(decoder=decoder, max_length=5, num_beams=num_beams,
                                                  do_sample=do_sample, temperature=0.9, top_k=50, top_p=0.5, bos_token_id=1,
                                                  eos_token_id=3, repetition_penalty=2, length_penalty=1.5, pad_token_id=0)
                    generator.generate(state=state, tokens=None)

    def test_greedy_decode(self):
        # 测试能否正确的generate
        class GreedyDummyDecoder(Seq2SeqDecoder):
            def __init__(self, decoder_output):
                super().__init__()
                self.cur_length = 0
                self.decoder_output = decoder_output

            def decode(self, tokens, state):
                self.cur_length += 1
                scores = self.decoder_output[:, self.cur_length]
                return scores

        class DummyState(State):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def reorder_state(self, indices: torch.LongTensor):
                self.decoder.decoder_output = self._reorder_state(self.decoder.decoder_output, indices, dim=0)

        # greedy
        for beam_search in [1, 3]:
            decoder_output = torch.randn(2, 10, 5)
            path = decoder_output.argmax(dim=-1)  # 2 x 10
            decoder = GreedyDummyDecoder(decoder_output)
            with self.subTest(msg=beam_search, beam_search=beam_search):
                generator = SequenceGenerator(decoder=decoder, max_length=decoder_output.size(1), num_beams=beam_search,
                                                      do_sample=False, temperature=1, top_k=50, top_p=1, bos_token_id=1,
                                                      eos_token_id=None, repetition_penalty=1, length_penalty=1, pad_token_id=0)
                decode_path = generator.generate(DummyState(decoder), tokens=decoder_output[:, 0].argmax(dim=-1, keepdim=True))

                self.assertEqual(decode_path.eq(path).sum(), path.numel())

        # greedy check eos_token_id
        for beam_search in [1, 3]:
            decoder_output = torch.randn(2, 10, 5)
            decoder_output[:, :7, 4].fill_(-100)
            decoder_output[0, 7, 4] = 1000  # 在第8个结束
            decoder_output[1, 5, 4] = 1000
            path = decoder_output.argmax(dim=-1)  # 2 x 4
            decoder = GreedyDummyDecoder(decoder_output)
            with self.subTest(beam_search=beam_search):
                generator = SequenceGenerator(decoder=decoder, max_length=decoder_output.size(1), num_beams=beam_search,
                                              do_sample=False, temperature=1, top_k=50, top_p=0.5, bos_token_id=1,
                                              eos_token_id=4, repetition_penalty=1, length_penalty=1, pad_token_id=0)
                decode_path = generator.generate(DummyState(decoder),
                                                 tokens=decoder_output[:, 0].argmax(dim=-1, keepdim=True))
                self.assertEqual(decode_path.size(1), 8)  # 长度为8
                self.assertEqual(decode_path[0].eq(path[0, :8]).sum(), 8)
                self.assertEqual(decode_path[1, :6].eq(path[1, :6]).sum(), 6)
