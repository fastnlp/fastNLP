
import pytest
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    from fastNLP.models.torch.seq2seq_model import TransformerSeq2SeqModel, LSTMSeq2SeqModel
    from fastNLP import Vocabulary
    from fastNLP.embeddings.torch import StaticEmbedding
    import torch
    from torch import optim
    import torch.nn.functional as F
from fastNLP import seq_len_to_mask


def prepare_env():
    vocab = Vocabulary().add_word_lst("This is a test .".split())
    vocab.add_word_lst("Another test !".split())
    embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5)

    src_words_idx = torch.LongTensor([[3, 1, 2], [1, 2, 0]])
    tgt_words_idx = torch.LongTensor([[1, 2, 3, 4], [2, 3, 0, 0]])
    src_seq_len = torch.LongTensor([3, 2])
    tgt_seq_len = torch.LongTensor([4, 2])

    return embed, src_words_idx, tgt_words_idx, src_seq_len, tgt_seq_len


def train_model(model, src_words_idx, tgt_words_idx, tgt_seq_len,  src_seq_len):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    mask = seq_len_to_mask(tgt_seq_len).eq(0)
    target = tgt_words_idx.masked_fill(mask, -100)

    for i in range(100):
        optimizer.zero_grad()
        pred = model(src_words_idx, tgt_words_idx, src_seq_len)['pred']  # bsz x max_len x vocab_size
        loss = F.cross_entropy(pred.transpose(1, 2), target)
        loss.backward()
        optimizer.step()

    right_count = pred.argmax(dim=-1).eq(target).masked_fill(mask, 1).sum()
    return right_count


@pytest.mark.torch
class TestTransformerSeq2SeqModel:
    def test_run(self):
        # 测试能否跑通
        embed, src_words_idx, tgt_words_idx, src_seq_len, tgt_seq_len = prepare_env()
        for pos_embed in ['learned', 'sin']:
            model = TransformerSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                        pos_embed=pos_embed, max_position=20, num_layers=2, d_model=30, n_head=6, dim_ff=20, dropout=0.1,
                        bind_encoder_decoder_embed=True,
                        bind_decoder_input_output_embed=True)

            output = model(src_words_idx, tgt_words_idx, src_seq_len)
            assert (output['pred'].size() == (2, 4, len(embed)))

        for bind_encoder_decoder_embed in [True, False]:
            tgt_embed = None
            for bind_decoder_input_output_embed in [True, False]:
                if bind_encoder_decoder_embed == False:
                    tgt_embed = embed

                model = TransformerSeq2SeqModel.build_model(src_embed=embed, tgt_embed=tgt_embed,
                                                            pos_embed='sin', max_position=20, num_layers=2,
                                                            d_model=30, n_head=6, dim_ff=20, dropout=0.1,
                                                            bind_encoder_decoder_embed=bind_encoder_decoder_embed,
                                                            bind_decoder_input_output_embed=bind_decoder_input_output_embed)

                output = model(src_words_idx, tgt_words_idx, src_seq_len)
                assert (output['pred'].size() == (2, 4, len(embed)))

    def test_train(self):
        # 测试能否train到overfit
        embed, src_words_idx, tgt_words_idx, src_seq_len, tgt_seq_len = prepare_env()

        model = TransformerSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                    pos_embed='sin', max_position=20, num_layers=2, d_model=30, n_head=6, dim_ff=20, dropout=0.1,
                    bind_encoder_decoder_embed=True,
                    bind_decoder_input_output_embed=True)

        right_count = train_model(model, src_words_idx, tgt_words_idx, tgt_seq_len,  src_seq_len)
        assert(right_count == tgt_words_idx.nelement())


@pytest.mark.torch
class TestLSTMSeq2SeqModel:
    def test_run(self):
        # 测试能否跑通
        embed, src_words_idx, tgt_words_idx, src_seq_len, tgt_seq_len = prepare_env()

        for bind_encoder_decoder_embed in [True, False]:
            tgt_embed = None
            for bind_decoder_input_output_embed in [True, False]:
                if bind_encoder_decoder_embed == False:
                    tgt_embed = embed
                model = LSTMSeq2SeqModel.build_model(src_embed=embed, tgt_embed=tgt_embed,
                                                     num_layers=2, hidden_size=20, dropout=0.1,
                                                     bind_encoder_decoder_embed=bind_encoder_decoder_embed,
                                                     bind_decoder_input_output_embed=bind_decoder_input_output_embed)
                output = model(src_words_idx, tgt_words_idx, src_seq_len)
                assert (output['pred'].size() == (2, 4, len(embed)))

    def test_train(self):
        embed, src_words_idx, tgt_words_idx, src_seq_len, tgt_seq_len = prepare_env()

        model = LSTMSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                    num_layers=1, hidden_size=20, dropout=0.1,
                    bind_encoder_decoder_embed=True,
                    bind_decoder_input_output_embed=True)

        right_count = train_model(model, src_words_idx, tgt_words_idx, tgt_seq_len,  src_seq_len)
        assert (right_count == tgt_words_idx.nelement())

