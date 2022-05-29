import pytest

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    from fastNLP.models.torch import LSTMSeq2SeqModel, TransformerSeq2SeqModel
    import torch
    from fastNLP.embeddings.torch import StaticEmbedding

from fastNLP import Vocabulary, DataSet
from fastNLP import Trainer, Accuracy
from fastNLP import Callback, TorchDataLoader


def prepare_env():
    vocab = Vocabulary().add_word_lst("This is a test .".split())
    vocab.add_word_lst("Another test !".split())
    embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5)

    src_words_idx = [[3, 1, 2], [1, 2]]
    # tgt_words_idx = [[1, 2, 3, 4], [2, 3]]
    src_seq_len = [3, 2]
    # tgt_seq_len = [4, 2]

    ds = DataSet({'src_tokens': src_words_idx, 'src_seq_len': src_seq_len, 'tgt_tokens': src_words_idx,
                  'tgt_seq_len':src_seq_len})

    dl = TorchDataLoader(ds, batch_size=32)
    return embed, dl


class ExitCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_valid_end(self, trainer, results):
        if results['acc#acc'] == 1:
            raise KeyboardInterrupt()


@pytest.mark.torch
class TestSeq2SeqGeneratorModel:
    def test_run(self):
        #  检测是否能够使用SequenceGeneratorModel训练, 透传预测
        embed, dl = prepare_env()
        model1 = TransformerSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                                                    pos_embed='sin', max_position=20, num_layers=2, d_model=30, n_head=6,
                                                    dim_ff=20, dropout=0.1,
                                                    bind_encoder_decoder_embed=True,
                                                    bind_decoder_input_output_embed=True)
        optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)
        trainer = Trainer(model1, driver='torch', optimizers=optimizer, train_dataloader=dl,
                          n_epochs=100, evaluate_dataloaders=dl, metrics={'acc': Accuracy()},
                          evaluate_input_mapping=lambda x: {'target': x['tgt_tokens'],
                                                            'seq_len': x['tgt_seq_len'],
                                                            **x},
                          callbacks=ExitCallback())

        trainer.run()

        embed, dl = prepare_env()
        model2 = LSTMSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                                              num_layers=1, hidden_size=20, dropout=0.1,
                                              bind_encoder_decoder_embed=True,
                                              bind_decoder_input_output_embed=True, attention=True)
        optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
        trainer = Trainer(model2, driver='torch', optimizers=optimizer, train_dataloader=dl,
                          n_epochs=100, evaluate_dataloaders=dl, metrics={'acc': Accuracy()},
                          evaluate_input_mapping=lambda x: {'target': x['tgt_tokens'],
                                                            'seq_len': x['tgt_seq_len'],
                                                            **x},
                          callbacks=ExitCallback())
        trainer.run()
