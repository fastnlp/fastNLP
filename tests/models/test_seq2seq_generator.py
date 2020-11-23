
import unittest
from fastNLP.models import SequenceGeneratorModel
from fastNLP.models import LSTMSeq2SeqModel, TransformerSeq2SeqModel
from fastNLP import Vocabulary, DataSet
import torch
from fastNLP.embeddings import StaticEmbedding
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
from fastNLP import Callback


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

    ds.set_input('src_tokens', 'tgt_tokens', 'src_seq_len')
    ds.set_target('tgt_seq_len', 'tgt_tokens')

    return embed, ds


class ExitCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if eval_result['AccuracyMetric']['acc']==1:
            raise KeyboardInterrupt()


class TestSeq2SeqGeneratorModel(unittest.TestCase):
    def test_run(self):
        #  检测是否能够使用SequenceGeneratorModel训练, 透传预测
        embed, ds = prepare_env()
        model1 = TransformerSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                                                    pos_embed='sin', max_position=20, num_layers=2, d_model=30, n_head=6,
                                                    dim_ff=20, dropout=0.1,
                                                    bind_encoder_decoder_embed=True,
                                                    bind_decoder_input_output_embed=True)
        trainer = Trainer(ds, model1, optimizer=None, loss=CrossEntropyLoss(target='tgt_tokens', seq_len='tgt_seq_len'),
                 batch_size=32, sampler=None, drop_last=False, update_every=1,
                 num_workers=0, n_epochs=100, print_every=5,
                 dev_data=ds, metrics=AccuracyMetric(target='tgt_tokens', seq_len='tgt_seq_len'), metric_key=None,
                 validate_every=-1, save_path=None, use_tqdm=False, device=None,
                 callbacks=ExitCallback(), check_code_level=0)
        res = trainer.train()
        self.assertEqual(res['best_eval']['AccuracyMetric']['acc'], 1)

        embed, ds = prepare_env()
        model2 = LSTMSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
                                              num_layers=1, hidden_size=20, dropout=0.1,
                                              bind_encoder_decoder_embed=True,
                                              bind_decoder_input_output_embed=True, attention=True)
        optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
        trainer = Trainer(ds, model2, optimizer=optimizer, loss=CrossEntropyLoss(target='tgt_tokens', seq_len='tgt_seq_len'),
                          batch_size=32, sampler=None, drop_last=False, update_every=1,
                          num_workers=0, n_epochs=200, print_every=1,
                          dev_data=ds, metrics=AccuracyMetric(target='tgt_tokens', seq_len='tgt_seq_len'),
                          metric_key=None,
                          validate_every=-1, save_path=None, use_tqdm=False, device=None,
                          callbacks=ExitCallback(), check_code_level=0)
        res = trainer.train()
        self.assertEqual(res['best_eval']['AccuracyMetric']['acc'], 1)




