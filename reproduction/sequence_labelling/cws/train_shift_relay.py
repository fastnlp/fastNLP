
import sys
sys.path.append('../../..')

from fastNLP import cache_results
from reproduction.sequence_labelling.cws.data.cws_shift_pipe import CWSShiftRelayPipe
from reproduction.sequence_labelling.cws.model.bilstm_shift_relay import ShiftRelayCWSModel
from fastNLP import Trainer
from torch.optim import Adam
from fastNLP import BucketSampler
from fastNLP import GradientClipCallback
from reproduction.sequence_labelling.cws.model.metric import RelayMetric
from fastNLP.embeddings import StaticEmbedding
from fastNLP import EvaluateCallback

#########hyper
L = 4
hidden_size = 200
num_layers = 1
drop_p = 0.2
lr = 0.008
data_name = 'pku'
#########hyper
device = 0

cache_fp = 'caches/{}.pkl'.format(data_name)
@cache_results(_cache_fp=cache_fp, _refresh=True)   # 将结果缓存到cache_fp中，这样下次运行就直接读取，而不需要再次运行
def prepare_data():
    data_bundle = CWSShiftRelayPipe(dataset_name=data_name, L=L).process_from_file()
    # 预训练的character embedding和bigram embedding
    char_embed = StaticEmbedding(data_bundle.get_vocab('chars'), dropout=0.5, word_dropout=0.01,
                                 model_dir_or_name='~/exps/CWS/pretrain/vectors/1grams_t3_m50_corpus.txt')
    bigram_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'), dropout=0.5, min_freq=3, word_dropout=0.01,
                                 model_dir_or_name='~/exps/CWS/pretrain/vectors/2grams_t3_m50_corpus.txt')

    return data_bundle, char_embed, bigram_embed

data, char_embed, bigram_embed = prepare_data()

model = ShiftRelayCWSModel(char_embed=char_embed, bigram_embed=bigram_embed,
                           hidden_size=hidden_size, num_layers=num_layers, drop_p=drop_p, L=L)

sampler = BucketSampler()
optimizer = Adam(model.parameters(), lr=lr)
clipper = GradientClipCallback(clip_value=5, clip_type='value')  # 截断太大的梯度
evaluator = EvaluateCallback(data.get_dataset('test'))  # 额外测试在test集上的效果
callbacks = [clipper, evaluator]

trainer = Trainer(data.get_dataset('train'), model, optimizer=optimizer, loss=None, batch_size=128, sampler=sampler,
                  update_every=1, n_epochs=10, print_every=5, dev_data=data.get_dataset('dev'), metrics=RelayMetric(),
                  metric_key='f', validate_every=-1, save_path=None, use_tqdm=True, device=device, callbacks=callbacks,
                  check_code_level=0, num_workers=1)
trainer.train()