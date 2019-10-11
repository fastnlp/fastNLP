import sys
sys.path.append('../../..')

from fastNLP.io.pipe.cws import CWSPipe
from reproduction.sequence_labelling.cws.model.bilstm_crf_cws import BiLSTMCRF
from fastNLP import Trainer, cache_results
from fastNLP.embeddings import StaticEmbedding
from fastNLP import EvaluateCallback, BucketSampler, SpanFPreRecMetric, GradientClipCallback
from torch.optim import Adagrad

###########hyper
dataname = 'pku'
hidden_size = 400
num_layers = 1
lr = 0.05
###########hyper


@cache_results('{}.pkl'.format(dataname), _refresh=False)
def get_data():
    data_bundle = CWSPipe(dataset_name=dataname, bigrams=True, trigrams=False).process_from_file()
    char_embed = StaticEmbedding(data_bundle.get_vocab('chars'), dropout=0.33, word_dropout=0.01,
                                 model_dir_or_name='~/exps/CWS/pretrain/vectors/1grams_t3_m50_corpus.txt')
    bigram_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'), dropout=0.33,min_freq=3, word_dropout=0.01,
                                 model_dir_or_name='~/exps/CWS/pretrain/vectors/2grams_t3_m50_corpus.txt')
    return data_bundle, char_embed, bigram_embed

data_bundle, char_embed, bigram_embed = get_data()
print(data_bundle)

model = BiLSTMCRF(char_embed, hidden_size, num_layers, target_vocab=data_bundle.get_vocab('target'), bigram_embed=bigram_embed,
                  trigram_embed=None, dropout=0.3)
model.cuda()

callbacks = []
callbacks.append(EvaluateCallback(data_bundle.get_dataset('test')))
callbacks.append(GradientClipCallback(clip_type='value', clip_value=5))
optimizer = Adagrad(model.parameters(), lr=lr)

metrics = []
metric1 = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type='bmes')
metrics.append(metric1)

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer=optimizer, loss=None,
                 batch_size=128, sampler=BucketSampler(), update_every=1,
                 num_workers=1, n_epochs=10, print_every=5,
                 dev_data=data_bundle.get_dataset('dev'),
                 metrics=metrics,
                 metric_key=None,
                 validate_every=-1, save_path=None, use_tqdm=True, device=0,
                 callbacks=callbacks, check_code_level=0, dev_batch_size=128)
trainer.train()
