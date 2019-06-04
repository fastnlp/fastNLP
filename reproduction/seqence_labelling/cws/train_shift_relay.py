
import os

from fastNLP import cache_results
from reproduction.seqence_labelling.cws.data.CWSDataLoader import SigHanLoader
from reproduction.seqence_labelling.cws.model.model import ShiftRelayCWSModel
from fastNLP.io.embed_loader import EmbeddingOption
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP import Trainer
from torch.optim import Adam
from fastNLP import BucketSampler
from fastNLP import GradientClipCallback
from reproduction.seqence_labelling.cws.model.metric import RelayMetric


# 借助一下fastNLP的自动缓存机制，但是只能缓存4G以下的结果
@cache_results(None)
def prepare_data():
    data = SigHanLoader(target_type='shift_relay').process(file_dir, char_embed_opt=char_embed_opt,
                                                           bigram_vocab_opt=bigram_vocab_opt,
                                                           bigram_embed_opt=bigram_embed_opt,
                                                           L=L)
    return data

#########hyper
L = 4
hidden_size = 200
num_layers = 1
drop_p = 0.2
lr = 0.02

#########hyper
device = 0

# !!!!这里前往不要放完全路径，因为这样会暴露你们在服务器上的用户名，比较危险。所以一定要使用相对路径，最好把数据放到
#   你们的reproduction路径下，然后设置.gitignore
file_dir = '/path/to/pku'
char_embed_path = '/path/to/1grams_t3_m50_corpus.txt'
bigram_embed_path = 'path/to/2grams_t3_m50_corpus.txt'
bigram_vocab_opt = VocabularyOption(min_freq=3)
char_embed_opt = EmbeddingOption(embed_filepath=char_embed_path)
bigram_embed_opt = EmbeddingOption(embed_filepath=bigram_embed_path)

data_name = os.path.basename(file_dir)
cache_fp = 'caches/{}.pkl'.format(data_name)

data = prepare_data(_cache_fp=cache_fp, _refresh=False)

model = ShiftRelayCWSModel(char_embed=data.embeddings['chars'], bigram_embed=data.embeddings['bigrams'],
                           hidden_size=hidden_size, num_layers=num_layers,
                           L=L, num_bigram_per_char=1, drop_p=drop_p)

sampler = BucketSampler(batch_size=32)
optimizer = Adam(model.parameters(), lr=lr)
clipper = GradientClipCallback(clip_value=5, clip_type='value')
callbacks = [clipper]
# if pretrain:
#     fixer = FixEmbedding([model.char_embedding, model.bigram_embedding], fix_until=fix_until)
#     callbacks.append(fixer)
trainer = Trainer(data.datasets['train'], model, optimizer=optimizer, loss=None,
                  batch_size=32, sampler=sampler, update_every=5,
                  n_epochs=3, print_every=5,
                  dev_data=data.datasets['dev'], metrics=RelayMetric(), metric_key='f',
                  validate_every=-1, save_path=None,
                  prefetch=True, use_tqdm=True, device=device,
                  callbacks=callbacks,
                  check_code_level=0)
trainer.train()