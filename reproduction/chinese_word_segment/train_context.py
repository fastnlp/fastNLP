
from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import FullSpaceToHalfSpaceProcessor
from fastNLP.api.processor import SeqLenProcessor
from reproduction.chinese_word_segment.process.cws_processor import CWSCharSegProcessor
from reproduction.chinese_word_segment.process.cws_processor import CWSBMESTagProcessor
from reproduction.chinese_word_segment.process.cws_processor import Pre2Post2BigramProcessor
from reproduction.chinese_word_segment.process.cws_processor import VocabIndexerProcessor


from reproduction.chinese_word_segment.cws_io.cws_reader import ConllCWSReader
from reproduction.chinese_word_segment.models.cws_model import CWSBiLSTMCRF

from reproduction.chinese_word_segment.utils import calculate_pre_rec_f1

ds_name = 'msr'

tr_filename = '/home/hyan/ctb3/train.conllx'
dev_filename = '/home/hyan/ctb3/dev.conllx'


reader = ConllCWSReader()

tr_dataset = reader.load(tr_filename, cut_long_sent=True)
dev_dataset = reader.load(dev_filename)

print("Train {}. Dev: {}".format(len(tr_dataset), len(dev_dataset)))

# 1. 准备processor
fs2hs_proc = FullSpaceToHalfSpaceProcessor('raw_sentence')

char_proc = CWSCharSegProcessor('raw_sentence', 'chars_lst')
tag_proc = CWSBMESTagProcessor('raw_sentence', 'target')

bigram_proc = Pre2Post2BigramProcessor('chars_lst', 'bigrams_lst')

char_vocab_proc = VocabIndexerProcessor('chars_lst', new_added_filed_name='chars')
bigram_vocab_proc = VocabIndexerProcessor('bigrams_lst', new_added_filed_name='bigrams', min_freq=4)

seq_len_proc = SeqLenProcessor('chars')

# 2. 使用processor
fs2hs_proc(tr_dataset)

char_proc(tr_dataset)
tag_proc(tr_dataset)
bigram_proc(tr_dataset)

char_vocab_proc(tr_dataset)
bigram_vocab_proc(tr_dataset)
seq_len_proc(tr_dataset)

# 2.1 处理dev_dataset
fs2hs_proc(dev_dataset)

char_proc(dev_dataset)
tag_proc(dev_dataset)
bigram_proc(dev_dataset)

char_vocab_proc(dev_dataset)
bigram_vocab_proc(dev_dataset)
seq_len_proc(dev_dataset)

dev_dataset.set_input('chars', 'bigrams', 'target')
tr_dataset.set_input('chars', 'bigrams', 'target')
dev_dataset.set_target('seq_lens')
tr_dataset.set_target('seq_lens')

print("Finish preparing data.")


# 3. 得到数据集可以用于训练了
# TODO pretrain的embedding是怎么解决的？

import torch
from torch import optim


tag_size = tag_proc.tag_size

cws_model = CWSBiLSTMCRF(char_vocab_proc.get_vocab_size(), embed_dim=100,
                            bigram_vocab_num=bigram_vocab_proc.get_vocab_size(),
                            bigram_embed_dim=100, num_bigram_per_char=8,
                            hidden_size=200, bidirectional=True, embed_drop_p=0.2,
                            num_layers=1, tag_size=tag_size)
cws_model.cuda()

num_epochs = 5
optimizer = optim.Adagrad(cws_model.parameters(), lr=0.02)

from fastNLP.core.trainer import Trainer
from fastNLP.core.sampler import BucketSampler
from fastNLP.core.metrics import BMESF1PreRecMetric

metric = BMESF1PreRecMetric(target='tags')
trainer = Trainer(train_data=tr_dataset, model=cws_model, loss=None, metrics=metric, n_epochs=3,
                  batch_size=32, print_every=50, validate_every=-1, dev_data=dev_dataset, save_path=None,
                 optimizer=optimizer, check_code_level=0, metric_key='f', sampler=BucketSampler(), use_tqdm=True)

trainer.train()
exit(0)

#
# print_every = 50
# batch_size = 32
# tr_batcher = Batch(tr_dataset, batch_size, BucketSampler(batch_size=batch_size), use_cuda=False)
# dev_batcher = Batch(dev_dataset, batch_size, SequentialSampler(), use_cuda=False)
# num_batch_per_epoch = len(tr_dataset) // batch_size
# best_f1 = 0
# best_epoch = 0
# for num_epoch in range(num_epochs):
#     print('X' * 10 + ' Epoch: {}/{} '.format(num_epoch + 1, num_epochs) + 'X' * 10)
#     sys.stdout.flush()
#     avg_loss = 0
#     with tqdm(total=num_batch_per_epoch, leave=True) as pbar:
#         pbar.set_description_str('Epoch:%d' % (num_epoch + 1))
#         cws_model.train()
#         for batch_idx, (batch_x, batch_y) in enumerate(tr_batcher, 1):
#             optimizer.zero_grad()
#
#             tags = batch_y['tags'].long()
#             pred_dict = cws_model(**batch_x, tags=tags)  # B x L x tag_size
#
#             seq_lens = pred_dict['seq_lens']
#             masks = seq_lens_to_mask(seq_lens).float()
#             tags = tags.to(seq_lens.device)
#
#             loss = pred_dict['loss']
#
#             # loss = torch.sum(loss_fn(pred_dict['pred_probs'].view(-1, tag_size),
#             #                          tags.view(-1)) * masks.view(-1)) / torch.sum(masks)
#             # loss = torch.mean(F.cross_entropy(probs.view(-1, 2), tags.view(-1)) * masks.float())
#
#             avg_loss += loss.item()
#
#             loss.backward()
#             for group in optimizer.param_groups:
#                 for param in group['params']:
#                     param.grad.clamp_(-5, 5)
#
#             optimizer.step()
#
#             if batch_idx % print_every == 0:
#                 pbar.set_postfix_str('batch=%d, avg_loss=%.5f' % (batch_idx, avg_loss / print_every))
#                 avg_loss = 0
#                 pbar.update(print_every)
#     tr_batcher = Batch(tr_dataset, batch_size, BucketSampler(batch_size=batch_size), use_cuda=False)
#     # 验证集
#     pre, rec, f1 = calculate_pre_rec_f1(cws_model, dev_batcher, type='bmes')
#     print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1*100,
#                                                          pre*100,
#                                                          rec*100))
#     if best_f1<f1:
#         best_f1 = f1
#         # 缓存最佳的parameter，可能之后会用于保存
#         best_state_dict = {
#                 key:value.clone() for key, value in
#                     cws_model.state_dict().items()
#             }
#         best_epoch = num_epoch
#
# cws_model.load_state_dict(best_state_dict)

# 4. 组装需要存下的内容
pp = Pipeline()
pp.add_processor(fs2hs_proc)
# pp.add_processor(sp_proc)
pp.add_processor(char_proc)
pp.add_processor(tag_proc)
pp.add_processor(bigram_proc)
pp.add_processor(char_vocab_proc)
pp.add_processor(bigram_vocab_proc)
pp.add_processor(seq_len_proc)

# te_filename = '/hdd/fudanNLP/CWS/CWS_semiCRF/all_data/{}/middle_files/{}_test.txt'.format(ds_name, ds_name)
te_filename = '/home/hyan/ctb3/test.conllx'
te_dataset = reader.load(te_filename)
pp(te_dataset)

from fastNLP.core.tester import Tester

tester = Tester(data=te_dataset, model=cws_model, metrics=metric, batch_size=64, use_cuda=False,
                verbose=1)
#
# batch_size = 64
# te_batcher = Batch(te_dataset, batch_size, SequentialSampler(), use_cuda=False)
# pre, rec, f1 = calculate_pre_rec_f1(cws_model, te_batcher, type='bmes')
# print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1 * 100,
#                                                  pre * 100,
#                                                  rec * 100))

# TODO 这里貌似需要区分test pipeline与infer pipeline

test_context_dict = {'pipeline': pp,
                     'model': cws_model}
torch.save(test_context_dict, 'models/test_context_crf.pkl')


# 5. dev的pp
# 4. 组装需要存下的内容

from fastNLP.api.processor import ModelProcessor
from reproduction.chinese_word_segment.process.cws_processor import BMES2OutputProcessor

model_proc = ModelProcessor(cws_model)
output_proc = BMES2OutputProcessor()

pp = Pipeline()
pp.add_processor(fs2hs_proc)
# pp.add_processor(sp_proc)
pp.add_processor(char_proc)
pp.add_processor(bigram_proc)
pp.add_processor(char_vocab_proc)
pp.add_processor(bigram_vocab_proc)
pp.add_processor(seq_len_proc)

pp.add_processor(model_proc)
pp.add_processor(output_proc)


# TODO 这里貌似需要区分test pipeline与infer pipeline

infer_context_dict = {'pipeline': pp}
# torch.save(infer_context_dict, 'models/cws_crf.pkl')


# TODO 还需要考虑如何替换回原文的问题？
# 1. 不需要将特殊tag替换
# 2. 需要将特殊tag替换回去