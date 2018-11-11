
from fastNLP.api.pipeline import Pipeline
from fastNLP.api.processor import FullSpaceToHalfSpaceProcessor
from fastNLP.api.processor import IndexerProcessor
from reproduction.chinese_word_segment.process.cws_processor import SpeicalSpanProcessor
from reproduction.chinese_word_segment.process.cws_processor import CWSCharSegProcessor
from reproduction.chinese_word_segment.process.cws_processor import CWSSegAppTagProcessor
from reproduction.chinese_word_segment.process.cws_processor import Pre2Post2BigramProcessor
from reproduction.chinese_word_segment.process.cws_processor import VocabProcessor
from reproduction.chinese_word_segment.process.cws_processor import SeqLenProcessor

from reproduction.chinese_word_segment.process.span_converter import AlphaSpanConverter
from reproduction.chinese_word_segment.process.span_converter import DigitSpanConverter
from reproduction.chinese_word_segment.process.span_converter import TimeConverter
from reproduction.chinese_word_segment.process.span_converter import MixNumAlphaConverter
from reproduction.chinese_word_segment.process.span_converter import EmailConverter
from reproduction.chinese_word_segment.cws_io.cws_reader import NaiveCWSReader
from reproduction.chinese_word_segment.models.cws_model import CWSBiLSTMSegApp

from reproduction.chinese_word_segment.utils import calculate_pre_rec_f1

ds_name = 'msr'
tr_filename = '/hdd/fudanNLP/CWS/Multi_Criterion/all_data/{}/middle_files/{}_train.txt'.format(ds_name, ds_name)
dev_filename = '/hdd/fudanNLP/CWS/Multi_Criterion/all_data/{}/middle_files/{}_dev.txt'.format(ds_name, ds_name)

reader = NaiveCWSReader()

tr_dataset = reader.load(tr_filename, cut_long_sent=True)
dev_dataset = reader.load(dev_filename)


# 1. 准备processor
fs2hs_proc = FullSpaceToHalfSpaceProcessor('raw_sentence')

sp_proc = SpeicalSpanProcessor('raw_sentence', 'sentence')
# sp_proc.add_span_converter(EmailConverter())
# sp_proc.add_span_converter(MixNumAlphaConverter())
sp_proc.add_span_converter(AlphaSpanConverter())
sp_proc.add_span_converter(DigitSpanConverter())
# sp_proc.add_span_converter(TimeConverter())


char_proc = CWSCharSegProcessor('sentence', 'chars_list')

tag_proc = CWSSegAppTagProcessor('sentence', 'tags')

bigram_proc = Pre2Post2BigramProcessor('chars_list', 'bigrams_list')

char_vocab_proc = VocabProcessor('chars_list')
bigram_vocab_proc = VocabProcessor('bigrams_list', min_count=4)

# 2. 使用processor
fs2hs_proc(tr_dataset)

sp_proc(tr_dataset)

char_proc(tr_dataset)
tag_proc(tr_dataset)
bigram_proc(tr_dataset)

char_vocab_proc(tr_dataset)
bigram_vocab_proc(tr_dataset)

char_index_proc = IndexerProcessor(char_vocab_proc.get_vocab(), 'chars_list', 'chars',
                                   delete_old_field=False)
bigram_index_proc = IndexerProcessor(bigram_vocab_proc.get_vocab(), 'bigrams_list','bigrams',
                                      delete_old_field=True)
seq_len_proc = SeqLenProcessor('chars')

char_index_proc(tr_dataset)
bigram_index_proc(tr_dataset)
seq_len_proc(tr_dataset)

# 2.1 处理dev_dataset
fs2hs_proc(dev_dataset)
sp_proc(dev_dataset)

char_proc(dev_dataset)
tag_proc(dev_dataset)
bigram_proc(dev_dataset)

char_index_proc(dev_dataset)
bigram_index_proc(dev_dataset)
seq_len_proc(dev_dataset)

print("Finish preparing data.")
print("Vocab size:{}, bigram size:{}.".format(char_vocab_proc.get_vocab_size(), bigram_vocab_proc.get_vocab_size()))


# 3. 得到数据集可以用于训练了
# TODO pretrain的embedding是怎么解决的？

from reproduction.chinese_word_segment.utils import FocalLoss
from reproduction.chinese_word_segment.utils import seq_lens_to_mask
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import BucketSampler
from fastNLP.core.sampler import SequentialSampler

import torch
from torch import optim
import sys
from tqdm import tqdm


tag_size = tag_proc.tag_size

cws_model = CWSBiLSTMSegApp(char_vocab_proc.get_vocab_size(), embed_dim=100,
                            bigram_vocab_num=bigram_vocab_proc.get_vocab_size(),
                            bigram_embed_dim=100, num_bigram_per_char=8,
                            hidden_size=200, bidirectional=True, embed_drop_p=None,
                            num_layers=1, tag_size=tag_size)
cws_model.cuda()

num_epochs = 3
loss_fn = FocalLoss(class_num=tag_size)
optimizer = optim.Adagrad(cws_model.parameters(), lr=0.02)


print_every = 50
batch_size = 32
tr_batcher = Batch(tr_dataset, batch_size, BucketSampler(batch_size=batch_size), use_cuda=False)
dev_batcher = Batch(dev_dataset, batch_size, SequentialSampler(), use_cuda=False)
num_batch_per_epoch = len(tr_dataset) // batch_size
best_f1 = 0
best_epoch = 0
for num_epoch in range(num_epochs):
    print('X' * 10 + ' Epoch: {}/{} '.format(num_epoch + 1, num_epochs) + 'X' * 10)
    sys.stdout.flush()
    avg_loss = 0
    with tqdm(total=num_batch_per_epoch, leave=True) as pbar:
        pbar.set_description_str('Epoch:%d' % (num_epoch + 1))
        cws_model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(tr_batcher, 1):
            optimizer.zero_grad()

            pred_dict = cws_model(batch_x)  # B x L x tag_size

            seq_lens = pred_dict['seq_lens']
            masks = seq_lens_to_mask(seq_lens).float()
            tags = batch_y['tags'].long().to(seq_lens.device)

            loss = torch.sum(loss_fn(pred_dict['pred_probs'].view(-1, tag_size),
                                     tags.view(-1)) * masks.view(-1)) / torch.sum(masks)
            # loss = torch.mean(F.cross_entropy(probs.view(-1, 2), tags.view(-1)) * masks.float())

            avg_loss += loss.item()

            loss.backward()
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.clamp_(-5, 5)

            optimizer.step()

            if batch_idx % print_every == 0:
                pbar.set_postfix_str('batch=%d, avg_loss=%.5f' % (batch_idx, avg_loss / print_every))
                avg_loss = 0
                pbar.update(print_every)
    tr_batcher = Batch(tr_dataset, batch_size, BucketSampler(batch_size=batch_size), use_cuda=False)
    # 验证集
    pre, rec, f1 = calculate_pre_rec_f1(cws_model, dev_batcher)
    print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1*100,
                                                         pre*100,
                                                         rec*100))
    if best_f1<f1:
        best_f1 = f1
        # 缓存最佳的parameter，可能之后会用于保存
        best_state_dict = {
                key:value.clone() for key, value in
                    cws_model.state_dict().items()
            }
        best_epoch = num_epoch

cws_model.load_state_dict(best_state_dict)

# 4. 组装需要存下的内容
pp = Pipeline()
pp.add_processor(fs2hs_proc)
pp.add_processor(sp_proc)
pp.add_processor(char_proc)
pp.add_processor(tag_proc)
pp.add_processor(bigram_proc)
pp.add_processor(char_index_proc)
pp.add_processor(bigram_index_proc)
pp.add_processor(seq_len_proc)




te_filename = '/hdd/fudanNLP/CWS/Multi_Criterion/all_data/{}/middle_files/{}_test.txt'.format(ds_name, ds_name)
te_dataset = reader.load(te_filename)
pp(te_dataset)

batch_size = 64
te_batcher = Batch(te_dataset, batch_size, SequentialSampler(), use_cuda=False)
pre, rec, f1 = calculate_pre_rec_f1(cws_model, te_batcher)
print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1 * 100,
                                                 pre * 100,
                                                 rec * 100))

# TODO 这里貌似需要区分test pipeline与infer pipeline

test_context_dict = {'pipeline': pp,
                     'model': cws_model}
torch.save(test_context_dict, 'models/test_context.pkl')


# 5. dev的pp
# 4. 组装需要存下的内容

from fastNLP.api.processor import ModelProcessor

model_proc = ModelProcessor(cws_model)
index2word_proc =

pp = Pipeline()
pp.add_processor(fs2hs_proc)
pp.add_processor(sp_proc)
pp.add_processor(char_proc)
pp.add_processor(bigram_proc)
pp.add_processor(char_index_proc)
pp.add_processor(bigram_index_proc)
pp.add_processor(seq_len_proc)


pp.add_processor()



te_filename = '/hdd/fudanNLP/CWS/Multi_Criterion/all_data/{}/middle_files/{}_test.txt'.format(ds_name, ds_name)
te_dataset = reader.load(te_filename)
pp(te_dataset)

batch_size = 64
te_batcher = Batch(te_dataset, batch_size, SequentialSampler(), use_cuda=False)
pre, rec, f1 = calculate_pre_rec_f1(cws_model, te_batcher)
print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1 * 100,
                                                 pre * 100,
                                                 rec * 100))

# TODO 这里貌似需要区分test pipeline与infer pipeline

test_context_dict = {'pipeline': pp,
                     'model': cws_model}
torch.save(test_context_dict, 'models/test_context.pkl')


# TODO 还需要考虑如何替换回原文的问题？
# 1. 不需要将特殊tag替换
# 2. 需要将特殊tag替换回去