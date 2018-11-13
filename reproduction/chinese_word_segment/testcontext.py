

import torch
from reproduction.chinese_word_segment.cws_io.cws_reader import NaiveCWSReader
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.batch import Batch
from reproduction.chinese_word_segment.utils import calculate_pre_rec_f1

def f1():
    ds_name = 'pku'

    test_dict = torch.load('models/test_context.pkl')


    pp = test_dict['pipeline']
    model = test_dict['model'].cuda()

    reader = NaiveCWSReader()
    te_filename = '/hdd/fudanNLP/CWS/Multi_Criterion/all_data/{}/{}_raw_data/{}_raw_test.txt'.format(ds_name, ds_name,
                                                                                                     ds_name)
    te_dataset = reader.load(te_filename)
    pp(te_dataset)

    batch_size = 64
    te_batcher = Batch(te_dataset, batch_size, SequentialSampler(), use_cuda=False)
    pre, rec, f1 = calculate_pre_rec_f1(model, te_batcher)
    print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1 * 100,
                                                     pre * 100,
                                                     rec * 100))


def f2():
    from fastNLP.api.api import CWS
    cws = CWS('models/maml-cws.pkl')
    datasets = ['msr', 'as', 'pku', 'ctb', 'ncc', 'cityu', 'ckip', 'sxu']
    for dataset in datasets:
        print(dataset)
        with open('/hdd/fudanNLP/CWS/others/benchmark/raw_and_gold/{}_raw.txt'.format(dataset), 'r') as f:
            lines = f.readlines()
            results = cws.predict(lines)

        with open('/hdd/fudanNLP/CWS/others/benchmark/fastNLP_output/{}_seg.txt'.format(dataset), 'w', encoding='utf-8') as f:
            for line in results:
                f.write(line)


f1()