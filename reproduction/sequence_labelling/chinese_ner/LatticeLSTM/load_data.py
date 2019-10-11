from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
import numpy as np
import fitlog
import pickle
import os
from fastNLP.embeddings import StaticEmbedding
from fastNLP import cache_results


@cache_results(_cache_fp='mtl16', _refresh=False)
def load_16_task(dict_path):
    '''

    :param dict_path: /remote-home/txsun/fnlp/MTL-LT/data
    :return:
    '''
    task_path = os.path.join(dict_path,'data.pkl')
    embedding_path = os.path.join(dict_path,'word_embedding.npy')

    embedding = np.load(embedding_path).astype(np.float32)

    task_list = pickle.load(open(task_path, 'rb'))['task_lst']

    for t in task_list:
        t.train_set.rename_field('words_idx', 'words')
        t.dev_set.rename_field('words_idx', 'words')
        t.test_set.rename_field('words_idx', 'words')

        t.train_set.rename_field('label', 'target')
        t.dev_set.rename_field('label', 'target')
        t.test_set.rename_field('label', 'target')

        t.train_set.add_seq_len('words')
        t.dev_set.add_seq_len('words')
        t.test_set.add_seq_len('words')

        t.train_set.set_input(Const.INPUT, Const.INPUT_LEN)
        t.dev_set.set_input(Const.INPUT, Const.INPUT_LEN)
        t.test_set.set_input(Const.INPUT, Const.INPUT_LEN)

    return task_list,embedding


@cache_results(_cache_fp='SST2', _refresh=False)
def load_sst2(dict_path,embedding_path=None):
    '''

    :param dict_path: /remote-home/xnli/data/corpus/text_classification/SST-2/
    :param embedding_path: glove 300d txt
    :return:
    '''
    train_path = os.path.join(dict_path,'train.tsv')
    dev_path = os.path.join(dict_path,'dev.tsv')

    loader = CSVLoader(headers=('words', 'target'), sep='\t')
    train_data = loader.load(train_path).datasets['train']
    dev_data = loader.load(dev_path).datasets['train']

    train_data.apply_field(lambda x: x.split(), field_name='words', new_field_name='words')
    dev_data.apply_field(lambda x: x.split(), field_name='words', new_field_name='words')

    train_data.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')
    dev_data.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')

    vocab = Vocabulary(min_freq=2)
    vocab.from_dataset(train_data, field_name='words')
    vocab.from_dataset(dev_data, field_name='words')

    # pretrained_embedding = load_word_emb(embedding_path, 300, vocab)

    label_vocab = Vocabulary(padding=None, unknown=None).from_dataset(train_data, field_name='target')

    label_vocab.index_dataset(train_data, field_name='target')
    label_vocab.index_dataset(dev_data, field_name='target')

    vocab.index_dataset(train_data, field_name='words', new_field_name='words')
    vocab.index_dataset(dev_data, field_name='words', new_field_name='words')

    train_data.set_input(Const.INPUT, Const.INPUT_LEN)
    train_data.set_target(Const.TARGET)

    dev_data.set_input(Const.INPUT, Const.INPUT_LEN)
    dev_data.set_target(Const.TARGET)

    if embedding_path is not None:
        pretrained_embedding = load_word_emb(embedding_path, 300, vocab)
        return (train_data,dev_data),(vocab,label_vocab),pretrained_embedding

    else:
        return (train_data,dev_data),(vocab,label_vocab)

@cache_results(_cache_fp='OntonotesPOS', _refresh=False)
def load_conllized_ontonote_POS(path,embedding_path=None):
    from fastNLP.io.data_loader import ConllLoader
    header2index = {'words':3,'POS':4,'NER':10}
    headers = ['words','POS']

    if 'NER' in headers:
        print('警告！通过 load_conllized_ontonote 函数读出来的NER标签不是BIOS，是纯粹的conll格式，是错误的！')
    indexes = list(map(lambda x:header2index[x],headers))

    loader = ConllLoader(headers,indexes)

    bundle = loader.load(path)

    # print(bundle.datasets)

    train_set = bundle.datasets['train']
    dev_set = bundle.datasets['dev']
    test_set = bundle.datasets['test']




    # train_set = loader.load(os.path.join(path,'train.txt'))
    # dev_set = loader.load(os.path.join(path, 'dev.txt'))
    # test_set = loader.load(os.path.join(path, 'test.txt'))

    # print(len(train_set))

    train_set.add_seq_len('words','seq_len')
    dev_set.add_seq_len('words','seq_len')
    test_set.add_seq_len('words','seq_len')



    # print(dataset['POS'])

    vocab = Vocabulary(min_freq=1)
    vocab.from_dataset(train_set,field_name='words')
    vocab.from_dataset(dev_set, field_name='words')
    vocab.from_dataset(test_set, field_name='words')

    vocab.index_dataset(train_set,field_name='words')
    vocab.index_dataset(dev_set, field_name='words')
    vocab.index_dataset(test_set, field_name='words')




    label_vocab_dict = {}

    for i,h in enumerate(headers):
        if h == 'words':
            continue
        label_vocab_dict[h] = Vocabulary(min_freq=1,padding=None,unknown=None)
        label_vocab_dict[h].from_dataset(train_set,field_name=h)

        label_vocab_dict[h].index_dataset(train_set,field_name=h)
        label_vocab_dict[h].index_dataset(dev_set,field_name=h)
        label_vocab_dict[h].index_dataset(test_set,field_name=h)

    train_set.set_input(Const.INPUT, Const.INPUT_LEN)
    train_set.set_target(headers[1])

    dev_set.set_input(Const.INPUT, Const.INPUT_LEN)
    dev_set.set_target(headers[1])

    test_set.set_input(Const.INPUT, Const.INPUT_LEN)
    test_set.set_target(headers[1])

    if len(headers) > 2:
        print('警告：由于任务数量大于1，所以需要每次手动设置target！')


    print('train:',len(train_set),'dev:',len(dev_set),'test:',len(test_set))

    if embedding_path is not None:
        pretrained_embedding = load_word_emb(embedding_path, 300, vocab)
        return (train_set,dev_set,test_set),(vocab,label_vocab_dict),pretrained_embedding
    else:
        return (train_set, dev_set, test_set), (vocab, label_vocab_dict)


@cache_results(_cache_fp='OntonotesNER', _refresh=False)
def load_conllized_ontonote_NER(path,embedding_path=None):
    from fastNLP.io.pipe.conll import OntoNotesNERPipe
    ontoNotesNERPipe = OntoNotesNERPipe(lower=True,target_pad_val=-100)
    bundle_NER = ontoNotesNERPipe.process_from_file(path)

    train_set_NER = bundle_NER.datasets['train']
    dev_set_NER = bundle_NER.datasets['dev']
    test_set_NER = bundle_NER.datasets['test']

    train_set_NER.add_seq_len('words','seq_len')
    dev_set_NER.add_seq_len('words','seq_len')
    test_set_NER.add_seq_len('words','seq_len')


    NER_vocab = bundle_NER.get_vocab('target')
    word_vocab = bundle_NER.get_vocab('words')

    if embedding_path is not None:

        embed = StaticEmbedding(vocab=word_vocab, model_dir_or_name=embedding_path, word_dropout=0.01,
                                dropout=0.5,lower=True)


        # pretrained_embedding = load_word_emb(embedding_path, 300, word_vocab)
        return (train_set_NER,dev_set_NER,test_set_NER),\
               (word_vocab,NER_vocab),embed
    else:
        return (train_set_NER, dev_set_NER, test_set_NER), (NER_vocab, word_vocab)

@cache_results(_cache_fp='OntonotesPOSNER', _refresh=False)

def load_conllized_ontonote_NER_POS(path,embedding_path=None):
    from fastNLP.io.pipe.conll import OntoNotesNERPipe
    ontoNotesNERPipe = OntoNotesNERPipe(lower=True)
    bundle_NER = ontoNotesNERPipe.process_from_file(path)

    train_set_NER = bundle_NER.datasets['train']
    dev_set_NER = bundle_NER.datasets['dev']
    test_set_NER = bundle_NER.datasets['test']

    NER_vocab = bundle_NER.get_vocab('target')
    word_vocab = bundle_NER.get_vocab('words')

    (train_set_POS,dev_set_POS,test_set_POS),(_,POS_vocab) = load_conllized_ontonote_POS(path)
    POS_vocab = POS_vocab['POS']

    train_set_NER.add_field('pos',train_set_POS['POS'],is_target=True)
    dev_set_NER.add_field('pos', dev_set_POS['POS'], is_target=True)
    test_set_NER.add_field('pos', test_set_POS['POS'], is_target=True)

    if train_set_NER.has_field('target'):
        train_set_NER.rename_field('target','ner')

    if dev_set_NER.has_field('target'):
        dev_set_NER.rename_field('target','ner')

    if test_set_NER.has_field('target'):
        test_set_NER.rename_field('target','ner')



    if train_set_NER.has_field('pos'):
        train_set_NER.rename_field('pos','posid')
    if dev_set_NER.has_field('pos'):
        dev_set_NER.rename_field('pos','posid')
    if test_set_NER.has_field('pos'):
        test_set_NER.rename_field('pos','posid')

    if train_set_NER.has_field('ner'):
        train_set_NER.rename_field('ner','nerid')
    if dev_set_NER.has_field('ner'):
        dev_set_NER.rename_field('ner','nerid')
    if test_set_NER.has_field('ner'):
        test_set_NER.rename_field('ner','nerid')

    if embedding_path is not None:

        embed = StaticEmbedding(vocab=word_vocab, model_dir_or_name=embedding_path, word_dropout=0.01,
                                dropout=0.5,lower=True)

        return (train_set_NER,dev_set_NER,test_set_NER),\
               (word_vocab,POS_vocab,NER_vocab),embed
    else:
        return (train_set_NER, dev_set_NER, test_set_NER), (NER_vocab, word_vocab)

@cache_results(_cache_fp='Ontonotes3', _refresh=True)
def load_conllized_ontonote_pkl(path,embedding_path=None):

    data_bundle = pickle.load(open(path,'rb'))
    train_set = data_bundle.datasets['train']
    dev_set = data_bundle.datasets['dev']
    test_set = data_bundle.datasets['test']

    train_set.rename_field('pos','posid')
    train_set.rename_field('ner','nerid')
    train_set.rename_field('chunk','chunkid')

    dev_set.rename_field('pos','posid')
    dev_set.rename_field('ner','nerid')
    dev_set.rename_field('chunk','chunkid')

    test_set.rename_field('pos','posid')
    test_set.rename_field('ner','nerid')
    test_set.rename_field('chunk','chunkid')


    word_vocab = data_bundle.vocabs['words']
    pos_vocab = data_bundle.vocabs['pos']
    ner_vocab = data_bundle.vocabs['ner']
    chunk_vocab = data_bundle.vocabs['chunk']


    if embedding_path is not None:

        embed = StaticEmbedding(vocab=word_vocab, model_dir_or_name=embedding_path, word_dropout=0.01,
                                dropout=0.5,lower=True)

        return (train_set,dev_set,test_set),\
               (word_vocab,pos_vocab,ner_vocab,chunk_vocab),embed
    else:
        return (train_set, dev_set, test_set), (word_vocab,ner_vocab)
    # print(data_bundle)










# @cache_results(_cache_fp='Conll2003', _refresh=False)
# def load_conll_2003(path,embedding_path=None):
#     f = open(path, 'rb')
#     data_pkl = pickle.load(f)
#
#     task_lst = data_pkl['task_lst']
#     vocabs = data_pkl['vocabs']
#     # word_vocab = vocabs['words']
#     # pos_vocab = vocabs['pos']
#     # chunk_vocab = vocabs['chunk']
#     # ner_vocab = vocabs['ner']
#
#     if embedding_path is not None:
#         embed = StaticEmbedding(vocab=vocabs['words'], model_dir_or_name=embedding_path, word_dropout=0.01,
#                                 dropout=0.5)
#         return task_lst,vocabs,embed
#     else:
#         return task_lst,vocabs

# @cache_results(_cache_fp='Conll2003_mine', _refresh=False)
@cache_results(_cache_fp='Conll2003_mine_embed_100', _refresh=True)
def load_conll_2003_mine(path,embedding_path=None,pad_val=-100):
    f = open(path, 'rb')

    data_pkl = pickle.load(f)
    # print(data_pkl)
    # print(data_pkl)
    train_set = data_pkl[0]['train']
    dev_set = data_pkl[0]['dev']
    test_set = data_pkl[0]['test']

    train_set.set_pad_val('posid',pad_val)
    train_set.set_pad_val('nerid', pad_val)
    train_set.set_pad_val('chunkid', pad_val)

    dev_set.set_pad_val('posid',pad_val)
    dev_set.set_pad_val('nerid', pad_val)
    dev_set.set_pad_val('chunkid', pad_val)

    test_set.set_pad_val('posid',pad_val)
    test_set.set_pad_val('nerid', pad_val)
    test_set.set_pad_val('chunkid', pad_val)

    if train_set.has_field('task_id'):

        train_set.delete_field('task_id')

    if dev_set.has_field('task_id'):
        dev_set.delete_field('task_id')

    if test_set.has_field('task_id'):
        test_set.delete_field('task_id')

    if train_set.has_field('words_idx'):
        train_set.rename_field('words_idx','words')

    if dev_set.has_field('words_idx'):
        dev_set.rename_field('words_idx','words')

    if test_set.has_field('words_idx'):
        test_set.rename_field('words_idx','words')



    word_vocab = data_pkl[1]['words']
    pos_vocab = data_pkl[1]['pos']
    ner_vocab = data_pkl[1]['ner']
    chunk_vocab = data_pkl[1]['chunk']

    if embedding_path is not None:
        embed = StaticEmbedding(vocab=word_vocab, model_dir_or_name=embedding_path, word_dropout=0.01,
                                dropout=0.5,lower=True)
        return (train_set,dev_set,test_set),(word_vocab,pos_vocab,ner_vocab,chunk_vocab),embed
    else:
        return (train_set,dev_set,test_set),(word_vocab,pos_vocab,ner_vocab,chunk_vocab)


def load_conllized_ontonote_pkl_yf(path):
    def init_task(task):
        task_name = task.task_name
        for ds in [task.train_set, task.dev_set, task.test_set]:
            if ds.has_field('words'):
                ds.rename_field('words', 'x')
            else:
                ds.rename_field('words_idx', 'x')
            if ds.has_field('label'):
                ds.rename_field('label', 'y')
            else:
                ds.rename_field(task_name, 'y')
            ds.set_input('x', 'y', 'task_id')
            ds.set_target('y')

            if task_name in ['ner', 'chunk'] or 'pos' in task_name:
                ds.set_input('seq_len')
                ds.set_target('seq_len')
        return task
    #/remote-home/yfshao/workdir/datasets/conll03/data.pkl
    def pload(fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)

    DB = pload(path)
    task_lst = DB['task_lst']
    vocabs = DB['vocabs']
    task_lst = [init_task(task) for task in task_lst]

    return task_lst, vocabs


@cache_results(_cache_fp='weiboNER old uni+bi', _refresh=False)
def load_weibo_ner_old(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                  normlize={'char':True,'bigram':True,'word':False}):
    from fastNLP.io.data_loader import ConllLoader
    from utils import get_bigrams

    loader = ConllLoader(['chars','target'])
    # from fastNLP.io.file_reader import _read_conll
    # from fastNLP.core import Instance,DataSet
    # def _load(path):
    #     ds = DataSet()
    #     for idx, data in _read_conll(path, indexes=loader.indexes, dropna=loader.dropna,
    #                                 encoding='ISO-8859-1'):
    #         ins = {h: data[i] for i, h in enumerate(loader.headers)}
    #         ds.append(Instance(**ins))
    #     return ds
    # from fastNLP.io.utils import check_loader_paths
    # paths = check_loader_paths(path)
    # datasets = {name: _load(path) for name, path in paths.items()}
    datasets = {}
    train_path = os.path.join(path,'train.all.bmes')
    dev_path = os.path.join(path,'dev.all.bmes')
    test_path = os.path.join(path,'test.all.bmes')
    datasets['train'] = loader.load(train_path).datasets['train']
    datasets['dev'] = loader.load(dev_path).datasets['train']
    datasets['test'] = loader.load(test_path).datasets['train']

    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))

    vocabs = {}
    word_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    word_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['char'] = word_vocab
    vocabs['label'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        word_vocab.index_dataset(*list(datasets.values()), field_name='raw_words', new_field_name='words')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='raw_target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(word_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=0.01,normalize=normlize['char'])
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,normalize=normlize['bigram'])
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='weiboNER uni+bi', _refresh=False)
def load_weibo_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   normlize={'char':True,'bigram':True,'word':False}):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    loader = ConllLoader(['chars','target'])
    bundle = loader.load(path)

    datasets = bundle.datasets
    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    word_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    word_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['char'] = word_vocab
    vocabs['label'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        word_vocab.index_dataset(*list(datasets.values()), field_name='raw_words', new_field_name='words')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='raw_target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(word_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=0.01,normalize=normlize['char'])
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,normalize=normlize['bigram'])
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings



# datasets,vocabs = load_weibo_ner('/remote-home/xnli/data/corpus/sequence_labelling/ner_weibo')
#
# print(datasets['train'][:5])
# print(vocabs['word'].idx2word)
# print(vocabs['target'].idx2word)


@cache_results(_cache_fp='cache/ontonotes4ner',_refresh=False)
def load_ontonotes4ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                       normalize={'char':True,'bigram':True,'word':False}):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path,'train.char.bmes')
    dev_path = os.path.join(path,'dev.char.bmes')
    test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         normalize=normalize['char'])
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           normalize=normalize['bigram'])
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings



@cache_results(_cache_fp='cache/resume_ner',_refresh=False)
def load_resume_ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                    normalize={'char':True,'bigram':True,'word':False}):
    from fastNLP.io.data_loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path,'train.char.bmes')
    dev_path = os.path.join(path,'dev.char.bmes')
    test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,normalize=normalize['char'])
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,normalize=normalize['bigram'])
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings


@cache_results(_cache_fp='need_to_defined_fp',_refresh=False)
def equip_chinese_ner_with_skip(datasets,vocabs,embeddings,w_list,word_embedding_path=None,
                                normalize={'char':True,'bigram':True,'word':False}):
    from utils_ import Trie,get_skip_path
    from functools import partial
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    # for k,v in datasets.items():
    #     v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','skips')

    def skips2skips_l2r(chars,w_trie):
        '''

        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')

        lexicons = get_skip_path(chars,w_trie=w_trie)


        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0

        result = [[] for _ in range(len(chars))]

        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]

            result[e].append([s,w])

        return result

    def skips2skips_r2l(chars,w_trie):
        '''

        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')

        lexicons = get_skip_path(chars,w_trie=w_trie)


        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0

        result = [[] for _ in range(len(chars))]

        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]

            result[s].append([e,w])

        return result

    for k,v in datasets.items():
        v.apply_field(partial(skips2skips_l2r,w_trie=w_trie),'chars','skips_l2r')

    for k,v in datasets.items():
        v.apply_field(partial(skips2skips_r2l,w_trie=w_trie),'chars','skips_r2l')

    # print(v['skips_l2r'][0])
    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab
    for k,v in datasets.items():
        v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_l2r','skips_l2r_source')
        v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_l2r', 'skips_l2r_word')

    for k,v in datasets.items():
        v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_r2l','skips_r2l_source')
        v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_r2l', 'skips_r2l_word')

    for k,v in datasets.items():
        v.apply_field(lambda x:list(map(len,x)), 'skips_l2r_word', 'lexicon_count')
        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z:word_vocab.to_index(z),y)),x)),
                      'skips_l2r_word',new_field_name='skips_l2r_word')

        v.apply_field(lambda x:list(map(len,x)), 'skips_r2l_word', 'lexicon_count_back')

        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z:word_vocab.to_index(z),y)),x)),
                      'skips_r2l_word',new_field_name='skips_r2l_word')





    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0,normalize=normalize['word'])
        embeddings['word'] = word_embedding

    vocabs['char'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                              field_name='target', new_field_name='target')

    return datasets,vocabs,embeddings



@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list',_refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path,drop_characters=True):
    f = open(embedding_path,'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x:len(x) != 1, w_list))

    return w_list



# from pathes import *
#
# datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,
#                                                 yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path)
# print(datasets.keys())
# print(vocabs.keys())
# print(embeddings)
# yangjie_rich_pretrain_word_path
# datasets['train'].set_pad_val