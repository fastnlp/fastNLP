from time import time
from datetime import timedelta

from fastNLP.io.dataset_loader import JsonLoader
from fastNLP.modules.encoder._bert import BertTokenizer
from fastNLP.io.data_bundle import DataBundle
from fastNLP.core.const import Const

class BertData(JsonLoader):
    
    def __init__(self, max_nsents=60, max_ntokens=100, max_len=512):
        
        fields = {'article': 'article', 
                  'label': 'label'}
        super(BertData, self).__init__(fields=fields)

        self.max_nsents = max_nsents
        self.max_ntokens = max_ntokens
        self.max_len = max_len

        self.tokenizer = BertTokenizer.from_pretrained('/path/to/uncased_L-12_H-768_A-12')
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']

    def _load(self, paths):
        dataset = super(BertData, self)._load(paths)
        return dataset

    def process(self, paths):
        
        def truncate_articles(instance, max_nsents=self.max_nsents, max_ntokens=self.max_ntokens):
            article = [' '.join(sent.lower().split()[:max_ntokens]) for sent in instance['article']]
            return article[:max_nsents]

        def truncate_labels(instance):
            label = list(filter(lambda x: x < len(instance['article']), instance['label']))
            return label
        
        def bert_tokenize(instance, tokenizer, max_len, pad_value):
            article = instance['article']
            article = ' [SEP] [CLS] '.join(article)
            word_pieces = tokenizer.tokenize(article)[:(max_len - 2)]
            word_pieces = ['[CLS]'] + word_pieces + ['[SEP]']
            token_ids = tokenizer.convert_tokens_to_ids(word_pieces)
            while len(token_ids) < max_len:
                token_ids.append(pad_value)
            assert len(token_ids) == max_len
            return token_ids

        def get_seg_id(instance, max_len, sep_id):
            _segs = [-1] + [i for i, idx in enumerate(instance['article']) if idx == sep_id]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            segment_id = []
            for i, length in enumerate(segs):
                if i % 2 == 0:
                    segment_id += length * [0]
                else:
                    segment_id += length * [1]
            while len(segment_id) < max_len:
                segment_id.append(0)
            return segment_id
        
        def get_cls_id(instance, cls_id):
            classification_id = [i for i, idx in enumerate(instance['article']) if idx == cls_id]
            return classification_id
        
        def get_labels(instance):
            labels = [0] * len(instance['cls_id'])
            label_idx = list(filter(lambda x: x < len(instance['cls_id']), instance['label']))
            for idx in label_idx:
                labels[idx] = 1
            return labels

        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])
            
            # remove empty samples
            datasets[name].drop(lambda ins: len(ins['article']) == 0 or len(ins['label']) == 0)
            
            # truncate articles
            datasets[name].apply(lambda ins: truncate_articles(ins, self.max_nsents, self.max_ntokens), new_field_name='article')
            
            # truncate labels
            datasets[name].apply(truncate_labels, new_field_name='label')
            
            # tokenize and convert tokens to id
            datasets[name].apply(lambda ins: bert_tokenize(ins, self.tokenizer, self.max_len, self.pad_id), new_field_name='article')
            
            # get segment id
            datasets[name].apply(lambda ins: get_seg_id(ins, self.max_len, self.sep_id), new_field_name='segment_id')
            
            # get classification id
            datasets[name].apply(lambda ins: get_cls_id(ins, self.cls_id), new_field_name='cls_id')

            # get label
            datasets[name].apply(get_labels, new_field_name='label')
            
            # rename filed
            datasets[name].rename_field('article', Const.INPUTS(0))
            datasets[name].rename_field('segment_id', Const.INPUTS(1))
            datasets[name].rename_field('cls_id', Const.INPUTS(2))
            datasets[name].rename_field('lbael', Const.TARGET)

            # set input and target
            datasets[name].set_input(Const.INPUTS(0), Const.INPUTS(1), Const.INPUTS(2))
            datasets[name].set_target(Const.TARGET)
            
            # set paddding value
            datasets[name].set_pad_val('article', 0)

        return DataBundle(datasets=datasets)


class BertSumLoader(JsonLoader):
    
    def __init__(self):
        fields = {'article': 'article',
               'segment_id': 'segment_id',
                   'cls_id': 'cls_id',
                    'label': Const.TARGET
                 }
        super(BertSumLoader, self).__init__(fields=fields)

    def _load(self, paths):
        dataset = super(BertSumLoader, self)._load(paths)
        return dataset

    def process(self, paths):
        
        def get_seq_len(instance):
            return len(instance['article'])

        print('Start loading datasets !!!')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])
            
            datasets[name].apply(get_seq_len, new_field_name='seq_len')

            # set input and target
            datasets[name].set_input('article', 'segment_id', 'cls_id')
            datasets[name].set_target(Const.TARGET)
        
            # set padding value
            datasets[name].set_pad_val('article', 0)
            datasets[name].set_pad_val('segment_id', 0)
            datasets[name].set_pad_val('cls_id', -1)
            datasets[name].set_pad_val(Const.TARGET, 0)

        print('Finished in {}'.format(timedelta(seconds=time()-start)))

        return DataBundle(datasets=datasets)
