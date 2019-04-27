import unittest

import fastNLP
from fastNLP.models.biaffine_parser import BiaffineParser, ParserLoss, ParserMetric

data_file = """
1       The     _       DET     DT      _       3       det     _       _
2       new     _       ADJ     JJ      _       3       amod    _       _
3       rate    _       NOUN    NN      _       6       nsubj   _       _
4       will    _       AUX     MD      _       6       aux     _       _
5       be      _       VERB    VB      _       6       cop     _       _
6       payable _       ADJ     JJ      _       0       root    _       _
7       mask    _       ADJ     JJ      _       6       punct    _       _
8       mask    _       ADJ     JJ      _       6       punct    _       _
9       cents   _       NOUN    NNS     _       4       nmod    _       _
10      from    _       ADP     IN      _       12      case    _       _
11      seven   _       NUM     CD      _       12      nummod  _       _
12      cents   _       NOUN    NNS     _       4       nmod    _       _
13      a       _       DET     DT      _       14      det     _       _
14      share   _       NOUN    NN      _       12      nmod:npmod      _       _
15      .       _       PUNCT   .       _       4       punct   _       _

1       The     _       DET     DT      _       3       det     _       _
2       new     _       ADJ     JJ      _       3       amod    _       _
3       rate    _       NOUN    NN      _       6       nsubj   _       _
4       will    _       AUX     MD      _       6       aux     _       _
5       be      _       VERB    VB      _       6       cop     _       _
6       payable _       ADJ     JJ      _       0       root    _       _
7       Feb.    _       PROPN   NNP     _       6       nmod:tmod       _       _
8       15      _       NUM     CD      _       7       nummod  _       _
9       .       _       PUNCT   .       _       6       punct   _       _

1       A       _       DET     DT      _       3       det     _       _
2       record  _       NOUN    NN      _       3       compound        _       _
3       date    _       NOUN    NN      _       7       nsubjpass       _       _
4       has     _       AUX     VBZ     _       7       aux     _       _
5       n't     _       PART    RB      _       7       neg     _       _
6       been    _       AUX     VBN     _       7       auxpass _       _
7       set     _       VERB    VBN     _       0       root    _       _
8       .       _       PUNCT   .       _       7       punct   _       _

"""


def init_data():
    ds = fastNLP.DataSet()
    v = {'words1': fastNLP.Vocabulary(),
         'words2': fastNLP.Vocabulary(),
         'label_true': fastNLP.Vocabulary()}
    data = []
    for line in data_file.split('\n'):
        line = line.split()
        if len(line) == 0 and len(data) > 0:
            data = list(zip(*data))
            ds.append(fastNLP.Instance(words1=data[1],
                                       words2=data[4],
                                       arc_true=data[6],
                                       label_true=data[7]))
            data = []
        elif len(line) > 0:
            data.append(line)

    for name in ['words1', 'words2', 'label_true']:
        ds.apply(lambda x: ['<st>'] + list(x[name]), new_field_name=name)
        ds.apply(lambda x: v[name].add_word_lst(x[name]))

    for name in ['words1', 'words2', 'label_true']:
        ds.apply(lambda x: [v[name].to_index(w) for w in x[name]], new_field_name=name)

    ds.apply(lambda x: [0] + list(map(int, x['arc_true'])), new_field_name='arc_true')
    ds.apply(lambda x: len(x['words1']), new_field_name='seq_len')
    ds.set_input('words1', 'words2', 'seq_len', flag=True)
    ds.set_target('arc_true', 'label_true', 'seq_len', flag=True)
    return ds, v['words1'], v['words2'], v['label_true']


class TestBiaffineParser(unittest.TestCase):
    def test_train(self):
        ds, v1, v2, v3 = init_data()
        model = BiaffineParser(init_embed=(len(v1), 30),
                               pos_vocab_size=len(v2), pos_emb_dim=30,
                               num_label=len(v3), encoder='var-lstm')
        trainer = fastNLP.Trainer(model=model, train_data=ds, dev_data=ds,
                                  loss=ParserLoss(), metrics=ParserMetric(), metric_key='UAS',
                                  batch_size=1, validate_every=10,
                                  n_epochs=10, use_tqdm=False)
        trainer.train(load_best_model=False)


if __name__ == '__main__':
    unittest.main()
