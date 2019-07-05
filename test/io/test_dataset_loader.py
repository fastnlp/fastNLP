import unittest
import os
from fastNLP.io import Conll2003Loader, PeopleDailyCorpusLoader, CSVLoader, JsonLoader
from fastNLP.io.dataset_loader import SSTLoader, SNLILoader
from reproduction.text_classification.data.yelpLoader import yelpLoader


class TestDatasetLoader(unittest.TestCase):
    
    def test_Conll2003Loader(self):
        """
            Test the the loader of Conll2003 dataset
        """
        dataset_path = "test/data_for_tests/conll_2003_example.txt"
        loader = Conll2003Loader()
        dataset_2003 = loader.load(dataset_path)
    
    def test_PeopleDailyCorpusLoader(self):
        data_set = PeopleDailyCorpusLoader().load("test/data_for_tests/people_daily_raw.txt")
    
    def test_CSVLoader(self):
        ds = CSVLoader(sep='\t', headers=['words', 'label']) \
            .load('test/data_for_tests/tutorial_sample_dataset.csv')
        assert len(ds) > 0
    
    def test_SNLILoader(self):
        ds = SNLILoader().load('test/data_for_tests/sample_snli.jsonl')
        assert len(ds) == 3
    
    def test_JsonLoader(self):
        ds = JsonLoader().load('test/data_for_tests/sample_snli.jsonl')
        assert len(ds) == 3

    def test_SST(self):
        train_data = """(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))
(4 (4 (4 (2 The) (4 (3 gorgeously) (3 (2 elaborate) (2 continuation)))) (2 (2 (2 of) (2 ``)) (2 (2 The) (2 (2 (2 Lord) (2 (2 of) (2 (2 the) (2 Rings)))) (2 (2 '') (2 trilogy)))))) (2 (3 (2 (2 is) (2 (2 so) (2 huge))) (2 (2 that) (3 (2 (2 (2 a) (2 column)) (2 (2 of) (2 words))) (2 (2 (2 (2 can) (1 not)) (3 adequately)) (2 (2 describe) (2 (3 (2 (2 co-writer\/director) (2 (2 Peter) (3 (2 Jackson) (2 's)))) (3 (2 expanded) (2 vision))) (2 (2 of) (2 (2 (2 J.R.R.) (2 (2 Tolkien) (2 's))) (2 Middle-earth))))))))) (2 .)))
(3 (3 (2 (2 (2 (2 (2 Singer\/composer) (2 (2 Bryan) (2 Adams))) (2 (2 contributes) (2 (2 (2 a) (2 slew)) (2 (2 of) (2 songs))))) (2 (2 --) (2 (2 (2 (2 a) (2 (2 few) (3 potential))) (2 (2 (2 hits) (2 ,)) (2 (2 (2 a) (2 few)) (1 (1 (2 more) (1 (2 simply) (2 intrusive))) (2 (2 to) (2 (2 the) (2 story))))))) (2 --)))) (2 but)) (3 (4 (2 the) (3 (2 whole) (2 package))) (2 (3 certainly) (3 (2 captures) (2 (1 (2 the) (2 (2 (2 intended) (2 (2 ,) (2 (2 er) (2 ,)))) (3 spirit))) (2 (2 of) (2 (2 the) (2 piece)))))))) (2 .))
(2 (2 (2 You) (2 (2 'd) (2 (2 think) (2 (2 by) (2 now))))) (2 (2 America) (2 (2 (2 would) (1 (2 have) (2 (2 (2 had) (1 (2 enough) (2 (2 of) (2 (2 plucky) (2 (2 British) (1 eccentrics)))))) (4 (2 with) (4 (3 hearts) (3 (2 of) (3 gold))))))) (2 .))))
"""
        test_data = """(3 (2 Yet) (3 (2 (2 the) (2 act)) (3 (4 (3 (2 is) (3 (2 still) (4 charming))) (2 here)) (2 .))))
(4 (2 (2 Whether) (2 (2 (2 (2 or) (1 not)) (3 (2 you) (2 (2 're) (3 (3 enlightened) (2 (2 by) (2 (2 any) (2 (2 of) (2 (2 Derrida) (2 's))))))))) (2 (2 lectures) (2 (2 on) (2 (2 ``) (2 (2 (2 (2 (2 (2 the) (2 other)) (2 '')) (2 and)) (2 ``)) (2 (2 the) (2 self)))))))) (3 (2 ,) (3 (2 '') (3 (2 Derrida) (3 (3 (2 is) (4 (2 an) (4 (4 (2 undeniably) (3 (4 (3 fascinating) (2 and)) (4 playful))) (2 fellow)))) (2 .))))))
(4 (3 (2 (2 Just) (2 (2 the) (2 labour))) (3 (2 involved) (3 (2 in) (4 (2 creating) (3 (3 (2 the) (3 (3 layered) (2 richness))) (3 (2 of) (3 (2 (2 the) (2 imagery)) (2 (2 in) (3 (2 (2 this) (2 chiaroscuro)) (2 (2 of) (2 (2 (2 madness) (2 and)) (2 light)))))))))))) (3 (3 (2 is) (4 astonishing)) (2 .)))
(3 (3 (2 Part) (3 (2 of) (4 (2 (2 the) (3 charm)) (2 (2 of) (2 (2 Satin) (2 Rouge)))))) (3 (3 (2 is) (3 (2 that) (3 (2 it) (2 (1 (2 avoids) (2 (2 the) (1 obvious))) (3 (2 with) (3 (3 (3 humour) (2 and)) (2 lightness))))))) (2 .)))
(4 (2 (2 a) (2 (2 screenplay) (2 more))) (3 (4 ingeniously) (2 (2 constructed) (2 (2 (2 (2 than) (2 ``)) (2 Memento)) (2 '')))))
(3 (2 ``) (3 (2 (2 Extreme) (2 Ops)) (3 (2 '') (4 (4 (3 exceeds) (2 expectations)) (2 .)))))
"""
        train, test = 'train--', 'test--'
        with open(train, 'w', encoding='utf-8') as f:
            f.write(train_data)
        with open(test, 'w', encoding='utf-8') as f:
            f.write(test_data)

        loader = SSTLoader()
        info = loader.process(
            {train: train, test: test},
            train_ds=[train],
            src_vocab_op=dict(min_freq=2)
        )
        assert len(list(info.vocabs.items())) == 2
        assert len(list(info.datasets.items())) == 2
        print(info.vocabs)
        print(info.datasets)
        os.remove(train), os.remove(test)
