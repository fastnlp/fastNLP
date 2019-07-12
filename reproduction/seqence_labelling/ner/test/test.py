
from reproduction.seqence_labelling.ner.data.Conll2003Loader import Conll2003DataLoader
from reproduction.seqence_labelling.ner.data.Conll2003Loader import iob2, iob2bioes
import unittest

class TestTagSchemaConverter(unittest.TestCase):
    def test_iob2(self):
        tags = ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
        golden = ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
        self.assertListEqual(golden, iob2(tags))

        tags = ['I-ORG', 'O']
        golden = ['B-ORG', 'O']
        self.assertListEqual(golden, iob2(tags))

        tags = ['I-MISC', 'I-MISC', 'O', 'I-PER', 'I-PER', 'O']
        golden = ['B-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
        self.assertListEqual(golden, iob2(tags))

    def test_iob2bemso(self):
        tags = ['B-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
        golden = ['B-MISC', 'E-MISC', 'O', 'B-PER', 'E-PER', 'O']
        self.assertListEqual(golden, iob2bioes(tags))


def test_conll2003_loader():
    path = '/hdd/fudanNLP/fastNLP/others/data/conll2003/train.txt'
    loader = Conll2003DataLoader().load(path)
    print(loader[:3])


if __name__ == '__main__':
    test_conll2003_loader()