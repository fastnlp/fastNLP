import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from fastNLP.data.vocabulary import Vocabulary, DEFAULT_WORD_TO_INDEX

class TestVocabulary(unittest.TestCase):
    def test_vocab(self):
        import _pickle as pickle
        import os
        vocab = Vocabulary()
        filename = 'vocab'
        vocab.update(filename)
        vocab.update([filename, ['a'], [['b']], ['c']])
        idx = vocab[filename]
        before_pic = (vocab.to_word(idx), vocab[filename])

        with open(filename, 'wb') as f:
            pickle.dump(vocab, f)
        with open(filename, 'rb') as f:
            vocab = pickle.load(f)
        os.remove(filename)
        
        vocab.build_reverse_vocab()
        after_pic = (vocab.to_word(idx), vocab[filename])
        TRUE_DICT = {'vocab': 5, 'a': 6, 'b': 7, 'c': 8}
        TRUE_DICT.update(DEFAULT_WORD_TO_INDEX)
        TRUE_IDXDICT = {0: '<pad>', 1: '<unk>', 2: '<reserved-2>', 3: '<reserved-3>', 4: '<reserved-4>', 5: 'vocab', 6: 'a', 7: 'b', 8: 'c'}
        self.assertEqual(before_pic, after_pic)
        self.assertDictEqual(TRUE_DICT, vocab.word2idx)
        self.assertDictEqual(TRUE_IDXDICT, vocab.idx2word)
    
if __name__ == '__main__':
    unittest.main()