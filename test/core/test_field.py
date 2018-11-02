import unittest

from fastNLP.core.field import CharTextField, LabelField, SeqLabelField


class TestField(unittest.TestCase):
    def test_char_field(self):
        text = "PhD applicants must submit a Research Plan and a resume " \
               "specify your class ranking written in English and a list of research" \
               " publications if any".split()
        max_word_len = max([len(w) for w in text])
        field = CharTextField(text, max_word_len, is_target=False)
        all_char = set()
        for word in text:
            all_char.update([ch for ch in word])
        char_vocab = {ch: idx + 1 for idx, ch in enumerate(all_char)}

        self.assertEqual(field.index(char_vocab),
                         [[char_vocab[ch] for ch in word] + [0] * (max_word_len - len(word)) for word in text])
        self.assertEqual(field.get_length(), len(text))
        self.assertEqual(field.contents(), text)
        tensor = field.to_tensor(50)
        self.assertEqual(tuple(tensor.shape), (50, max_word_len))

    def test_label_field(self):
        label = LabelField("A", is_target=True)
        self.assertEqual(label.get_length(), 1)
        self.assertEqual(label.index({"A": 10}), 10)

        label = LabelField(30, is_target=True)
        self.assertEqual(label.get_length(), 1)
        tensor = label.to_tensor(0)
        self.assertEqual(tensor.shape, ())
        self.assertEqual(int(tensor), 30)

    def test_seq_label_field(self):
        seq = ["a", "b", "c", "d", "a", "c", "a", "b"]
        field = SeqLabelField(seq)
        vocab = {"a": 10, "b": 20, "c": 30, "d": 40}
        self.assertEqual(field.index(vocab), [vocab[x] for x in seq])
        tensor = field.to_tensor(10)
        self.assertEqual(tuple(tensor.shape), (10,))
