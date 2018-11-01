import unittest

from fastNLP.core.field import CharTextField


class TestField(unittest.TestCase):
    def test_case(self):
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
