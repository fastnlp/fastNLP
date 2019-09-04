
import unittest
from fastNLP import Vocabulary

class TestCRF(unittest.TestCase):
    def test_case1(self):
        # 检查allowed_transitions()能否正确使用
        from fastNLP.modules.decoder.crf import allowed_transitions

        id2label = {0: 'B', 1: 'I', 2:'O'}
        expected_res = {(0, 0), (0, 1), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 2),
                        (2, 4), (3, 0), (3, 2)}
        self.assertSetEqual(expected_res, set(allowed_transitions(id2label, include_start_end=True)))

        id2label = {0: 'B', 1:'M', 2:'E', 3:'S'}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 5), (3, 0), (3, 3), (3, 5), (4, 0), (4, 3)}
        self.assertSetEqual(expected_res, set(
            allowed_transitions(id2label, encoding_type='BMES', include_start_end=True)))

        id2label = {0: 'B', 1: 'I', 2:'O', 3: '<pad>', 4:"<unk>"}
        allowed_transitions(id2label, include_start_end=True)

        labels = ['O']
        for label in ['X', 'Y']:
            for tag in 'BI':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx:label for idx, label in enumerate(labels)}
        expected_res = {(0, 0), (0, 1), (0, 3), (0, 6), (1, 0), (1, 1), (1, 2), (1, 3), (1, 6), (2, 0), (2, 1),
                        (2, 2), (2, 3), (2, 6), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (4, 0), (4, 1), (4, 3),
                        (4, 4), (4, 6), (5, 0), (5, 1), (5, 3)}
        self.assertSetEqual(expected_res, set(allowed_transitions(id2label, include_start_end=True)))

        labels = []
        for label in ['X', 'Y']:
            for tag in 'BMES':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 7), (2, 9), (3, 0), (3, 3), (3, 4),
                        (3, 7), (3, 9), (4, 5), (4, 6), (5, 5), (5, 6), (6, 0), (6, 3), (6, 4), (6, 7), (6, 9), (7, 0),
                        (7, 3), (7, 4), (7, 7), (7, 9), (8, 0), (8, 3), (8, 4), (8, 7)}
        self.assertSetEqual(expected_res, set(
            allowed_transitions(id2label, include_start_end=True)))

    def test_case11(self):
        # 测试自动推断encoding类型
        from fastNLP.modules.decoder.crf import allowed_transitions

        id2label = {0: 'B', 1: 'I', 2: 'O'}
        expected_res = {(0, 0), (0, 1), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 2),
                        (2, 4), (3, 0), (3, 2)}
        self.assertSetEqual(expected_res, set(allowed_transitions(id2label, include_start_end=True)))

        id2label = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 5), (3, 0), (3, 3), (3, 5), (4, 0), (4, 3)}
        self.assertSetEqual(expected_res, set(
            allowed_transitions(id2label, include_start_end=True)))

        id2label = {0: 'B', 1: 'I', 2: 'O', 3: '<pad>', 4: "<unk>"}
        allowed_transitions(id2label, include_start_end=True)

        labels = ['O']
        for label in ['X', 'Y']:
            for tag in 'BI':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        expected_res = {(0, 0), (0, 1), (0, 3), (0, 6), (1, 0), (1, 1), (1, 2), (1, 3), (1, 6), (2, 0), (2, 1),
                        (2, 2), (2, 3), (2, 6), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (4, 0), (4, 1), (4, 3),
                        (4, 4), (4, 6), (5, 0), (5, 1), (5, 3)}
        self.assertSetEqual(expected_res, set(allowed_transitions(id2label, include_start_end=True)))

        labels = []
        for label in ['X', 'Y']:
            for tag in 'BMES':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 7), (2, 9), (3, 0), (3, 3), (3, 4),
                        (3, 7), (3, 9), (4, 5), (4, 6), (5, 5), (5, 6), (6, 0), (6, 3), (6, 4), (6, 7), (6, 9), (7, 0),
                        (7, 3), (7, 4), (7, 7), (7, 9), (8, 0), (8, 3), (8, 4), (8, 7)}
        self.assertSetEqual(expected_res, set(
            allowed_transitions(id2label, include_start_end=True)))

    def test_case12(self):
        # 测试能否通过vocab生成转移矩阵
        from fastNLP.modules.decoder.crf import allowed_transitions

        id2label = {0: 'B', 1: 'I', 2: 'O'}
        vocab = Vocabulary(unknown=None, padding=None)
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        expected_res = {(0, 0), (0, 1), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 2),
                        (2, 4), (3, 0), (3, 2)}
        self.assertSetEqual(expected_res, set(allowed_transitions(vocab, include_start_end=True)))

        id2label = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
        vocab = Vocabulary(unknown=None, padding=None)
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 5), (3, 0), (3, 3), (3, 5), (4, 0), (4, 3)}
        self.assertSetEqual(expected_res, set(
            allowed_transitions(vocab, include_start_end=True)))

        id2label = {0: 'B', 1: 'I', 2: 'O', 3: '<pad>', 4: "<unk>"}
        vocab = Vocabulary()
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        allowed_transitions(vocab, include_start_end=True)

        labels = ['O']
        for label in ['X', 'Y']:
            for tag in 'BI':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        expected_res = {(0, 0), (0, 1), (0, 3), (0, 6), (1, 0), (1, 1), (1, 2), (1, 3), (1, 6), (2, 0), (2, 1),
                        (2, 2), (2, 3), (2, 6), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (4, 0), (4, 1), (4, 3),
                        (4, 4), (4, 6), (5, 0), (5, 1), (5, 3)}
        vocab = Vocabulary(unknown=None, padding=None)
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        self.assertSetEqual(expected_res, set(allowed_transitions(vocab, include_start_end=True)))

        labels = []
        for label in ['X', 'Y']:
            for tag in 'BMES':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        vocab = Vocabulary(unknown=None, padding=None)
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 7), (2, 9), (3, 0), (3, 3), (3, 4),
                        (3, 7), (3, 9), (4, 5), (4, 6), (5, 5), (5, 6), (6, 0), (6, 3), (6, 4), (6, 7), (6, 9), (7, 0),
                        (7, 3), (7, 4), (7, 7), (7, 9), (8, 0), (8, 3), (8, 4), (8, 7)}
        self.assertSetEqual(expected_res, set(
            allowed_transitions(vocab, include_start_end=True)))


    def test_case2(self):
        # 测试CRF能否避免解码出非法跃迁, 使用allennlp做了验证。
        pass
        # import torch
        # from fastNLP.modules.decoder.crf import seq_len_to_byte_mask
        #
        # labels = ['O']
        # for label in ['X', 'Y']:
        #     for tag in 'BI':
        #         labels.append('{}-{}'.format(tag, label))
        # id2label = {idx: label for idx, label in enumerate(labels)}
        # num_tags = len(id2label)
        #
        # from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
        # allen_CRF = ConditionalRandomField(num_tags=num_tags, constraints=allowed_transitions('BIO', id2label),
        #                                    include_start_end_transitions=False)
        # batch_size = 3
        # logits = torch.nn.functional.softmax(torch.rand(size=(batch_size, 20, num_tags))).log()
        # trans_m = allen_CRF.transitions
        # seq_lens = torch.randint(1, 20, size=(batch_size,))
        # seq_lens[-1] = 20
        # mask = seq_len_to_byte_mask(seq_lens)
        # allen_res = allen_CRF.viterbi_tags(logits, mask)
        #
        # from fastNLP.modules.decoder.crf import ConditionalRandomField, allowed_transitions
        # fast_CRF = ConditionalRandomField(num_tags=num_tags, allowed_transitions=allowed_transitions(id2label))
        # fast_CRF.trans_m = trans_m
        # fast_res = fast_CRF.viterbi_decode(logits, mask, get_score=True, unpad=True)
        # # score equal
        # self.assertListEqual([score for _, score in allen_res], fast_res[1])
        # # seq equal
        # self.assertListEqual([_ for _, score in allen_res], fast_res[0])
        #
        #
        # labels = []
        # for label in ['X', 'Y']:
        #     for tag in 'BMES':
        #         labels.append('{}-{}'.format(tag, label))
        # id2label = {idx: label for idx, label in enumerate(labels)}
        # num_tags = len(id2label)
        #
        # from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
        # allen_CRF = ConditionalRandomField(num_tags=num_tags, constraints=allowed_transitions('BMES', id2label),
        #                                    include_start_end_transitions=False)
        # batch_size = 3
        # logits = torch.nn.functional.softmax(torch.rand(size=(batch_size, 20, num_tags))).log()
        # trans_m = allen_CRF.transitions
        # seq_lens = torch.randint(1, 20, size=(batch_size,))
        # seq_lens[-1] = 20
        # mask = seq_len_to_byte_mask(seq_lens)
        # allen_res = allen_CRF.viterbi_tags(logits, mask)
        #
        # from fastNLP.modules.decoder.crf import ConditionalRandomField, allowed_transitions
        # fast_CRF = ConditionalRandomField(num_tags=num_tags, allowed_transitions=allowed_transitions(id2label,
        #                                                                                              encoding_type='BMES'))
        # fast_CRF.trans_m = trans_m
        # fast_res = fast_CRF.viterbi_decode(logits, mask, get_score=True, unpad=True)
        # # score equal
        # self.assertListEqual([score for _, score in allen_res], fast_res[1])
        # # seq equal
        # self.assertListEqual([_ for _, score in allen_res], fast_res[0])

    def test_case3(self):
        # 测试crf的loss不会出现负数
        import torch
        from fastNLP.modules.decoder.crf import ConditionalRandomField
        from fastNLP.core.utils import seq_len_to_mask
        from torch import optim
        from torch import nn

        num_tags, include_start_end_trans = 4, True
        num_samples = 4
        lengths = torch.randint(3, 50, size=(num_samples, )).long()
        max_len = lengths.max()
        tags = torch.randint(num_tags, size=(num_samples, max_len))
        masks = seq_len_to_mask(lengths)
        feats = nn.Parameter(torch.randn(num_samples, max_len, num_tags))
        crf = ConditionalRandomField(num_tags, include_start_end_trans)
        optimizer = optim.SGD([param for param in crf.parameters() if param.requires_grad] + [feats], lr=0.1)
        for _ in range(10):
            loss = crf(feats, tags, masks).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if _%1000==0:
                print(loss)
            self.assertGreater(loss.item(), 0, "CRF loss cannot be less than 0.")
