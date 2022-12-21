import pytest
import os
from fastNLP import Vocabulary


@pytest.mark.torch
class TestCRF:
    def test_case1(self):
        from fastNLP.modules.torch.decoder.crf import allowed_transitions
        # 检查allowed_transitions()能否正确使用

        id2label = {0: 'B', 1: 'I', 2:'O'}
        expected_res = {(0, 0), (0, 1), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 2),
                        (2, 4), (3, 0), (3, 2)}
        assert expected_res == set(allowed_transitions(id2label, include_start_end=True))

        id2label = {0: 'B', 1:'M', 2:'E', 3:'S'}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 5), (3, 0), (3, 3), (3, 5), (4, 0), (4, 3)}
        assert (expected_res == set(
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
        assert (expected_res == set(allowed_transitions(id2label, include_start_end=True)))

        labels = []
        for label in ['X', 'Y']:
            for tag in 'BMES':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 7), (2, 9), (3, 0), (3, 3), (3, 4),
                        (3, 7), (3, 9), (4, 5), (4, 6), (5, 5), (5, 6), (6, 0), (6, 3), (6, 4), (6, 7), (6, 9), (7, 0),
                        (7, 3), (7, 4), (7, 7), (7, 9), (8, 0), (8, 3), (8, 4), (8, 7)}
        assert (expected_res == set(
            allowed_transitions(id2label, include_start_end=True)))

    def test_case11(self):
        # 测试自动推断encoding类型
        from fastNLP.modules.torch.decoder.crf import allowed_transitions

        id2label = {0: 'B', 1: 'I', 2: 'O'}
        expected_res = {(0, 0), (0, 1), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 2),
                        (2, 4), (3, 0), (3, 2)}
        assert (expected_res == set(allowed_transitions(id2label, include_start_end=True)))

        id2label = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 5), (3, 0), (3, 3), (3, 5), (4, 0), (4, 3)}
        assert (expected_res == set(
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
        assert (expected_res == set(allowed_transitions(id2label, include_start_end=True)))

        labels = []
        for label in ['X', 'Y']:
            for tag in 'BMES':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 7), (2, 9), (3, 0), (3, 3), (3, 4),
                        (3, 7), (3, 9), (4, 5), (4, 6), (5, 5), (5, 6), (6, 0), (6, 3), (6, 4), (6, 7), (6, 9), (7, 0),
                        (7, 3), (7, 4), (7, 7), (7, 9), (8, 0), (8, 3), (8, 4), (8, 7)}
        assert (expected_res == set(
            allowed_transitions(id2label, include_start_end=True)))

    def test_case12(self):
        # 测试能否通过vocab生成转移矩阵
        from fastNLP.modules.torch.decoder.crf import allowed_transitions

        id2label = {0: 'B', 1: 'I', 2: 'O'}
        vocab = Vocabulary(unknown=None, padding=None)
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        expected_res = {(0, 0), (0, 1), (0, 2), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (2, 2),
                        (2, 4), (3, 0), (3, 2)}
        assert (expected_res == set(allowed_transitions(vocab, include_start_end=True)))

        id2label = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
        vocab = Vocabulary(unknown=None, padding=None)
        for idx, tag in id2label.items():
            vocab.add_word(tag)
        expected_res = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 5), (3, 0), (3, 3), (3, 5), (4, 0), (4, 3)}
        assert (expected_res == set(
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
        assert (expected_res == set(allowed_transitions(vocab, include_start_end=True)))

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
        assert (expected_res == set(
            allowed_transitions(vocab, include_start_end=True)))

    # def test_case2(self):
    #     # 测试CRF能否避免解码出非法跃迁, 使用allennlp做了验证。
    #     pass
    #     import torch
    #     from fastNLP import seq_len_to_mask
    #
    #     labels = ['O']
    #     for label in ['X', 'Y']:
    #         for tag in 'BI':
    #             labels.append('{}-{}'.format(tag, label))
    #     id2label = {idx: label for idx, label in enumerate(labels)}
    #     num_tags = len(id2label)
    #     max_len = 10
    #     batch_size = 4
    #     bio_logits = torch.nn.functional.softmax(torch.rand(size=(batch_size, max_len, num_tags)), dim=-1).log()
    #     from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
    #     allen_CRF = ConditionalRandomField(num_tags=num_tags, constraints=allowed_transitions('BIO', id2label),
    #                                        include_start_end_transitions=False)
    #     bio_trans_m = allen_CRF.transitions
    #     bio_seq_lens = torch.randint(1, max_len, size=(batch_size,))
    #     bio_seq_lens[0] = 1
    #     bio_seq_lens[-1] = max_len
    #     mask = seq_len_to_mask(bio_seq_lens)
    #     allen_res = allen_CRF.viterbi_tags(bio_logits, mask)
    #
    #     from fastNLP.modules.decoder.crf import ConditionalRandomField, allowed_transitions
    #     fast_CRF = ConditionalRandomField(num_tags=num_tags, allowed_transitions=allowed_transitions(id2label,
    #                                                                                                  include_start_end=True))
    #     fast_CRF.trans_m = bio_trans_m
    #     fast_res = fast_CRF.viterbi_decode(bio_logits, mask, unpad=True)
    #     bio_scores = [round(score, 4) for _, score in allen_res]
    #     # score equal
    #     self.assertListEqual(bio_scores, [round(s, 4) for s in fast_res[1].tolist()])
    #     # seq equal
    #     bio_path = [_ for _, score in allen_res]
    #     self.assertListEqual(bio_path, fast_res[0])
    #
    #     labels = []
    #     for label in ['X', 'Y']:
    #         for tag in 'BMES':
    #             labels.append('{}-{}'.format(tag, label))
    #     id2label = {idx: label for idx, label in enumerate(labels)}
    #     num_tags = len(id2label)
    #
    #     from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
    #     allen_CRF = ConditionalRandomField(num_tags=num_tags, constraints=allowed_transitions('BMES', id2label),
    #                                        include_start_end_transitions=False)
    #     bmes_logits = torch.nn.functional.softmax(torch.rand(size=(batch_size, max_len, num_tags)), dim=-1).log()
    #     bmes_trans_m = allen_CRF.transitions
    #     bmes_seq_lens = torch.randint(1, max_len, size=(batch_size,))
    #     bmes_seq_lens[0] = 1
    #     bmes_seq_lens[-1] = max_len
    #     mask = seq_len_to_mask(bmes_seq_lens)
    #     allen_res = allen_CRF.viterbi_tags(bmes_logits, mask)
    #
    #     from fastNLP.modules.decoder.crf import ConditionalRandomField, allowed_transitions
    #     fast_CRF = ConditionalRandomField(num_tags=num_tags, allowed_transitions=allowed_transitions(id2label,
    #                                                                                                  encoding_type='BMES',
    #                                                                                                  include_start_end=True))
    #     fast_CRF.trans_m = bmes_trans_m
    #     fast_res = fast_CRF.viterbi_decode(bmes_logits, mask, unpad=True)
    #     # score equal
    #     bmes_scores = [round(score, 4) for _, score in allen_res]
    #     self.assertListEqual(bmes_scores, [round(s, 4) for s in fast_res[1].tolist()])
    #     # seq equal
    #     bmes_path = [_ for _, score in allen_res]
    #     self.assertListEqual(bmes_path, fast_res[0])
    #
    #     data = {
    #         'bio_logits': bio_logits.tolist(),
    #         'bio_scores': bio_scores,
    #         'bio_path': bio_path,
    #         'bio_trans_m': bio_trans_m.tolist(),
    #         'bio_seq_lens': bio_seq_lens.tolist(),
    #         'bmes_logits': bmes_logits.tolist(),
    #         'bmes_scores': bmes_scores,
    #         'bmes_path': bmes_path,
    #         'bmes_trans_m': bmes_trans_m.tolist(),
    #         'bmes_seq_lens': bmes_seq_lens.tolist(),
    #     }
    #
    #     with open('weights.json', 'w') as f:
    #         import json
    #         json.dump(data, f)

    def test_case2(self):
        # 测试CRF是否正常work。
        import json
        import torch
        from fastNLP import seq_len_to_mask
        folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(folder, '../../../', 'helpers/data/modules/decoder/crf.json')

        with open(os.path.abspath(path), 'r') as f:
            data = json.load(f)

        bio_logits = torch.FloatTensor(data['bio_logits'])
        bio_scores = data['bio_scores']
        bio_path = data['bio_path']
        bio_trans_m = torch.FloatTensor(data['bio_trans_m'])
        bio_seq_lens = torch.LongTensor(data['bio_seq_lens'])

        bmes_logits = torch.FloatTensor(data['bmes_logits'])
        bmes_scores = data['bmes_scores']
        bmes_path = data['bmes_path']
        bmes_trans_m = torch.FloatTensor(data['bmes_trans_m'])
        bmes_seq_lens = torch.LongTensor(data['bmes_seq_lens'])

        labels = ['O']
        for label in ['X', 'Y']:
            for tag in 'BI':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        num_tags = len(id2label)

        mask = seq_len_to_mask(bio_seq_lens)

        from fastNLP.modules.torch.decoder.crf import ConditionalRandomField, allowed_transitions
        fast_CRF = ConditionalRandomField(num_tags=num_tags, allowed_transitions=allowed_transitions(id2label,
                                                                                                     include_start_end=True))
        fast_CRF.trans_m.data = bio_trans_m
        fast_res = fast_CRF.viterbi_decode(bio_logits, mask, unpad=True)
        # score equal
        assert (bio_scores == [round(s, 4) for s in fast_res[1].tolist()])
        # seq equal
        assert (bio_path == fast_res[0])

        labels = []
        for label in ['X', 'Y']:
            for tag in 'BMES':
                labels.append('{}-{}'.format(tag, label))
        id2label = {idx: label for idx, label in enumerate(labels)}
        num_tags = len(id2label)

        mask = seq_len_to_mask(bmes_seq_lens)

        from fastNLP.modules.torch.decoder.crf import ConditionalRandomField, allowed_transitions
        fast_CRF = ConditionalRandomField(num_tags=num_tags, allowed_transitions=allowed_transitions(id2label,
                                                                                                     encoding_type='BMES',
                                                                                                     include_start_end=True))
        fast_CRF.trans_m.data = bmes_trans_m
        fast_res = fast_CRF.viterbi_decode(bmes_logits, mask, unpad=True)
        # score equal
        assert (bmes_scores == [round(s, 4) for s in fast_res[1].tolist()])
        # seq equal
        assert (bmes_path == fast_res[0])

    def test_case3(self):
        # 测试crf的loss不会出现负数
        import torch
        from fastNLP.modules.torch.decoder.crf import ConditionalRandomField
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
            assert (loss.item()> 0)

    def test_masking(self):
        # 测试crf的pad masking正常运行
        import torch
        from fastNLP.modules.torch.decoder.crf import ConditionalRandomField
        max_len = 5
        n_tags = 5
        pad_len = 5

        torch.manual_seed(4)
        logit = torch.rand(1, max_len+pad_len, n_tags)
        # logit[0, -1, :] = 0.0
        mask = torch.ones(1, max_len+pad_len)
        mask[0,-pad_len] = 0
        model = ConditionalRandomField(n_tags)
        pred, score = model.viterbi_decode(logit[:,:-pad_len], mask[:,:-pad_len])
        mask_pred, mask_score = model.viterbi_decode(logit, mask)
        assert (pred[0].tolist() == mask_pred[0,:-pad_len].tolist())

