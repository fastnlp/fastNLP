
import unittest

import torch
import os

from fastNLP import DataSet, Vocabulary
from fastNLP.embeddings.roberta_embedding import RobertaWordPieceEncoder, RobertaEmbedding


class TestRobertWordPieceEncoder(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = RobertaEmbedding(vocab, model_dir_or_name='en')
        words = torch.LongTensor([[2, 3, 4, 0]])
        print(embed(words).size())

        for pool_method in ['first', 'last', 'max', 'avg']:
            for include_cls_sep in [True, False]:
                embed = RobertaEmbedding(vocab, model_dir_or_name='en', pool_method=pool_method,
                                      include_cls_sep=include_cls_sep)
                print(embed(words).size())

    def test_robert_word_piece_encoder(self):
        # 可正常运行即可
        weight_path = 'tests/data_for_tests/embedding/small_roberta'
        encoder = RobertaWordPieceEncoder(model_dir_or_name=weight_path, word_dropout=0.1)
        ds = DataSet({'words': ["this is a test . [SEP]".split()]})
        encoder.index_datasets(ds, field_name='words')
        self.assertTrue(ds.has_field('word_pieces'))
        result = encoder(torch.LongTensor([[1,2,3,4]]))

    def test_roberta_embed_eq_roberta_piece_encoder(self):
        # 主要检查一下embedding的结果与wordpieceencoder的结果是否一致
        weight_path = 'tests/data_for_tests/embedding/small_roberta'
        ds = DataSet({'words': ["this is a texta a sentence".split(), 'this is'.split()]})
        encoder = RobertaWordPieceEncoder(model_dir_or_name=weight_path)
        encoder.eval()
        encoder.index_datasets(ds, field_name='words')
        word_pieces = torch.LongTensor(ds['word_pieces'].get([0, 1]))
        word_pieces_res = encoder(word_pieces)

        vocab = Vocabulary()
        vocab.from_dataset(ds, field_name='words')
        vocab.index_dataset(ds, field_name='words', new_field_name='words')
        ds.set_input('words')
        words = torch.LongTensor(ds['words'].get([0, 1]))
        embed = RobertaEmbedding(vocab, model_dir_or_name=weight_path,
                                pool_method='first', include_cls_sep=True, pooled_cls=False, min_freq=1)
        embed.eval()
        words_res = embed(words)

        # 检查word piece什么的是正常work的
        self.assertEqual((word_pieces_res[0, :5]-words_res[0, :5]).sum(), 0)
        self.assertEqual((word_pieces_res[0, 6:]-words_res[0, 5:]).sum(), 0)
        self.assertEqual((word_pieces_res[1, :3]-words_res[1, :3]).sum(), 0)

    @unittest.skipIf(True, "Only for local debugging")
    def test_eq_transformers(self):
        weight_path = ''
        ds = DataSet({'words': ["this is a texta model vocab".split(), 'this is'.split()]})
        encoder = RobertaWordPieceEncoder(model_dir_or_name=weight_path)
        encoder.eval()
        encoder.index_datasets(ds, field_name='words')
        word_pieces = torch.LongTensor(ds['word_pieces'].get([0, 1]))
        word_pieces_res = encoder(word_pieces)

        import transformers
        input1 = ' '.join(ds[0]['words'])
        input2 = ' '.join(ds[1]['words'])
        tokenizer = transformers.RobertaTokenizer.from_pretrained(weight_path)
        idx_list1 = tokenizer.encode(input1)
        idx_list2 = tokenizer.encode(input2)
        self.assertEqual(idx_list1, ds[0]['word_pieces'])
        self.assertEqual(idx_list2, ds[1]['word_pieces'])

        pad_value = tokenizer.encode('<pad>')[0]
        tensor = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(idx_list1),
                                                  torch.LongTensor(idx_list2)],
                                                 batch_first=True,
                                                 padding_value=pad_value)
        roberta = transformers.RobertaModel.from_pretrained(weight_path, output_hidden_states=True)
        roberta.eval()
        output, pooled_output, hidden_states = roberta(tensor, attention_mask=tensor.ne(pad_value))

        self.assertEqual((output-word_pieces_res).sum(), 0)

    @unittest.skipIf(True, "Only for local usage")
    def test_generate_small_roberta(self):
        """
        因为Roberta使用的是GPT2的tokenizer，所以没办法直接生成权重，需要用点下面的方式

        :return:
        """
        weight_path = ''
        from fastNLP.modules.tokenizer import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(weight_path)

        used_pairs = {}
        used_vocab = {}
        # 修改这里即可获得更多的sentence的数据
        sent1 = "This is a demo sentence"
        sent2 = "another demo"
        sent3 = 'this is a texta model vocab'
        all_tokens = []

        for sent in [sent1, sent2, sent3]:
            tokens = []
            for word in sent.split():
                word = ' '+ word
                token = "".join(
                    tokenizer.byte_encoder[b] for b in word.encode("utf-8")
                )
                _token, _used_pairs = tokenizer.get_used_merge_pair_vocab(token)
                tokens.extend(_token.split())
                used_pairs.update(_used_pairs)
            all_tokens.extend(tokens)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            used_vocab.update({t:i for t,i in zip(tokens, token_ids)})

        import json
        with open('tests/data_for_tests/embedding/small_roberta/vocab.json', 'w') as f:
            new_used_vocab = {}
            for token in ['<s>', '<pad>', '</s>', '<unk>', '<mask>']:  # <pad>必须为1
                new_used_vocab[token] = len(new_used_vocab)
            for i in range(65, 91):
                if chr(i) not in new_used_vocab:
                    new_used_vocab[chr(i)] = len(new_used_vocab)
            for i in range(97, 123):
                if chr(i) not in new_used_vocab:
                    new_used_vocab[chr(i)] = len(new_used_vocab)
            for idx, key in enumerate(used_vocab.keys()):
                if key not in new_used_vocab:
                 new_used_vocab[key] = len(new_used_vocab)
            json.dump(new_used_vocab, f)

        with open('tests/data_for_tests/embedding/small_roberta/merges.txt', 'w') as f:
            f.write('#version: tiny\n')
            for k,v in sorted(sorted(used_pairs.items(), key=lambda kv:kv[1])):
                f.write('{} {}\n'.format(k[0], k[1]))

        config = {
              "architectures": [
                "RobertaForMaskedLM"
              ],
              "attention_probs_dropout_prob": 0.1,
              "finetuning_task": None,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 16,
              "initializer_range": 0.02,
              "intermediate_size": 20,
              "layer_norm_eps": 1e-05,
              "max_position_embeddings": 20,
              "num_attention_heads": 4,
              "num_hidden_layers": 2,
              "num_labels": 2,
              "output_attentions": False,
              "output_hidden_states": False,
              "torchscript": False,
              "type_vocab_size": 1,
              "vocab_size": len(new_used_vocab)
            }
        with open('tests/data_for_tests/embedding/small_roberta/config.json', 'w') as f:
            json.dump(config, f)

        new_tokenizer = RobertaTokenizer.from_pretrained('tests/data_for_tests/embedding/small_roberta')
        new_all_tokens = []
        for sent in [sent1, sent2, sent3]:
            tokens = new_tokenizer.tokenize(sent, add_prefix_space=True)
            new_all_tokens.extend(tokens)
        print(all_tokens, new_all_tokens)

        self.assertSequenceEqual(all_tokens, new_all_tokens)

        # 生成更小的merges.txt与vocab.json, 方法是通过记录tokenizer中的值实现
        from fastNLP.modules.encoder.roberta import RobertaModel, BertConfig

        config = BertConfig.from_json_file('tests/data_for_tests/embedding/small_roberta/config.json')

        model = RobertaModel(config)
        torch.save(model.state_dict(), 'tests/data_for_tests/embedding/small_roberta/small_pytorch_model.bin')
        print(model(torch.LongTensor([[0,1,2,3]])))

    def test_save_load(self):
        bert_save_test = 'roberta_save_test'
        try:
            os.makedirs(bert_save_test, exist_ok=True)
            embed = RobertaWordPieceEncoder(model_dir_or_name='tests/data_for_tests/embedding/small_roberta', word_dropout=0.0,
                                         layers='-2')
            ds = DataSet({'words': ["this is a test . [SEP]".split()]})
            embed.index_datasets(ds, field_name='words')
            self.assertTrue(ds.has_field('word_pieces'))
            words = torch.LongTensor([[1, 2, 3, 4]])
            embed.save(bert_save_test)
            load_embed = RobertaWordPieceEncoder.load(bert_save_test)
            embed.eval(), load_embed.eval()
            self.assertEqual((embed(words) - load_embed(words)).sum(), 0)
        finally:
            import shutil
            shutil.rmtree(bert_save_test)


class TestRobertaEmbedding(unittest.TestCase):
    def test_roberta_embedding_1(self):
        weight_path = 'tests/data_for_tests/embedding/small_roberta'
        vocab = Vocabulary().add_word_lst("this is a test . [SEP] NotInRoberta".split())
        embed = RobertaEmbedding(vocab, model_dir_or_name=weight_path, word_dropout=0.1)
        requires_grad = embed.requires_grad
        embed.requires_grad = not requires_grad
        embed.train()
        words = torch.LongTensor([[2, 3, 4, 1]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))

        # 自动截断而不报错
        embed = RobertaEmbedding(vocab, model_dir_or_name=weight_path, word_dropout=0.1, auto_truncate=True)
        words = torch.LongTensor([[2, 3, 4, 1]*10,
                                  [2, 3]+[0]*38])
        result = embed(words)
        self.assertEqual(result.size(), (2, 40, 16))

    def test_roberta_ebembedding_2(self):
        # 测试only_use_pretrain_vocab与truncate_embed是否正常工作
        Embedding = RobertaEmbedding
        weight_path = 'tests/data_for_tests/embedding/small_roberta'
        vocab = Vocabulary().add_word_lst("this is a texta and".split())
        embed1 = Embedding(vocab, model_dir_or_name=weight_path, layers=list(range(3)),
                              only_use_pretrain_bpe=True, truncate_embed=True, min_freq=1)
        # embed_bpe_vocab_size = len(vocab)-1 + 2  # 排除NotInBERT, 额外加##a, [CLS]
        # self.assertEqual(embed_bpe_vocab_size, len(embed1.model.tokenzier.vocab))

        embed2 = Embedding(vocab, model_dir_or_name=weight_path, layers=list(range(3)),
                              only_use_pretrain_bpe=True, truncate_embed=False, min_freq=1)
        # embed_bpe_vocab_size = num_word  # 排除NotInBERT
        # self.assertEqual(embed_bpe_vocab_size, len(embed2.model.tokenzier.vocab))

        embed3 = Embedding(vocab, model_dir_or_name=weight_path, layers=list(range(3)),
                              only_use_pretrain_bpe=False, truncate_embed=True, min_freq=1)
        # embed_bpe_vocab_size = len(vocab)+2  # 新增##a, [CLS]
        # self.assertEqual(embed_bpe_vocab_size, len(embed3.model.tokenzier.vocab))

        embed4 = Embedding(vocab, model_dir_or_name=weight_path, layers=list(range(3)),
                              only_use_pretrain_bpe=False, truncate_embed=False, min_freq=1)
        # embed_bpe_vocab_size = num_word+1  # 新增##a
        # self.assertEqual(embed_bpe_vocab_size, len(embed4.model.tokenzier.vocab))

        # 测试各种情况下以下tensor的值是相等的
        embed1.eval()
        embed2.eval()
        embed3.eval()
        embed4.eval()
        tensor = torch.LongTensor([[vocab.to_index(w) for w in 'this is a texta and'.split()]])
        t1 = embed1(tensor)
        t2 = embed2(tensor)
        t3 = embed3(tensor)
        t4 = embed4(tensor)

        self.assertEqual((t1-t2).sum(), 0)
        self.assertEqual((t1-t3).sum(), 0)
        self.assertEqual((t1-t4).sum(), 0)

    def test_save_load(self):
        bert_save_test = 'roberta_save_test'
        try:
            os.makedirs(bert_save_test, exist_ok=True)
            vocab = Vocabulary().add_word_lst("this is a test . [SEP] NotInBERT".split())
            embed = RobertaEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_roberta',
                                     word_dropout=0.1,
                                     auto_truncate=True)
            embed.save(bert_save_test)
            load_embed = RobertaEmbedding.load(bert_save_test)
            words = torch.randint(len(vocab), size=(2, 20))
            embed.eval(), load_embed.eval()
            self.assertEqual((embed(words) - load_embed(words)).sum(), 0)
        finally:
            import shutil
            shutil.rmtree(bert_save_test)
