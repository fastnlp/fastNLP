
import unittest
import torch
import os

from fastNLP.modules.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from fastNLP.embeddings import GPT2WordPieceEncoder, GPT2Embedding
from fastNLP import DataSet, Vocabulary


class TestGPT2Embedding(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = GPT2Embedding(vocab, model_dir_or_name='en')
        words = torch.LongTensor([[2, 3, 4, 0]])
        print(embed(words).size())

        for pool_method in ['first', 'last', 'max', 'avg']:
            embed = GPT2Embedding(vocab, model_dir_or_name='en', pool_method=pool_method)
            print(embed(words).size())

    def test_gpt2_embedding(self):
        weight_path = 'test/data_for_tests/embedding/small_gpt2'
        vocab = Vocabulary().add_word_lst("this is a texta sentence".split())
        embed = GPT2Embedding(vocab, model_dir_or_name=weight_path, word_dropout=0.1)
        requires_grad = embed.requires_grad
        embed.requires_grad = not requires_grad
        embed.train()
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))

        embed = GPT2Embedding(vocab, model_dir_or_name=weight_path, word_dropout=0.1,
                              only_use_pretrain_bpe=False, language_model=True)
        embed.eval()
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))
        embed.get_lm_loss()

        vocab.add_word("NotInGpt2")
        embed = GPT2Embedding(vocab, model_dir_or_name=weight_path, word_dropout=0.1,
                              only_use_pretrain_bpe=False, auto_truncate=True, min_freq=1)
        words = torch.LongTensor([[2, 3, 4, 0]*20])
        result = embed(words)
        self.assertEqual(result.size(), (1, 80, 16))

    def test_gpt2_ebembedding_2(self):
        # 测试only_use_pretrain_vocab与truncate_embed是否正常工作
        Embedding = GPT2Embedding
        weight_path = 'test/data_for_tests/embedding/small_gpt2'
        vocab = Vocabulary().add_word_lst("this is a texta and".split())
        embed1 = Embedding(vocab, model_dir_or_name=weight_path,layers=list(range(3)),
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

    def test_gpt2_tokenizer(self):
        from fastNLP.modules.tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained('test/data_for_tests/embedding/small_gpt2')
        print(tokenizer.encode("this is a texta a sentence"))
        print(tokenizer.encode('this is'))

    def test_gpt2_embed_eq_gpt2_piece_encoder(self):
        # 主要检查一下embedding的结果与wordpieceencoder的结果是否一致
        weight_path = 'test/data_for_tests/embedding/small_gpt2'
        ds = DataSet({'words': ["this is a texta a sentence".split(), 'this is'.split()]})
        encoder = GPT2WordPieceEncoder(model_dir_or_name=weight_path)
        encoder.eval()
        encoder.index_datasets(ds, field_name='words')
        word_pieces = torch.LongTensor(ds['word_pieces'].get([0, 1]))
        word_pieces_res = encoder(word_pieces)

        vocab = Vocabulary()
        vocab.from_dataset(ds, field_name='words')
        vocab.index_dataset(ds, field_name='words', new_field_name='words')
        ds.set_input('words')
        words = torch.LongTensor(ds['words'].get([0, 1]))
        embed = GPT2Embedding(vocab, model_dir_or_name=weight_path, pool_method='first')
        embed.eval()
        words_res = embed(words)

        # 检查word piece什么的是正常work的
        self.assertEqual((word_pieces_res[0, :4]-words_res[0, :4]).sum(), 0)
        self.assertEqual((word_pieces_res[0, 5:]-words_res[0, 4:]).sum(), 0)
        self.assertEqual((word_pieces_res[1, :2]-words_res[1, :2]).sum(), 0)


class TestGPT2WordPieceEncoder(unittest.TestCase):
    @unittest.skipIf(True, "Only for local debugging")
    def test_eq_transformers(self):
        # 测试能否正确得到类似于transformers的结果
        weight_path = ''

        # tokenizer = transformers.GPT2Tokenizer.from_pretrained(weight_path)

        ds = DataSet({'words': ["this this this a is texta model vocab".split(), 'this is'.split()]})

        import transformers
        input1 = ' '.join(ds[0]['words'])
        input2 = ' '.join(ds[1]['words'])
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(weight_path)
        idx_list1 = tokenizer.encode(input1)
        idx_list2 = tokenizer.encode(input2)

        pad_value = tokenizer.encode('<|endoftext|>')[0]
        tensor = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(idx_list1),
                                                  torch.LongTensor(idx_list2)],
                                                 batch_first=True,
                                                 padding_value=pad_value)
        gpt2 = transformers.GPT2Model.from_pretrained(weight_path, output_hidden_states=True)
        gpt2.eval()
        tensor = tensor
        output, _, trans_hidden_states = gpt2(tensor, attention_mask=tensor.ne(pad_value))

        encoder = GPT2WordPieceEncoder(model_dir_or_name=weight_path, layers=list(range(13)))
        encoder.eval()
        encoder.index_datasets(ds, field_name='words', add_endoftext=False)
        word_pieces = torch.LongTensor(ds['word_pieces'].get([0, 1]))

        self.assertEqual(idx_list1, ds[0]['word_pieces'])
        self.assertEqual(idx_list2, ds[1]['word_pieces'])

        word_pieces_res = encoder(word_pieces)

        self.assertEqual((torch.cat(trans_hidden_states, dim=-1)-word_pieces_res).sum(), 0)

    @unittest.skipIf(True, "Only for local usage")
    def test_generate_small_gpt2(self):
        # 因为GPT2使用的是GPT2的tokenizer，所以没办法直接生成权重，需要用点下面的方式
        weight_path = ''
        tokenizer = GPT2Tokenizer.from_pretrained(weight_path)

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

        print(used_pairs)
        import json
        with open('test/data_for_tests/embedding/small_gpt2/vocab.json', 'w') as f:
            new_used_vocab = {}
            for idx, key in enumerate(used_vocab.keys()):
                new_used_vocab[key] = len(new_used_vocab)
            new_used_vocab['<|endoftext|>'] = len(new_used_vocab)
            for i in range(65, 91):
                if chr(i) not in new_used_vocab:
                    new_used_vocab[chr(i)] = len(new_used_vocab)
            for i in range(97, 123):
                if chr(i) not in new_used_vocab:
                    new_used_vocab[chr(i)] = len(new_used_vocab)

            json.dump(new_used_vocab, f)

        with open('test/data_for_tests/embedding/small_gpt2/merges.txt', 'w') as f:
            f.write('#version: small\n')
            for k,v in sorted(sorted(used_pairs.items(), key=lambda kv:kv[1])):
                f.write('{} {}\n'.format(k[0], k[1]))

        new_tokenizer = GPT2Tokenizer.from_pretrained('test/data_for_tests/embedding/small_gpt2')
        new_all_tokens = []
        for sent in [sent1, sent2, sent3]:
            tokens = new_tokenizer.tokenize(sent, add_prefix_space=True)
            new_all_tokens.extend(tokens)
        print(all_tokens, new_all_tokens)

        self.assertSequenceEqual(all_tokens, new_all_tokens)
        config = {
                      "architectures": [
                        "GPT2LMHeadModel"
                      ],
                      "initializer_range": 0.02,
                      "layer_norm_epsilon": 1e-05,
                      "n_ctx": 20,
                      "n_embd": 16,
                      "n_head": 4,
                      "n_layer": 2,
                      "n_positions": 20,
                      "vocab_size": len(new_used_vocab)
                    }
        with open('test/data_for_tests/embedding/small_gpt2/config.json', 'w') as f:
            json.dump(config, f)

        # 生成更小的merges.txt与vocab.json, 方法是通过记录tokenizer中的值实现
        from fastNLP.modules.encoder.gpt2 import GPT2LMHeadModel, GPT2Config

        config = GPT2Config.from_pretrained('test/data_for_tests/embedding/small_gpt2')

        model = GPT2LMHeadModel(config)
        torch.save(model.state_dict(), 'test/data_for_tests/embedding/small_gpt2/small_pytorch_model.bin')
        print(model(torch.LongTensor([[0,1,2,3]])))

    def test_gpt2_word_piece_encoder(self):
        # 主要检查可以运行
        weight_path = 'test/data_for_tests/embedding/small_gpt2'
        ds = DataSet({'words': ["this is a test sentence".split()]})
        embed = GPT2WordPieceEncoder(model_dir_or_name=weight_path, word_dropout=0.1)
        embed.index_datasets(ds, field_name='words')
        self.assertTrue(ds.has_field('word_pieces'))
        result = embed(torch.LongTensor([[1, 2, 3, 4]]))

        embed = GPT2WordPieceEncoder(model_dir_or_name=weight_path, word_dropout=0.1,
                                     language_model=True)
        embed.index_datasets(ds, field_name='words')
        self.assertTrue(ds.has_field('word_pieces'))
        result = embed(torch.LongTensor([[1, 2, 3, 4]]))

    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_generate(self):
        # weight_path = 'test/data_for_tests/embedding/small_gpt2'
        weight_path = 'en'

        encoder = GPT2WordPieceEncoder(model_dir_or_name=weight_path, language_model=True)

        # 测试一下各项东西是否正常work
        print(encoder.generate_from_str('This', max_len=20, do_sample=False, num_beams=1, temperature=1, top_k=50, top_p=1.0,
                          repetition_penalty=1.0, length_penalty=1.0))
        print(encoder.generate_from_str('This day', max_len=20, do_sample=False, num_beams=1, temperature=1, top_k=50, top_p=1.0,
                          repetition_penalty=1.0, length_penalty=1.0))
        print(encoder.generate_from_str('This', max_len=20, do_sample=True, num_beams=3, temperature=1, top_k=50, top_p=1.0,
                          repetition_penalty=1.0, length_penalty=1.0))
        print(encoder.generate_from_str('This', max_len=20, do_sample=True, num_beams=3, temperature=2, top_k=20, top_p=2.0,
                          repetition_penalty=2.0, length_penalty=1.5))
