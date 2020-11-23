import unittest

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.io.loader import CSVLoader


class TestTutorial(unittest.TestCase):
    def test_tutorial_1_data_preprocess(self):
        from fastNLP import DataSet
        data = {'raw_words': ["This is the first instance .", "Second instance .", "Third instance ."],
                'words': [['this', 'is', 'the', 'first', 'instance', '.'], ['Second', 'instance', '.'],
                          ['Third', 'instance', '.']],
                'seq_len': [6, 3, 3]}
        dataset = DataSet(data)
        # 传入的dict的每个key的value应该为具有相同长度的list

        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        instance = Instance(raw_words="This is the first instance",
                            words=['this', 'is', 'the', 'first', 'instance', '.'],
                            seq_len=6)
        dataset.append(instance)

        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet([
            Instance(raw_words="This is the first instance",
                     words=['this', 'is', 'the', 'first', 'instance', '.'],
                     seq_len=6),
            Instance(raw_words="Second instance .",
                     words=['Second', 'instance', '.'],
                     seq_len=3)
        ])

        from fastNLP import DataSet
        dataset = DataSet({'a': range(-5, 5), 'c': [0] * 10})

        # 不改变dataset，生成一个删除了满足条件的instance的新 DataSet
        dropped_dataset = dataset.drop(lambda ins: ins['a'] < 0, inplace=False)
        # 在dataset中删除满足条件的instance
        dataset.drop(lambda ins: ins['a'] < 0)
        #  删除第3个instance
        dataset.delete_instance(2)
        #  删除名为'a'的field
        dataset.delete_field('a')

        #  检查是否存在名为'a'的field
        print(dataset.has_field('a'))  # 或 ('a' in dataset)
        #  将名为'a'的field改名为'b'
        dataset.rename_field('c', 'b')
        #  DataSet的长度
        len(dataset)

        from fastNLP import DataSet
        data = {'raw_words': ["This is the first instance .", "Second instance .", "Third instance ."]}
        dataset = DataSet(data)

        # 将句子分成单词形式, 详见DataSet.apply()方法
        dataset.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')

        # 或使用DataSet.apply_field()
        dataset.apply_field(lambda sent: sent.split(), field_name='raw_words', new_field_name='words')

        # 除了匿名函数，也可以定义函数传递进去
        def get_words(instance):
            sentence = instance['raw_words']
            words = sentence.split()
            return words

        dataset.apply(get_words, new_field_name='words')
        
    def setUp(self):
        import os
        self._init_wd = os.path.abspath(os.curdir)

    def tearDown(self):
        import os
        os.chdir(self._init_wd)
        
class TestOldTutorial(unittest.TestCase):
    def test_fastnlp_10min_tutorial(self):
        # 从csv读取数据到DataSet
        sample_path = "test/data_for_tests/tutorial_sample_dataset.csv"
        dataset = CSVLoader(headers=['raw_sentence', 'label'], sep='	')._load(sample_path)
        print(len(dataset))
        print(dataset[0])
        print(dataset[-3])

        dataset.append(Instance(raw_sentence='fake data', label='0'))
        # 将所有数字转为小写
        dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
        # label转int
        dataset.apply(lambda x: int(x['label']), new_field_name='label')

        # 使用空格分割句子
        def split_sent(ins):
            return ins['raw_sentence'].split()

        dataset.apply(split_sent, new_field_name='words')

        # 增加长度信息
        dataset.apply(lambda x: len(x['words']), new_field_name='seq_len')
        print(len(dataset))
        print(dataset[0])

        # DataSet.drop(func)筛除数据
        dataset.drop(lambda x: x['seq_len'] <= 3, inplace=True)
        print(len(dataset))

        # 设置DataSet中，哪些field要转为tensor
        # set target，loss或evaluate中的golden，计算loss，模型评估时使用
        dataset.set_target("label")
        # set input，模型forward时使用
        dataset.set_input("words", "seq_len")

        # 分出测试集、训练集
        test_data, train_data = dataset.split(0.5)
        print(len(test_data))
        print(len(train_data))

        # 构建词表, Vocabulary.add(word)
        vocab = Vocabulary(min_freq=2)
        train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
        vocab.build_vocab()

        # index句子, Vocabulary.to_index(word)
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
        test_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
        print(test_data[0])

        # 如果你们需要做强化学习或者GAN之类的项目，你们也可以使用这些数据预处理的工具
        from fastNLP.core.batch import DataSetIter
        from fastNLP.core.sampler import RandomSampler

        batch_iterator = DataSetIter(dataset=train_data, batch_size=2, sampler=RandomSampler())
        for batch_x, batch_y in batch_iterator:
            print("batch_x has: ", batch_x)
            print("batch_y has: ", batch_y)
            break

        from fastNLP.models import CNNText
        model = CNNText((len(vocab), 50), num_classes=5, dropout=0.1)

        from fastNLP import Trainer
        from copy import deepcopy

        # 更改DataSet中对应field的名称，要以模型的forward等参数名一致
        train_data.rename_field('label', 'label_seq')
        test_data.rename_field('label', 'label_seq')

        loss = CrossEntropyLoss(target="label_seq")
        metric = AccuracyMetric(target="label_seq")

        # 实例化Trainer，传入模型和数据，进行训练
        # 先在test_data拟合（确保模型的实现是正确的）
        copy_model = deepcopy(model)
        overfit_trainer = Trainer(train_data=test_data, model=copy_model, loss=loss, batch_size=32, n_epochs=5,
                                  dev_data=test_data, metrics=metric, save_path=None)
        overfit_trainer.train()

        # 用train_data训练，在test_data验证
        trainer = Trainer(model=model, train_data=train_data, dev_data=test_data,
                          loss=CrossEntropyLoss(target="label_seq"),
                          metrics=AccuracyMetric(target="label_seq"),
                          save_path=None,
                          batch_size=32,
                          n_epochs=5)
        trainer.train()
        print('Train finished!')

        # 调用Tester在test_data上评价效果
        from fastNLP import Tester

        tester = Tester(data=test_data, model=model, metrics=AccuracyMetric(target="label_seq"),
                        batch_size=4)
        acc = tester.test()
        print(acc)

    def test_fastnlp_1min_tutorial(self):
        # tutorials/fastnlp_1min_tutorial.ipynb
        data_path = "test/data_for_tests/tutorial_sample_dataset.csv"
        ds = CSVLoader(headers=['raw_sentence', 'label'], sep='	')._load(data_path)
        print(ds[1])

        # 将所有数字转为小写
        ds.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
        # label转int
        ds.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

        def split_sent(ins):
            return ins['raw_sentence'].split()

        ds.apply(split_sent, new_field_name='words', is_input=True)

        # 分割训练集/验证集
        train_data, dev_data = ds.split(0.3)
        print("Train size: ", len(train_data))
        print("Test size: ", len(dev_data))

        from fastNLP import Vocabulary
        vocab = Vocabulary(min_freq=2)
        train_data.apply(lambda x: [vocab.add(word) for word in x['words']])

        # index句子, Vocabulary.to_index(word)
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words',
                         is_input=True)
        dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words',
                       is_input=True)

        from fastNLP.models import CNNText
        model = CNNText((len(vocab), 50), num_classes=5, dropout=0.1)

        from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Adam

        trainer = Trainer(train_data=train_data, model=model, optimizer=Adam(), loss=CrossEntropyLoss(),
                          dev_data=dev_data, metrics=AccuracyMetric(target='target'))
        trainer.train()
        print('Train finished!')

    def setUp(self):
        import os
        self._init_wd = os.path.abspath(os.curdir)

    def tearDown(self):
        import os
        os.chdir(self._init_wd)
