import unittest

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Tester
from fastNLP import Vocabulary
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.models import CNNText


class TestTutorial(unittest.TestCase):
    def test_tutorial(self):
        # 从csv读取数据到DataSet
        dataset = DataSet.read_csv("./data_for_tests/tutorial_sample_dataset.csv", headers=('raw_sentence', 'label'),
                                   sep='\t')
        print(len(dataset))
        print(dataset[0])

        dataset.append(Instance(raw_sentence='fake data', label='0'))
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
        dataset.drop(lambda x: x['seq_len'] <= 3)
        print(len(dataset))

        # 设置DataSet中，哪些field要转为tensor
        # set target，loss或evaluate中的golden，计算loss，模型评估时使用
        dataset.set_target("label")
        # set input，模型forward时使用
        dataset.set_input("words")

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

        model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)

        from fastNLP import Trainer
        from copy import deepcopy

        # 更改DataSet中对应field的名称，要以模型的forward等参数名一致
        train_data.rename_field('words', 'word_seq')  # input field 与 forward 参数一致
        train_data.rename_field('label', 'label_seq')
        test_data.rename_field('words', 'word_seq')
        test_data.rename_field('label', 'label_seq')

        # 实例化Trainer，传入模型和数据，进行训练
        copy_model = deepcopy(model)
        overfit_trainer = Trainer(model=copy_model, train_data=test_data, dev_data=test_data,
                                  losser=CrossEntropyLoss(input="output", target="label_seq"),
                                  metrics=AccuracyMetric(pred="predict", target="label_seq"),
                                  save_path="./save",
                                  batch_size=4,
                                  n_epochs=10)
        overfit_trainer.train()

        trainer = Trainer(model=model, train_data=train_data, dev_data=test_data,
                          losser=CrossEntropyLoss(input="output", target="label_seq"),
                          metrics=AccuracyMetric(pred="predict", target="label_seq"),
                          save_path="./save",
                          batch_size=4,
                          n_epochs=10)
        trainer.train()
        print('Train finished!')

        # 使用fastNLP的Tester测试脚本

        tester = Tester(data=test_data, model=model, metrics=AccuracyMetric(pred="predict", target="label_seq"),
                        batch_size=4)
        acc = tester.test()
        print(acc)
