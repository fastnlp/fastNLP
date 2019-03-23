import unittest

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric


class TestENAS(unittest.TestCase):
    def testENAS(self):
        # 从csv读取数据到DataSet
        sample_path = "tutorials/sample_data/tutorial_sample_dataset.csv"
        dataset = DataSet.read_csv(sample_path, headers=('raw_sentence', 'label'),
                                   sep='\t')
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
        dataset.drop(lambda x: x['seq_len'] <= 3)
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
        from fastNLP.core.batch import Batch
        from fastNLP.core.sampler import RandomSampler

        batch_iterator = Batch(dataset=train_data, batch_size=2, sampler=RandomSampler())
        for batch_x, batch_y in batch_iterator:
            print("batch_x has: ", batch_x)
            print("batch_y has: ", batch_y)
            break

        from fastNLP.automl.enas_model import ENASModel
        from fastNLP.automl.enas_controller import Controller
        model = ENASModel(embed_num=len(vocab), num_classes=5)
        controller = Controller()

        from fastNLP.automl.enas_trainer import ENASTrainer

        # 更改DataSet中对应field的名称，要以模型的forward等参数名一致
        train_data.rename_field('words', 'word_seq')  # input field 与 forward 参数一致
        train_data.rename_field('label', 'label_seq')
        test_data.rename_field('words', 'word_seq')
        test_data.rename_field('label', 'label_seq')

        loss = CrossEntropyLoss(pred="output", target="label_seq")
        metric = AccuracyMetric(pred="predict", target="label_seq")

        trainer = ENASTrainer(model=model, controller=controller, train_data=train_data, dev_data=test_data,
                          loss=CrossEntropyLoss(pred="output", target="label_seq"),
                          metrics=AccuracyMetric(pred="predict", target="label_seq"),
                          check_code_level=-1,
                          save_path=None,
                          batch_size=32,
                          print_every=1,
                          n_epochs=3,
                          final_epochs=1)
        trainer.train()
        print('Train finished!')

        # 调用Tester在test_data上评价效果
        from fastNLP import Tester

        tester = Tester(data=test_data, model=model, metrics=AccuracyMetric(pred="predict", target="label_seq"),
                        batch_size=4)

        acc = tester.test()
        print(acc)


if __name__ == '__main__':
    unittest.main()