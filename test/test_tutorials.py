import unittest

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric


class TestTutorial(unittest.TestCase):
    def test_fastnlp_10min_tutorial(self):
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

        from fastNLP.models import CNNText
        model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)

        from fastNLP import Trainer
        from copy import deepcopy

        # 更改DataSet中对应field的名称，要以模型的forward等参数名一致
        train_data.rename_field('words', 'word_seq')  # input field 与 forward 参数一致
        train_data.rename_field('label', 'label_seq')
        test_data.rename_field('words', 'word_seq')
        test_data.rename_field('label', 'label_seq')

        loss = CrossEntropyLoss(pred="output", target="label_seq")
        metric = AccuracyMetric(pred="predict", target="label_seq")

        # 实例化Trainer，传入模型和数据，进行训练
        # 先在test_data拟合（确保模型的实现是正确的）
        copy_model = deepcopy(model)
        overfit_trainer = Trainer(model=copy_model, train_data=test_data, dev_data=test_data,
                                  loss=loss,
                                  metrics=metric,
                                  save_path=None,
                                  batch_size=32,
                                  n_epochs=5)
        overfit_trainer.train()

        # 用train_data训练，在test_data验证
        trainer = Trainer(model=model, train_data=train_data, dev_data=test_data,
                          loss=CrossEntropyLoss(pred="output", target="label_seq"),
                          metrics=AccuracyMetric(pred="predict", target="label_seq"),
                          save_path=None,
                          batch_size=32,
                          n_epochs=5)
        trainer.train()
        print('Train finished!')

        # 调用Tester在test_data上评价效果
        from fastNLP import Tester

        tester = Tester(data=test_data, model=model, metrics=AccuracyMetric(pred="predict", target="label_seq"),
                        batch_size=4)
        acc = tester.test()
        print(acc)

    def test_fastnlp_1min_tutorial(self):
        # tutorials/fastnlp_1min_tutorial.ipynb
        data_path = "tutorials/sample_data/tutorial_sample_dataset.csv"
        ds = DataSet.read_csv(data_path, headers=('raw_sentence', 'label'), sep='\t')
        print(ds[1])

        # 将所有数字转为小写
        ds.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
        # label转int
        ds.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

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
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq',
                         is_input=True)
        dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq',
                       is_input=True)

        from fastNLP.models import CNNText
        model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)

        from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
        trainer = Trainer(model=model,
                          train_data=train_data,
                          dev_data=dev_data,
                          loss=CrossEntropyLoss(),
                          metrics=AccuracyMetric()
                          )
        trainer.train()
        print('Train finished!')

    def test_fastnlp_advanced_tutorial(self):
        import os
        os.chdir("tutorials/fastnlp_advanced_tutorial")

        from fastNLP import DataSet
        from fastNLP import Instance
        from fastNLP import Vocabulary
        from fastNLP import Trainer
        from fastNLP import Tester

        # ### Instance
        # Instance表示一个样本，由一个或者多个field（域、属性、特征）组成，每个field具有自己的名字以及值
        # 在初始化Instance的时候可以定义它包含的field，使用"field_name=field_value"的写法

        # In[2]:

        # 组织一个Instance，这个Instance由premise、hypothesis、label三个field组成
        instance = Instance(premise='an premise example .', hypothesis='an hypothesis example.', label=1)
        instance

        # In[3]:

        data_set = DataSet([instance] * 5)
        data_set.append(instance)
        data_set[-2:]

        # In[4]:

        # 如果某一个field的类型与dataset对应的field类型不一样仍可被加入dataset中
        instance2 = Instance(premise='the second premise example .', hypothesis='the second hypothesis example.',
                             label='1')
        try:
            data_set.append(instance2)
        except:
            pass
        data_set[-2:]

        # In[5]:

        # 如果某一个field的名字不对，则该instance不能被append到dataset中
        instance3 = Instance(premises='the third premise example .', hypothesis='the third hypothesis example.',
                             label=1)
        try:
            data_set.append(instance3)
        except:
            print('cannot append instance')
            pass
        data_set[-2:]

        # In[6]:

        # 除了文本以外，还可以将tensor作为其中一个field的value
        import torch
        tensor_ins = Instance(image=torch.randn(5, 5), label=0)
        ds = DataSet()
        ds.append(tensor_ins)
        ds

        from fastNLP import DataSet
        from fastNLP import Instance

        # 从csv读取数据到DataSet
        # 类csv文件，即每一行为一个example的文件，都可以使用这种方法进行数据读取
        dataset = DataSet.read_csv('tutorial_sample_dataset.csv', headers=('raw_sentence', 'label'), sep='\t')
        # 查看DataSet的大小
        len(dataset)

        # In[8]:

        # 使用数字索引[k]，获取第k个样本
        dataset[0]

        # In[9]:

        # 获取的样本是一个Instance
        type(dataset[0])

        # In[10]:

        # 使用数字索引[a: b]，获取第a到第b个样本
        dataset[0: 3]

        # In[11]:

        # 索引也可以是负数
        dataset[-1]

        data_path = ['premise', 'hypothesis', 'label']

        # 读入文件
        with open(data_path[0]) as f:
            premise = f.readlines()

        with open(data_path[1]) as f:
            hypothesis = f.readlines()

        with open(data_path[2]) as f:
            label = f.readlines()

        assert len(premise) == len(hypothesis) and len(hypothesis) == len(label)

        # 组织DataSet
        data_set = DataSet()
        for p, h, l in zip(premise, hypothesis, label):
            p = p.strip()  # 将行末空格去除
            h = h.strip()  # 将行末空格去除
            data_set.append(Instance(premise=p, hypothesis=h, truth=l))

        data_set[0]

        # ### DataSet的其他操作
        # 在构建完毕DataSet后，仍然可以对DataSet的内容进行操作，函数接口为DataSet.apply()

        # In[13]:

        # 将premise域的所有文本转成小写
        data_set.apply(lambda x: x['premise'].lower(), new_field_name='premise')
        data_set[-2:]

        # In[14]:

        # label转int
        data_set.apply(lambda x: int(x['truth']), new_field_name='truth')
        data_set[-2:]

        # In[15]:

        # 使用空格分割句子
        def split_sent(ins):
            return ins['premise'].split()

        data_set.apply(split_sent, new_field_name='premise')
        data_set.apply(lambda x: x['hypothesis'].split(), new_field_name='hypothesis')
        data_set[-2:]

        # In[16]:

        # 筛选数据
        origin_data_set_len = len(data_set)
        data_set.drop(lambda x: len(x['premise']) <= 6)
        origin_data_set_len, len(data_set)

        # In[17]:

        # 增加长度信息
        data_set.apply(lambda x: [1] * len(x['premise']), new_field_name='premise_len')
        data_set.apply(lambda x: [1] * len(x['hypothesis']), new_field_name='hypothesis_len')
        data_set[-1]

        # In[18]:

        # 设定特征域、标签域
        data_set.set_input("premise", "premise_len", "hypothesis", "hypothesis_len")
        data_set.set_target("truth")

        # In[19]:

        # 重命名field
        data_set.rename_field('truth', 'label')
        data_set[-1]

        # In[20]:

        # 切分训练、验证集、测试集
        train_data, vad_data = data_set.split(0.5)
        dev_data, test_data = vad_data.split(0.4)
        len(train_data), len(dev_data), len(test_data)

        # In[21]:

        # 深拷贝一个数据集
        import copy
        train_data_2, dev_data_2 = copy.deepcopy(train_data), copy.deepcopy(dev_data)
        del copy

        # 初始化词表，该词表最大的vocab_size为10000，词表中每个词出现的最低频率为2，'<unk>'表示未知词语，'<pad>'表示padding词语
        # Vocabulary默认初始化参数为max_size=None, min_freq=None, unknown='<unk>', padding='<pad>'
        vocab = Vocabulary(max_size=10000, min_freq=2, unknown='<unk>', padding='<pad>')

        # 构建词表
        train_data.apply(lambda x: [vocab.add(word) for word in x['premise']])
        train_data.apply(lambda x: [vocab.add(word) for word in x['hypothesis']])
        vocab.build_vocab()

        # In[23]:

        # 根据词表index句子
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['premise']], new_field_name='premise')
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['hypothesis']], new_field_name='hypothesis')
        dev_data.apply(lambda x: [vocab.to_index(word) for word in x['premise']], new_field_name='premise')
        dev_data.apply(lambda x: [vocab.to_index(word) for word in x['hypothesis']], new_field_name='hypothesis')
        test_data.apply(lambda x: [vocab.to_index(word) for word in x['premise']], new_field_name='premise')
        test_data.apply(lambda x: [vocab.to_index(word) for word in x['hypothesis']], new_field_name='hypothesis')
        train_data[-1], dev_data[-1], test_data[-1]

        # 读入vocab文件
        with open('vocab.txt') as f:
            lines = f.readlines()
        vocabs = []
        for line in lines:
            vocabs.append(line.strip())

        # 实例化Vocabulary
        vocab_bert = Vocabulary(unknown=None, padding=None)
        # 将vocabs列表加入Vocabulary
        vocab_bert.add_word_lst(vocabs)
        # 构建词表
        vocab_bert.build_vocab()
        # 更新unknown与padding的token文本
        vocab_bert.unknown = '[UNK]'
        vocab_bert.padding = '[PAD]'

        # In[25]:

        # 根据词表index句子
        train_data_2.apply(lambda x: [vocab_bert.to_index(word) for word in x['premise']], new_field_name='premise')
        train_data_2.apply(lambda x: [vocab_bert.to_index(word) for word in x['hypothesis']],
                           new_field_name='hypothesis')
        dev_data_2.apply(lambda x: [vocab_bert.to_index(word) for word in x['premise']], new_field_name='premise')
        dev_data_2.apply(lambda x: [vocab_bert.to_index(word) for word in x['hypothesis']], new_field_name='hypothesis')
        train_data_2[-1], dev_data_2[-1]

        # step 1：加载模型参数（非必选）
        from fastNLP.io.config_io import ConfigSection, ConfigLoader
        args = ConfigSection()
        ConfigLoader().load_config("./data/config", {"esim_model": args})
        args["vocab_size"] = len(vocab)
        args.data

        # In[27]:

        # step 2：加载ESIM模型
        from fastNLP.models import ESIM
        model = ESIM(**args.data)
        model

        # In[28]:

        # 另一个例子：加载CNN文本分类模型
        from fastNLP.models import CNNText
        cnn_text_model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)
        cnn_text_model

        from fastNLP import CrossEntropyLoss
        from fastNLP import Adam
        from fastNLP import AccuracyMetric
        trainer = Trainer(
            train_data=train_data,
            model=model,
            loss=CrossEntropyLoss(pred='pred', target='label'),
            metrics=AccuracyMetric(),
            n_epochs=3,
            batch_size=16,
            print_every=-1,
            validate_every=-1,
            dev_data=dev_data,
            use_cuda=False,
            optimizer=Adam(lr=1e-3, weight_decay=0),
            check_code_level=-1,
            metric_key='acc',
            use_tqdm=False,
        )
        trainer.train()

        tester = Tester(
            data=test_data,
            model=model,
            metrics=AccuracyMetric(),
            batch_size=args["batch_size"],
        )
        tester.test()

        os.chdir("../..")
