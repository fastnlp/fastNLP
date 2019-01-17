import sys
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import DataSet

# 问题：fastNLP虽然已经提供了split函数，可以将数据集划分成训练集和测试机，但一般网上用作训练的标准集都已经提前划分好了训练集和测试机，
# 而使用split将数据集进行随机划分还引来了一个问题：
#       因为每次都是随机划分，导致每次的字典都不一样，保存好模型下次再载入进行测试时，因为字典不同导致结果差异非常大。
#
# 解决方法：在Vocabulary增加一个字典保存函数和一个字典读取函数，而不是每次都生成一个新字典，同时减少下次运行的成本，第一次使用save_vocab()
# 生成字典后，下次可以直接使用load_vocab()载入的字典。
if __name__ == '__main__':

    data_path = "data_for_tests/tutorial_sample_dataset.csv"

    train_data = DataSet.read_csv(data_path, headers=('raw_sentence', 'label'), sep='\t')
    print('len(train_data)', train_data)

    # 将所有字母转为小写
    train_data.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')

    # label转int
    train_data.apply(lambda x: int(x['label']) - 1, new_field_name='label_seq', is_target=True)


    # 使用空格分割句子
    def split_sent(ins):
        return ins['raw_sentence'].split()


    train_data.apply(split_sent, new_field_name='words')

    # 增加长度信息
    train_data.apply(lambda x: len(x['words']), new_field_name='seq_len')

    # 筛选数据
    train_data.drop(lambda x: x['seq_len'] <= 3)

    # set input，模型forward时使用
    train_data.set_input("words")

    # 构建词表, Vocabulary.add(word)
    vocab = Vocabulary(min_freq=2, write_path='data_for_tests')
    train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
    vocab.build_vocab()
    vocab.save_vocab()
    voc = vocab.load_vocab()
    print('载入的字典为：')
    print(voc)



