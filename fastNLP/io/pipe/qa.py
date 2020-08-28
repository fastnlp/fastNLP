r"""
本文件中的Pipe主要用于处理问答任务的数据。

"""


from copy import deepcopy

from .pipe import Pipe
from .. import DataBundle
from ..loader.qa import CMRC2018Loader
from .utils import get_tokenizer
from ...core import DataSet
from ...core import Vocabulary

__all__ = ['CMRC2018BertPipe']


def _concat_clip(data_bundle, max_len, concat_field_name='raw_chars'):
    r"""
    处理data_bundle中的DataSet，将context与question按照character进行tokenize，然后使用[SEP]将两者连接起来。

    会新增field: context_len(int), raw_words(list[str]), target_start(int), target_end(int)其中target_start
    与target_end是与raw_chars等长的。其中target_start和target_end是前闭后闭的区间。

    :param DataBundle data_bundle: 类似["a", "b", "[SEP]", "c", ]
    :return:
    """
    tokenizer = get_tokenizer('cn-char', lang='cn')
    for name in list(data_bundle.datasets.keys()):
        ds = data_bundle.get_dataset(name)
        data_bundle.delete_dataset(name)
        new_ds = DataSet()
        for ins in ds:
            new_ins = deepcopy(ins)
            context = ins['context']
            question = ins['question']

            cnt_lst = tokenizer(context)
            q_lst = tokenizer(question)

            answer_start = -1

            if len(cnt_lst) + len(q_lst) + 3 > max_len:  # 预留开头的[CLS]和[SEP]和中间的[sep]
                if 'answer_starts' in ins and 'answers' in ins:
                    answer_start = int(ins['answer_starts'][0])
                    answer = ins['answers'][0]
                    answer_end = answer_start + len(answer)
                    if answer_end > max_len - 3 - len(q_lst):
                        span_start = answer_end + 3 + len(q_lst) - max_len
                        span_end = answer_end
                    else:
                        span_start = 0
                        span_end = max_len - 3 - len(q_lst)
                    cnt_lst = cnt_lst[span_start:span_end]
                    answer_start = int(ins['answer_starts'][0])
                    answer_start -= span_start
                    answer_end = answer_start + len(ins['answers'][0])
                else:
                    cnt_lst = cnt_lst[:max_len - len(q_lst) - 3]
            else:
                if 'answer_starts' in ins and 'answers' in ins:
                    answer_start = int(ins['answer_starts'][0])
                    answer_end = answer_start + len(ins['answers'][0])

            tokens = cnt_lst + ['[SEP]'] + q_lst
            new_ins['context_len'] = len(cnt_lst)
            new_ins[concat_field_name] = tokens

            if answer_start != -1:
                new_ins['target_start'] = answer_start
                new_ins['target_end'] = answer_end - 1

            new_ds.append(new_ins)
        data_bundle.set_dataset(new_ds, name)

    return data_bundle


class CMRC2018BertPipe(Pipe):
    r"""
    处理之后的DataSet将新增以下的field(传入的field仍然保留)

    .. csv-table::
        :header: "context_len", "raw_chars",  "target_start", "target_end", "chars"
        
        492, ['范', '廷', '颂... ], 30, 34, "[21, 25, ...]"
        491, ['范', '廷', '颂... ], 41, 61, "[21, 25, ...]"

        ".", "...", "...","...", "..."

    raw_words列是context与question拼起来的结果(连接的地方加入了[SEP])，words是转为index的值, target_start为答案start的index，target_end为答案end的index
    （闭区间）；context_len指示的是words列中context的长度。

    其中各列的meta信息如下:
    
    .. code::
    
        +-------------+-------------+-----------+--------------+------------+-------+---------+
        | field_names | context_len | raw_chars | target_start | target_end | chars | answers |
        +-------------+-------------+-----------+--------------+------------+-------+---------|
        |   is_input  |    False    |   False   |    False     |   False    |  True |  False  |
        |  is_target  |     True    |    True   |     True     |    True    | False |  True   |
        | ignore_type |    False    |    True   |    False     |   False    | False |  True   |
        |  pad_value  |      0      |     0     |      0       |     0      |   0   |   0     |
        +-------------+-------------+-----------+--------------+------------+-------+---------+
    
    """
    def __init__(self, max_len=510):
        super().__init__()
        self.max_len = max_len

    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        传入的DataSet应该具备以下的field

        .. csv-table::
           :header:"title", "context", "question", "answers", "answer_starts", "id"

           "范廷颂", "范廷颂枢机（，），圣名保禄·若瑟（）...", "范廷颂是什么时候被任为主教的？", ["1963年"], ["30"], "TRAIN_186_QUERY_0"
           "范廷颂", "范廷颂枢机（，），圣名保禄·若瑟（）...", "1990年，范廷颂担任什么职务？", ["1990年被擢升为天..."], ["41"],"TRAIN_186_QUERY_1"
           "...", "...", "...","...", ".", "..."

        :param data_bundle:
        :return:
        """
        data_bundle = _concat_clip(data_bundle, max_len=self.max_len, concat_field_name='raw_chars')

        src_vocab = Vocabulary()
        src_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                               field_name='raw_chars',
                               no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                        if 'train' not in name]
                               )
        src_vocab.index_dataset(*data_bundle.datasets.values(), field_name='raw_chars', new_field_name='chars')
        data_bundle.set_vocab(src_vocab, 'chars')

        data_bundle.set_ignore_type('raw_chars', 'answers', flag=True)
        data_bundle.set_input('chars')
        data_bundle.set_target('raw_chars', 'answers', 'target_start', 'target_end', 'context_len')

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = CMRC2018Loader().load(paths)
        return self.process(data_bundle)