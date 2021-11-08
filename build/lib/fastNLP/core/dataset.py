r"""
:class:`~fastNLP.core.dataset.DataSet` 是fastNLP中用于承载数据的容器。可以将DataSet看做是一个表格，
每一行是一个sample (在fastNLP中被称为 :mod:`~fastNLP.core.instance` )，
每一列是一个feature (在fastNLP中称为 :mod:`~fastNLP.core.field` )。

.. csv-table:: Following is a demo layout of DataSet
   :header: "sentence", "words", "seq_len"

   "This is the first instance .", "[This, is, the, first, instance, .]", 6
   "Second instance .", "[Second, instance, .]", 3
   "Third instance .", "[Third, instance, .]", 3
   "...", "[...]", "..."

在fastNLP内部每一行是一个 :class:`~fastNLP.Instance` 对象； 每一列是一个 :class:`~fastNLP.FieldArray` 对象。

----------------------------
1.DataSet的创建
----------------------------

创建DataSet主要有以下的3种方式

1.1 传入dict
----------------------------

    .. code-block::

        from fastNLP import DataSet
        data = {'sentence':["This is the first instance .", "Second instance .", "Third instance ."],
                'words': [['this', 'is', 'the', 'first', 'instance', '.'], ['Second', 'instance', '.'], ['Third', 'instance', '.'],
                'seq_len': [6, 3, 3]}
        dataset = DataSet(data)
        # 传入的dict的每个key的value应该为具有相同长度的list

1.2 通过 Instance 构建
----------------------------

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        instance = Instance(sentence="This is the first instance",
                            words=['this', 'is', 'the', 'first', 'instance', '.'],
                            seq_len=6)
        dataset.append(instance)
        # 可以继续append更多内容，但是append的instance应该和第一个instance拥有完全相同的field

1.3 通过 List[Instance] 构建
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import Instance
        instances = []
        winstances.append(Instance(sentence="This is the first instance",
                            ords=['this', 'is', 'the', 'first', 'instance', '.'],
                            seq_len=6))
        instances.append(Instance(sentence="Second instance .",
                            words=['Second', 'instance', '.'],
                            seq_len=3))
        dataset = DataSet(instances)
        
--------------------------------------
2.DataSet与预处理
--------------------------------------

常见的预处理有如下几种

2.1 从某个文本文件读取内容
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        filepath='some/text/file'
        # 假设文件中每行内容如下(sentence  label):
        #    This is a fantastic day    positive
        #    The bad weather    negative
        #    .....
        with open(filepath, 'r') as f:
            for line in f:
                sent, label = line.strip().split('\t')
                dataset.append(Instance(sentence=sent, label=label))

    .. note::
        直接读取特定数据集的数据请参考  :doc:`/tutorials/tutorial_4_load_dataset`

2.2 对DataSet中的内容处理
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        data = {'sentence':["This is the first instance .", "Second instance .", "Third instance ."]}
        dataset = DataSet(data)
        # 将句子分成单词形式, 详见DataSet.apply()方法
        dataset.apply(lambda ins: ins['sentence'].split(), new_field_name='words')
        # 或使用DataSet.apply_field()
        dataset.apply_field(lambda sent:sent.split(), field_name='sentence', new_field_name='words')
        # 除了匿名函数，也可以定义函数传递进去
        def get_words(instance):
            sentence = instance['sentence']
            words = sentence.split()
            return words
        dataset.apply(get_words, new_field_name='words')

2.3 删除DataSet的内容
--------------------------------------

    .. code-block::

        from fastNLP import DataSet
        dataset = DataSet({'a': list(range(-5, 5))})
        # 返回满足条件的instance,并放入DataSet中
        dropped_dataset = dataset.drop(lambda ins:ins['a']<0, inplace=False)
        # 在dataset中删除满足条件的instance
        dataset.drop(lambda ins:ins['a']<0)  # dataset的instance数量减少
        #  删除第3个instance
        dataset.delete_instance(2)
        #  删除名为'a'的field
        dataset.delete_field('a')


2.4 遍历DataSet的内容
--------------------------------------

    .. code-block::

        for instance in dataset:
            # do something

2.5 一些其它操作
--------------------------------------

    .. code-block::

        #  检查是否存在名为'a'的field
        dataset.has_field('a')  # 或 ('a' in dataset)
        #  将名为'a'的field改名为'b'
        dataset.rename_field('a', 'b')
        #  DataSet的长度
        len(dataset)
        
--------------------------------------
3.DataSet与自然语言处理(NLP)
--------------------------------------

在目前深度学习的模型中，大都依赖于随机梯度下降法(SGD)进行模型的优化。随机梯度下降需要将数据切分成一个个的 batch，
一个batch进行一次前向计算(forward)与梯度后向传播(backward)。在自然语言处理的场景下，往往还需要对数据进行pad。这是
由于句子的长度一般是不同的，但是一次batch中的每个field都必须是一个tensor，所以需要将所有句子都补齐到相同的长度。

3.1 DataSet与DataSetIter
--------------------------------------

    我们先看fastNLP中如何将数据分成一个一个的batch的例子, 这里我们使用随机生成的数据来模拟一个二分类文本分类任务，
    words和characters是输入，labels是文本类别

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import DataSetIter
        from fastNLP import SequentialSampler
        from fastNLP import EngChar2DPadder

        num_instances = 100
        # 假设每句话最少2个词，最多5个词; 词表的大小是100个; 一共26个字母，每个单词最短1个字母，最长5个字母
        lengths = [random.randint(2, 5) for _ in range(num_instances)]
        data = {'words': [[random.randint(1, 100) for _ in range(lengths[idx]) ] for idx in range(num_instances)],
                'chars': [
                            [[random.randint(1, 27) for _ in range(random.randint(1, 5))]
                            for _ in range(lengths[idx])]
                     for idx in range(num_instances)],
                'label': [random.randint(0, 1) for _ in range(num_instances)]}

        d = DataSet(data)
        d.set_padder('chars', EngChar2DPadder())  # 因为英文character的pad方式与word的pad方式不一样

        d.set_target('label')
        d.set_input('words', 'chars')

        for batch_x, batch_y in DataSetIter(d, sampler=SequentialSampler(), batch_size=2):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            break
            # 输出为
            # {'words': tensor([[49, 27, 20, 36, 63],
            #     [53, 82, 23, 11,  0]]), 'chars': tensor([[[13,  3, 14, 25,  1],
            #      [ 8, 20, 12,  0,  0],
            #      [27,  8,  0,  0,  0],
            #      [ 1, 15, 26,  0,  0],
            #      [11, 24, 17,  0,  0]],
            #
            #     [[ 6, 14, 11, 27, 22],
            #      [18,  6,  4, 19,  0],
            #      [19, 22,  9,  0,  0],
            #      [10, 25,  0,  0,  0],
            #      [ 0,  0,  0,  0,  0]]])}
            # {'label': tensor([0, 0])}

    其中 :class:`~fastNLP.DataSetIter` 是用于从DataSet中按照batch_size为大小取出batch的迭代器，
    :class:`~fastNLP.SequentialSampler` 用于指示 :class:`~fastNLP.DataSetIter` 以怎样的
    顺序从DataSet中取出instance以组成一个batch，
    更详细的说明请参照 :class:`~fastNLP.DataSetIter` 和 :class:`~fastNLP.SequentialSampler` 文档。

    通过 ``DataSet.set_input('words', 'chars')`` , fastNLP将认为 `words` 和 `chars` 这两个field都是input，并将它们都放入迭代器
    生成的第一个dict中; ``DataSet.set_target('labels')`` , fastNLP将认为 `labels` 这个field是target，并将其放入到迭代器的第
    二个dict中。如上例中所打印结果。分为input和target的原因是由于它们在被 :class:`~fastNLP.Trainer` 所使用时会有所差异，
    详见  :class:`~fastNLP.Trainer`

    当把某个field设置为 `target` 或者 `input` 的时候(两者不是互斥的，可以同时设为两种)，fastNLP不仅仅只是将其放
    置到不同的dict中，而还会对被设置为 `input` 或 `target` 的 field 进行类型检查。类型检查的目的是为了看能否把该 field 转为
    pytorch的 :class:`torch.LongTensor` 或 :class:`torch.FloatTensor` 类型
    (也可以在 :class:`~fastNLP.DataSetIter` 中设置输出numpy类型，参考 :class:`~fastNLP.DataSetIter` )。
    
    如上例所示，fastNLP已将 `words` ，`chars` 和 `label` 转为了 :class:`Tensor` 类型。
    如果 field 在每个 `instance` 都拥有相同的维度(不能超过两维)，且最内层的元素都为相同的 type(int, float, np.int*, np.float*)，
    则fastNLP默认将对该 field 进行pad。也支持全为str的field作为target和input，这种情况下，fastNLP默认不进行pad。
    另外，当某个 field 已经被设置为了 target 或者 input 后，之后 `append` 的
    `instance` 对应的 field 必须要和前面已有的内容一致，否则会报错。

    可以查看field的dtype::
        
        from fastNLP import DataSet

        d = DataSet({'a': [0, 1, 3], 'b':[[1.0, 2.0], [0.1, 0.2], [3]]})
        d.set_input('a', 'b')
        d.a.dtype
        >> numpy.int64
        d.b.dtype
        >> numpy.float64
        # 默认情况下'a'这个field将被转换为torch.LongTensor，但如果需要其为torch.FloatTensor可以手动修改dtype
        d.a.dtype = float  #  请确保该field的确可以全部转换为float。

    如果某个field中出现了多种类型混合(比如一部分为str，一部分为int)的情况，fastNLP无法判断该field的类型，会报如下的
    错误::

        from fastNLP import DataSet
        
        d = DataSet({'data': [1, 'a']})
        d.set_input('data')
        >> RuntimeError: Mixed data types in Field data: [<class 'str'>, <class 'int'>]

    可以通过设置以忽略对该field进行类型检查::

        from fastNLP import DataSet
        d = DataSet({'data': [1, 'a']})
        d.set_ignore_type('data')
        d.set_input('data')

    当某个field被设置为忽略type之后，fastNLP将不对其进行pad。

3.2 DataSet与pad
--------------------------------------

    在fastNLP里，pad是与一个field绑定的。即不同的field可以使用不同的pad方式，比如在英文任务中word需要的pad和
    character的pad方式往往是不同的。fastNLP是通过一个叫做 :class:`~fastNLP.Padder` 的子类来完成的。
    默认情况下，所有field使用 :class:`~fastNLP.AutoPadder`
    。可以通过使用以下方式设置Padder(如果将padder设置为None，则该field不会进行pad操作)。
    大多数情况下直接使用 :class:`~fastNLP.AutoPadder` 就可以了。
    如果 :class:`~fastNLP.AutoPadder` 或 :class:`~fastNLP.EngChar2DPadder` 无法满足需求，
    也可以自己写一个 :class:`~fastNLP.Padder` 。

    .. code-block::

        from fastNLP import DataSet
        from fastNLP import EngChar2DPadder
        import random
        dataset = DataSet()
        max_chars, max_words, sent_num = 5, 10, 20
        contents = [[
                        [random.randint(1, 27) for _ in range(random.randint(1, max_chars))]
                            for _ in range(random.randint(1, max_words))
                    ]  for _ in range(sent_num)]
        #  初始化时传入
        dataset.add_field('chars', contents, padder=EngChar2DPadder())
        #  直接设置
        dataset.set_padder('chars', EngChar2DPadder())
        #  也可以设置pad的value
        dataset.set_pad_val('chars', -1)

3.3 根据DataSet中多个field合成新的field
------------------------------------------------------------

    DataSet支持在进行batch时，默认只能看到当前的field的值，但在某些训练中可能存在以下的情况: (1)需要两个field拼接成为一个field;
    (2)需要在batch中进行负采样。这时候就需要能够同时利用多个field进行batch的操作，DataSet中的add_collate_fn()函数支持添加
    自定义涉及多个field的collate_fn函数。例如下例中将两个field拼接成一个field的场景

    .. code-block::

        from fastNLP import DataSet, DataSetIter
        import torch

        data = DataSet({
            'x1': [[0, 1],
                   [2]],
            'x2': [[3],
                   [2, 4, 5]],
            'y': [0, 1]
        })
        data.set_target('y')

        # 所有的collate_fn函数都接受list[(ind1, instance1), (ind2, instance2), ...]作为输入，其中ind1/ind2是该instance在dataset中
        #   的index，instance1/instance2是这次batch取出来的数据，包含了所有的field.
        def concat_collate_fn(ins_list):
            x1 = [ins['x1'] for ind,ins in ins_list]
            x2 = [ins['x2'] for ind,ins in ins_list]
            xs = []
            for i in range(len(ins_list)):
                xs.append(torch.LongTensor(x1[i] + x2[i]))
            # 需要自行pad并转换为tensor，但不需要移动到gpu
            arr = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
            b_x = {'x': arr}
            b_y = {}
            # 返回值一定是两个dict，第一个dict的值会认为是input，第二个dict的值会认为是target. 若名称与已有input或target重复，则
            #   采用返回值。
            return b_x, b_y

        data.add_collate_fn(concat_collate_fn)

        for batch_x, batch_y in DataSetIter(data, sampler=SequentialSampler(), batch_size=2):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            # batch_x: {'x': tensor([[0, 1, 3, 0],
            #                        [2, 2, 4, 5]])}
            # batch_y: {'y': array([0, 1])}

        # 如果取batch过程含有一些参数，可以通过类来实现
        class ConCollateFn:
            def __init__(self, max_len=3):
                self.max_len = max_len

            def __call__(self, ins_list):  # 实现该类的__call__函数
                x1 = [ins['x1'] for ind, ins in ins_list]
                x2 = [ins['x2'] for ind, ins in ins_list]
                xs = []
                for i in range(len(ins_list)):
                    xs.append(torch.LongTensor(x1[i] + x2[i])[:self.max_len])
                arr = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
                b_x = {'x': arr}
                b_y = {}
                return b_x, b_y
        data.delete_collate_fn()  # 删除之前的collate_fn
        data.add_collate_fn(ConCollateFn(max_len=3))
        for batch_x, batch_y in DataSetIter(data, sampler=SequentialSampler(), batch_size=2):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            # batch_x: {'x': tensor([[0, 1, 3],
            #                        [2, 2, 4]])}
            # batch_y: {'y': array([0, 1])}

"""
__all__ = [
    "DataSet",
]

import _pickle as pickle
from copy import deepcopy

import numpy as np
from prettytable import PrettyTable

from ._logger import logger
from .const import Const
from .field import AppendToTargetOrInputException
from .field import AutoPadder
from .field import FieldArray
from .field import SetInputOrTargetException
from .instance import Instance
from .utils import pretty_table_printer
from .collate_fn import Collater
try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm


class ApplyResultException(Exception):
    def __init__(self, msg, index=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
    
class DataSet(object):
    r"""
    fastNLP的数据容器，详细的使用方法见文档  :mod:`fastNLP.core.dataset`
    """

    def __init__(self, data=None):
        r"""
        
        :param data: 如果为dict类型，则每个key的value应该为等长的list; 如果为list，
            每个元素应该为具有相同field的 :class:`~fastNLP.Instance` 。
        """
        self.field_arrays = {}
        if data is not None:
            if isinstance(data, dict):
                length_set = set()
                for key, value in data.items():
                    length_set.add(len(value))
                assert len(length_set) == 1, "Arrays must all be same length."
                for key, value in data.items():
                    self.add_field(field_name=key, fields=value)
            elif isinstance(data, list):
                for ins in data:
                    assert isinstance(ins, Instance), "Must be Instance type, not {}.".format(type(ins))
                    self.append(ins)

            else:
                raise ValueError("data only be dict or list type.")
        self._collater = Collater()

    @property
    def collater(self):
        if self._collater is None:
            self._collater = Collater()
        return self._collater

    @collater.setter
    def collater(self, value):
        assert isinstance(value, Collater)
        self._collater = value

    def __contains__(self, item):
        return item in self.field_arrays

    def __iter__(self):
        def iter_func():
            for idx in range(len(self)):
                yield self[idx]

        return iter_func()

    def _inner_iter(self):
        class Iter_ptr:
            def __init__(self, dataset, idx):
                self.dataset = dataset
                self.idx = idx

            def __getitem__(self, item):
                assert item in self.dataset.field_arrays, "no such field:{} in Instance {}".format(item, self.dataset[
                    self.idx])
                assert self.idx < len(self.dataset.field_arrays[item]), "index:{} out of range".format(self.idx)
                return self.dataset.field_arrays[item][self.idx]

            def __setitem__(self, key, value):
                raise TypeError("You cannot modify value directly.")

            def items(self):
                ins = self.dataset[self.idx]
                return ins.items()

            def __repr__(self):
                return self.dataset[self.idx].__repr__()

        def inner_iter_func():
            for idx in range(len(self)):
                yield Iter_ptr(self, idx)

        return inner_iter_func()

    def __getitem__(self, idx):
        r"""给定int的index，返回一个Instance; 给定slice，返回包含这个slice内容的新的DataSet。

        :param idx: can be int or slice.
        :return: If `idx` is int, return an Instance object.
                If `idx` is slice, return a DataSet object.
        """
        if isinstance(idx, int):
            return Instance(**{name: self.field_arrays[name][idx] for name in self.field_arrays})
        elif isinstance(idx, slice):
            if idx.start is not None and (idx.start >= len(self) or idx.start <= -len(self)):
                raise RuntimeError(f"Start index {idx.start} out of range 0-{len(self) - 1}")
            data_set = DataSet()
            for field_name, field in self.field_arrays.items():
                data_set.add_field(field_name=field_name, fields=field.content[idx], padder=field.padder,
                                   is_input=field.is_input, is_target=field.is_target, ignore_type=field.ignore_type)
            data_set.collater = self.collater.copy_from(self.collater)
            return data_set
        elif isinstance(idx, str):
            if idx not in self:
                raise KeyError("No such field called {} in DataSet.".format(idx))
            return self.field_arrays[idx]
        elif isinstance(idx, list):
            dataset = DataSet()
            for i in idx:
                assert isinstance(i, int), "Only int index allowed."
                instance = self[i]
                dataset.append(instance)
            for field_name, field in self.field_arrays.items():
                dataset.field_arrays[field_name].to(field)
            dataset.collater = self.collater.copy_from(self.collater)
            return dataset
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __getattr__(self, item):
        # Not tested. Don't use !!
        if item == "field_arrays":
            raise AttributeError
        if isinstance(item, str) and item in self.field_arrays:
            return self.field_arrays[item]

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __len__(self):
        r"""Fetch the length of the dataset.

        :return length:
        """
        if len(self.field_arrays) == 0:
            return 0
        field = iter(self.field_arrays.values()).__next__()
        return len(field)

    def __repr__(self):
        return str(pretty_table_printer(self))

    def print_field_meta(self):
        r"""
        输出当前field的meta信息, 形似下列的输出::

            +-------------+-------+-------+
            | field_names |   x   |   y   |
            +=============+=======+=======+
            |   is_input  |  True | False |
            |  is_target  | False | False |
            | ignore_type | False |       |
            |  pad_value  |   0   |       |
            +-------------+-------+-------+

        str field_names: DataSet中field的名称
        bool is_input: field是否为input
        bool is_target: field是否为target
        bool ignore_type: 是否忽略该field的type, 一般仅在该field至少为input或target时才有意义
        int pad_value: 该field的pad的值，仅在该field为input或target时有意义
        :return:
        """
        if len(self.field_arrays)>0:
            field_names = ['field_names']
            is_inputs = ['is_input']
            is_targets = ['is_target']
            pad_values = ['pad_value']
            ignore_types = ['ignore_type']

            for name, field_array in self.field_arrays.items():
                field_names.append(name)
                if field_array.is_input:
                    is_inputs.append(True)
                else:
                    is_inputs.append(False)
                if field_array.is_target:
                    is_targets.append(True)
                else:
                    is_targets.append(False)

                if (field_array.is_input or field_array.is_target) and field_array.padder is not None:
                    pad_values.append(field_array.padder.get_pad_val())
                else:
                    pad_values.append(' ')

                if field_array._ignore_type:
                    ignore_types.append(True)
                elif field_array.is_input or field_array.is_target:
                    ignore_types.append(False)
                else:
                    ignore_types.append(' ')
            table = PrettyTable(field_names=field_names)
            fields = [is_inputs, is_targets, ignore_types, pad_values]
            for field in fields:
                table.add_row(field)
            logger.info(table)
            return table

    def append(self, instance):
        r"""
        将一个instance对象append到DataSet后面。

        :param ~fastNLP.Instance instance: 若DataSet不为空，则instance应该拥有和DataSet完全一样的field。

        """
        if len(self.field_arrays) == 0:
            # DataSet has no field yet
            for name, field in instance.fields.items():
                # field = field.tolist() if isinstance(field, np.ndarray) else field
                self.field_arrays[name] = FieldArray(name, [field])  # 第一个样本，必须用list包装起来
        else:
            if len(self.field_arrays) != len(instance.fields):
                raise ValueError(
                    "DataSet object has {} fields, but attempt to append an Instance object with {} fields."
                        .format(len(self.field_arrays), len(instance.fields)))
            for name, field in instance.fields.items():
                assert name in self.field_arrays
                try:
                    self.field_arrays[name].append(field)
                except AppendToTargetOrInputException as e:
                    logger.error(f"Cannot append to field:{name}.")
                    raise e

    def add_fieldarray(self, field_name, fieldarray):
        r"""
        将fieldarray添加到DataSet中.

        :param str field_name: 新加入的field的名称
        :param ~fastNLP.core.FieldArray fieldarray: 需要加入DataSet的field的内容
        :return:
        """
        if not isinstance(fieldarray, FieldArray):
            raise TypeError("Only fastNLP.FieldArray supported.")
        if len(self) != len(fieldarray):
            raise RuntimeError(f"The field to add must have the same size as dataset. "
                               f"Dataset size {len(self)} != field size {len(fieldarray)}")
        fieldarray.name = field_name
        self.field_arrays[field_name] = fieldarray

    def add_field(self, field_name, fields, padder=AutoPadder(), is_input=False, is_target=False, ignore_type=False):
        r"""
        新增一个field
        
        :param str field_name: 新增的field的名称
        :param list fields: 需要新增的field的内容
        :param None,~fastNLP.Padder padder: 如果为None,则不进行pad，默认使用 :class:`~fastNLP.AutoPadder` 自动判断是否需要做pad。
        :param bool is_input: 新加入的field是否是input
        :param bool is_target: 新加入的field是否是target
        :param bool ignore_type: 是否忽略对新加入的field的类型检查
        """

        if len(self.field_arrays) != 0:
            if len(self) != len(fields):
                raise RuntimeError(f"The field to add must have the same size as dataset. "
                                   f"Dataset size {len(self)} != field size {len(fields)}")
        self.field_arrays[field_name] = FieldArray(field_name, fields, is_target=is_target, is_input=is_input,
                                                   padder=padder, ignore_type=ignore_type)

    def delete_instance(self, index):
        r"""
        删除第index个instance

        :param int index: 需要删除的instance的index，序号从0开始。
        """
        assert isinstance(index, int), "Only integer supported."
        if len(self) <= index:
            raise IndexError("{} is too large for as DataSet with {} instances.".format(index, len(self)))
        if len(self) == 1:
            self.field_arrays.clear()
        else:
            for field in self.field_arrays.values():
                field.pop(index)
        return self

    def delete_field(self, field_name):
        r"""
        删除名为field_name的field

        :param str field_name: 需要删除的field的名称.
        """
        self.field_arrays.pop(field_name)
        return self

    def copy_field(self, field_name, new_field_name):
        r"""
        深度copy名为field_name的field到new_field_name

        :param str field_name: 需要copy的field。
        :param str new_field_name: copy生成的field名称
        :return: self
        """
        if not self.has_field(field_name):
            raise KeyError(f"Field:{field_name} not found in DataSet.")
        fieldarray = deepcopy(self.get_field(field_name))
        fieldarray.name = new_field_name
        self.add_fieldarray(field_name=new_field_name, fieldarray=fieldarray)
        return self

    def has_field(self, field_name):
        r"""
        判断DataSet中是否有名为field_name这个field

        :param str field_name: field的名称
        :return bool: 表示是否有名为field_name这个field
        """
        if isinstance(field_name, str):
            return field_name in self.field_arrays
        return False

    def get_field(self, field_name):
        r"""
        获取field_name这个field

        :param str field_name: field的名称
        :return: :class:`~fastNLP.FieldArray`
        """
        if field_name not in self.field_arrays:
            raise KeyError("Field name {} not found in DataSet".format(field_name))
        return self.field_arrays[field_name]

    def get_all_fields(self):
        r"""
        返回一个dict，key为field_name, value为对应的 :class:`~fastNLP.FieldArray`

        :return dict: 返回如上所述的字典
        """
        return self.field_arrays

    def get_field_names(self) -> list:
        r"""
        返回一个list，包含所有 field 的名字

        :return list: 返回如上所述的列表
        """
        return sorted(self.field_arrays.keys())

    def get_length(self):
        r"""
        获取DataSet的元素数量

        :return: int: DataSet中Instance的个数。
        """
        return len(self)

    def rename_field(self, field_name, new_field_name):
        r"""
        将某个field重新命名.

        :param str field_name: 原来的field名称。
        :param str new_field_name: 修改为new_name。
        """
        if field_name in self.field_arrays:
            self.field_arrays[new_field_name] = self.field_arrays.pop(field_name)
            self.field_arrays[new_field_name].name = new_field_name
        else:
            raise KeyError("DataSet has no field named {}.".format(field_name))
        return self

    def set_target(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True):
        r"""
        将field_names的field设置为target

        Example::

            dataset.set_target('labels', 'seq_len')  # 将labels和seq_len这两个field的target属性设置为True
            dataset.set_target('labels', 'seq_lens', flag=False) # 将labels和seq_len的target属性设置为False

        :param str field_names: field的名称
        :param bool flag: 将field_name的target状态设置为flag
        :param bool use_1st_ins_infer_dim_type: 如果为True，将不会check该列是否所有数据都是同样的维度，同样的类型。将直接使用第一
            行的数据进行类型和维度推断本列的数据的类型和维度。
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                try:
                    self.field_arrays[name]._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
                    self.field_arrays[name].is_target = flag
                except SetInputOrTargetException as e:
                    logger.error(f"Cannot set field:{name} as target.")
                    raise e
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_input(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True):
        r"""
        将field_names的field设置为input::

            dataset.set_input('words', 'seq_len')   # 将words和seq_len这两个field的input属性设置为True
            dataset.set_input('words', flag=False)  # 将words这个field的input属性设置为False

        :param str field_names: field的名称
        :param bool flag: 将field_name的input状态设置为flag
        :param bool use_1st_ins_infer_dim_type: 如果为True，将不会check该列是否所有数据都是同样的维度，同样的类型。将直接使用第一
            行的数据进行类型和维度推断本列的数据的类型和维度。
        """
        for name in field_names:
            if name in self.field_arrays:
                try:
                    self.field_arrays[name]._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
                    self.field_arrays[name].is_input = flag
                except SetInputOrTargetException as e:
                    logger.error(f"Cannot set field:{name} as input, exception happens at the {e.index} value.")
                    raise e
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_ignore_type(self, *field_names, flag=True):
        r"""
        将field设置为忽略类型状态。当某个field被设置了ignore_type, 则在被设置为target或者input时将不进行类型检查，
        默认情况下也不进行pad。如果仍需要pad该field，可通过自定义Padder实现，若该field需要转换为tensor，需要在padder
        中转换，但不需要在padder中移动到gpu。

        :param str field_names: field的名称
        :param bool flag: 将field_name的ignore_type状态设置为flag
        :return:
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].ignore_type = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_padder(self, field_name, padder):
        r"""
        为field_name设置padder::

            from fastNLP import EngChar2DPadder
            padder = EngChar2DPadder()
            dataset.set_padder('chars', padder)  # 则chars这个field会使用EngChar2DPadder进行pad操作

        :param str field_name: 设置field的padding方式为padder
        :param None,~fastNLP.Padder padder: 设置为None即删除padder, 即对该field不进行pad操作。
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_padder(padder)
        return self

    def set_pad_val(self, field_name, pad_val):
        r"""
        为某个field设置对应的pad_val.

        :param str field_name: 修改该field的pad_val
        :param int pad_val: 该field的padder会以pad_val作为padding index
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_pad_val(pad_val)
        return self

    def get_input_name(self):
        r"""
        返回所有is_input被设置为True的field名称

        :return list: 里面的元素为被设置为input的field名称
        """
        return [name for name, field in self.field_arrays.items() if field.is_input]

    def get_target_name(self):
        r"""
        返回所有is_target被设置为True的field名称

        :return list: 里面的元素为被设置为target的field名称
        """
        return [name for name, field in self.field_arrays.items() if field.is_target]

    def apply_field(self, func, field_name, new_field_name=None, **kwargs):
        r"""
        将DataSet中的每个instance中的名为 `field_name` 的field传给func，并获取它的返回值。

        :param callable func: input是instance中名为 `field_name` 的field的内容。
        :param str field_name: 传入func的是哪个field。
        :param None,str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将名为 `new_field_name` 的field设置为input

            2. is_target: bool, 如果为True则将名为 `new_field_name` 的field设置为target

            3. ignore_type: bool, 如果为True则将名为 `new_field_name` 的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return List[Any]:   里面的元素为func的返回值，所以list长度为DataSet的长度
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        return self.apply(func, new_field_name, _apply_field=field_name, **kwargs)

    def apply_field_more(self, func, field_name, modify_fields=True, **kwargs):
        r"""
        将 ``DataSet`` 中的每个 ``Instance`` 中的名为 `field_name` 的field 传给 func，并获取它的返回值。
        func 可以返回一个或多个 field 上的结果。
        
        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :meth:`~fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。
            
        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param str field_name: 传入func的是哪个field。
        :param bool modify_fields: 是否用结果修改 `DataSet` 中的 `Field`， 默认为 True
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将被修改的field设置为input

            2. is_target: bool, 如果为True则将被修改的field设置为target

            3. ignore_type: bool, 如果为True则将被修改的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return Dict[str:Field]: 返回一个字典
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        return self.apply_more(func, modify_fields, _apply_field=field_name, **kwargs)
    
    def _add_apply_field(self, results, new_field_name, kwargs):
        r"""
        将results作为加入到新的field中，field名称为new_field_name

        :param List[str] results: 一般是apply*()之后的结果
        :param str new_field_name: 新加入的field的名称
        :param dict kwargs: 用户apply*()时传入的自定义参数
        :return:
        """
        extra_param = {}
        if 'is_input' in kwargs:
            extra_param['is_input'] = kwargs['is_input']
        if 'is_target' in kwargs:
            extra_param['is_target'] = kwargs['is_target']
        if 'ignore_type' in kwargs:
            extra_param['ignore_type'] = kwargs['ignore_type']
        if new_field_name in self.field_arrays:
            # overwrite the field, keep same attributes
            old_field = self.field_arrays[new_field_name]
            if 'is_input' not in extra_param:
                extra_param['is_input'] = old_field.is_input
            if 'is_target' not in extra_param:
                extra_param['is_target'] = old_field.is_target
            if 'ignore_type' not in extra_param:
                extra_param['ignore_type'] = old_field.ignore_type
            self.add_field(field_name=new_field_name, fields=results, is_input=extra_param["is_input"],
                           is_target=extra_param["is_target"], ignore_type=extra_param['ignore_type'],
                           padder=self.get_field(new_field_name).padder)
        else:
            self.add_field(field_name=new_field_name, fields=results, is_input=extra_param.get("is_input", None),
                           is_target=extra_param.get("is_target", None),
                           ignore_type=extra_param.get("ignore_type", False))

    def apply_more(self, func, modify_fields=True, **kwargs):
        r"""
        将 ``DataSet`` 中每个 ``Instance`` 传入到func中，并获取它的返回值。func可以返回一个或多个 field 上的结果。
        
        .. note::
            ``apply_more`` 与 ``apply`` 的区别：
            
            1. ``apply_more`` 可以返回多个 field 的结果， ``apply`` 只可以返回一个field 的结果；
            
            2. ``apply_more`` 的返回值是一个字典，每个 key-value 对中的 key 表示 field 的名字，value 表示计算结果；
            
            3. ``apply_more`` 默认修改 ``DataSet`` 中的 field ，``apply`` 默认不修改。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param bool modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 True
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将被修改的的field设置为input

            2. is_target: bool, 如果为True则将被修改的的field设置为target

            3. ignore_type: bool, 如果为True则将被修改的的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return Dict[str:Field]: 返回一个字典
        """
        # 返回 dict , 检查是否一直相同
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        idx = -1
        try:
            results = {}
            for idx, ins in tqdm(enumerate(self._inner_iter()), total=len(self), dynamic_ncols=True,
                                 desc=kwargs.get('tqdm_desc', ''),
                                 leave=False, disable=not kwargs.get('use_tqdm', False)):
                if "_apply_field" in kwargs:
                    res = func(ins[kwargs["_apply_field"]])
                else:
                    res = func(ins)
                if not isinstance(res, dict):
                    raise ApplyResultException("The result of func is not a dict", idx)
                if idx == 0:
                    for key, value in res.items():
                        results[key] = [value]
                else:
                    for key, value in res.items():
                        if key not in results:
                            raise ApplyResultException("apply results have different fields", idx)
                        results[key].append(value)
                    if len(res) != len(results):
                        raise ApplyResultException("apply results have different fields", idx)
        except Exception as e:
            if idx != -1:
                if isinstance(e, ApplyResultException):
                    logger.error(e.msg)
                logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e
    
        if modify_fields is True:
            for field, result in results.items():
                self._add_apply_field(result, field, kwargs)
    
        return results

    def apply(self, func, new_field_name=None, **kwargs):
        r"""
        将DataSet中每个instance传入到func中，并获取它的返回值.

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance``
        :param None,str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将 `new_field_name` 的field设置为input

            2. is_target: bool, 如果为True则将 `new_field_name` 的field设置为target

            3. ignore_type: bool, 如果为True则将 `new_field_name` 的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称
            
        :return List[Any]: 里面的元素为func的返回值，所以list长度为DataSet的长度
        """
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        idx = -1
        try:
            results = []
            for idx, ins in tqdm(enumerate(self._inner_iter()), total=len(self), dynamic_ncols=True, leave=False,
                                 desc=kwargs.get('tqdm_desc', ''),
                                 disable=not kwargs.get('use_tqdm', False)):
                if "_apply_field" in kwargs:
                    results.append(func(ins[kwargs["_apply_field"]]))
                else:
                    results.append(func(ins))
        except BaseException as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e

        if new_field_name is not None:
            self._add_apply_field(results, new_field_name, kwargs)

        return results

    def add_seq_len(self, field_name: str, new_field_name=Const.INPUT_LEN):
        r"""
        将使用len()直接对field_name中每个元素作用，将其结果作为sequence length, 并放入seq_len这个field。

        :param field_name: str.
        :param new_field_name: str. 新的field_name
        :return:
        """
        if self.has_field(field_name=field_name):
            self.apply_field(len, field_name, new_field_name=new_field_name)
        else:
            raise KeyError(f"Field:{field_name} not found.")
        return self

    def drop(self, func, inplace=True):
        r"""
        func接受一个Instance，返回bool值。返回值为True时，该Instance会被移除或者不会包含在返回的DataSet中。

        :param callable func: 接受一个Instance作为参数，返回bool值。为True时删除该instance
        :param bool inplace: 是否在当前DataSet中直接删除instance；如果为False，将返回一个新的DataSet。

        :return: DataSet
        """
        if inplace:
            results = [ins for ins in self._inner_iter() if not func(ins)]
            for name, old_field in self.field_arrays.items():
                self.field_arrays[name].content = [ins[name] for ins in results]
            return self
        else:
            results = [ins for ins in self if not func(ins)]
            if len(results) != 0:
                dataset = DataSet(results)
                for field_name, field in self.field_arrays.items():
                    dataset.field_arrays[field_name].to(field)
                return dataset
            else:
                return DataSet()

    def split(self, ratio, shuffle=True):
        r"""
        将DataSet按照ratio的比例拆分，返回两个DataSet

        :param float ratio: 0<ratio<1, 返回的第一个DataSet拥有 `(1-ratio)` 这么多数据，第二个DataSet拥有`ratio`这么多数据
        :param bool shuffle: 在split前是否shuffle一下
        :return: [ :class:`~fastNLP.读取后的DataSet` , :class:`~fastNLP.读取后的DataSet` ]
        """
        assert len(self) > 1, f'DataSet with {len(self)} instance cannot be split.'
        assert isinstance(ratio, float)
        assert 0 < ratio < 1
        all_indices = [_ for _ in range(len(self))]
        if shuffle:
            np.random.shuffle(all_indices)
        split = int(ratio * len(self))
        if split == 0:
            error_msg = f'Dev DataSet has {split} instance after split.'
            logger.error(error_msg)
            raise IndexError(error_msg)
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        dev_set = DataSet()
        train_set = DataSet()
        for idx in dev_indices:
            dev_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        for field_name in self.field_arrays:
            train_set.field_arrays[field_name].to(self.field_arrays[field_name])
            dev_set.field_arrays[field_name].to(self.field_arrays[field_name])

        train_set.collater.copy_from(self.collater)
        dev_set.collater.copy_from(self.collater)
        return train_set, dev_set

    def save(self, path):
        r"""
        保存DataSet.

        :param str path: 将DataSet存在哪个路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        r"""
        从保存的DataSet pickle文件的路径中读取DataSet

        :param str path: 从哪里读取DataSet
        :return: 读取后的 :class:`~fastNLP.读取后的DataSet`。
        """
        with open(path, 'rb') as f:
            d = pickle.load(f)
            assert isinstance(d, DataSet), "The object is not DataSet, but {}.".format(type(d))
        return d

    def add_collate_fn(self, fn, name=None):
        r"""
        添加 CollateFn，collate_fn允许在生成的batch的过程中动态生成一些数据(在DataSetIter作为迭代器的情况下有效，默认情况下就是用的
        这个)。支持依次添加多个collate_fn, 如果相同的key，后面的collate_fn的结果覆盖前面的collate_fn的结果。

        :param callable fn: 传入一个可调用的function, 该function可接受的参数为List[(ind1, instance1), (ind2, instance2)]
            (某个batch被选中的所有的indice以及instance),其中ind1/ind2是该instance在dataset中的index，instance1/instance2是
            这次batch取出来的数据，包含了所有的field。返回值需要为两个dict，第一个dict的值将被认为是input，第二个dict的值被认为是
            target，返回的值至多允许一个空dict。若返回的dict中包含了被设置为input或target的field的名称，将覆盖dataset中的field。
            fastNLP不会将collate_fn的返回结果pad和转换为tensor，需要在collate_fn中完成pad和转换为tensor（不需要将tensor移动到
            gpu中，fastNLP会自动将其移动到特定gpu）。不要修改传入collate_fn中的数据，否则可能导致未知问题。
        :param str,int name: collate_fn的名称，如果不传入，默认使用自增长的数字作为key。相同的name会覆盖之前的collate_fn。
        """
        assert callable(fn), "You must pass in a callable object."
        self.collater.add_fn(fn, name=name)

    def delete_collate_fn(self, name=None):
        r"""
        删除某个collate_fn

        :param str,int name: 如果为None，则删除最近加入的collate_fn
        :return:
        """
        self.collater.delete_fn(name)

    def _collate_batch(self, ins_list):
        return self.collater.collate_batch(ins_list)

    def concat(self, dataset, inplace=True, field_mapping=None):
        """
        将当前dataset与输入的dataset结合成一个更大的dataset，需要保证两个dataset都包含了相同的field。结合后的dataset的input,target
            以及collate_fn以当前dataset为准。当dataset中包含的field多于当前的dataset，则多余的field会被忽略；若dataset中未包含所有
            当前dataset含有field，则会报错。

        :param DataSet, dataset: 需要和当前dataset concat的dataset
        :param bool, inplace: 是否直接将dataset组合到当前dataset中
        :param dict, field_mapping: 当dataset中的field名称和当前dataset不一致时，需要通过field_mapping把输入的dataset中的field
            名称映射到当前field. field_mapping为dict类型，key为dataset中的field名称，value是需要映射成的名称

        :return: DataSet
        """
        assert isinstance(dataset, DataSet), "Can only concat two datasets."

        fns_in_this_dataset = set(self.get_field_names())
        fns_in_other_dataset = dataset.get_field_names()
        reverse_field_mapping = {}
        if field_mapping is not None:
            fns_in_other_dataset = [field_mapping.get(fn, fn) for fn in fns_in_other_dataset]
            reverse_field_mapping = {v:k for k, v in field_mapping.items()}
        fns_in_other_dataset = set(fns_in_other_dataset)
        fn_not_seen = list(fns_in_this_dataset - fns_in_other_dataset)

        if fn_not_seen:
            raise RuntimeError(f"The following fields are not provided in the dataset:{fn_not_seen}")

        if inplace:
            ds = self
        else:
            ds = deepcopy(self)

        for fn in fns_in_this_dataset:
            ds.get_field(fn).content.extend(deepcopy(dataset.get_field(reverse_field_mapping.get(fn, fn)).content))

        return ds
