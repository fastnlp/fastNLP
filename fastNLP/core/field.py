"""
field模块实现了 FieldArray 和若干 Padder。 FieldArray 是  :class:`~fastNLP.DataSet` 中一列的存储方式，
原理部分请参考 :doc:`fastNLP.core.dataset`

"""
__all__ = [
    "FieldArray",
    "Padder",
    "AutoPadder",
    "EngChar2DPadder"
]

from copy import deepcopy

import numpy as np


class FieldArray(object):
    """
    别名：:class:`fastNLP.FieldArray` :class:`fastNLP.core.field.FieldArray`

    FieldArray 是用于保存 :class:`~fastNLP.DataSet` 中一个field的类型。
    
    :param str name: FieldArray的名称
    :param list,numpy.ndarray content: 列表的元素可以为list，int，float，
    :param bool is_target: 这个field是否是一个target field。
    :param bool is_input: 这个field是否是一个input field。
    :param padder: :class:`~fastNLP.Padder` 类型。赋值给fieldarray的padder的对象会被deepcopy一份，需要修改padder参数必须通过
       fieldarray.set_pad_val()。默认为None，即使用 :class:`~fastNLP.AutoPadder`  。
    :param bool ignore_type: 是否忽略该field的type，一般如果这个field不需要转为torch.FloatTensor或torch.LongTensor,
        就可以设置为True。具体意义请参考 :class:`~fastNLP.DataSet` 。
    """
    
    def __init__(self, name, content, is_target=None, is_input=None, padder=None, ignore_type=False):
        self.name = name
        if isinstance(content, list):
            # 如果DataSet使用dict初始化, content 可能是二维list/二维array/三维list
            # 如果DataSet使用list of Instance 初始化, content可能是 [list]/[array]/[2D list]
            for idx, item in enumerate(content):
                # 这是使用list of Instance 初始化时第一个样本：FieldArray(name, [field])
                # 将[np.array] 转化为 list of list
                # 也可以支持[array, array, array]的情况
                if isinstance(item, np.ndarray):
                    content[idx] = content[idx].tolist()
        elif isinstance(content, np.ndarray):
            content = content.tolist()  # convert np.ndarray into 2-D list
        else:
            raise TypeError("content in FieldArray can only be list or numpy.ndarray, got {}.".format(type(content)))
        if len(content) == 0:
            raise RuntimeError("Cannot initialize FieldArray with empty list.")
        
        self.content = content  # 1维 或 2维 或 3维 list, 形状可能不对齐
        self.content_dim = None  # 表示content是多少维的list
        if padder is None:
            padder = AutoPadder(pad_val=0)
        else:
            assert isinstance(padder, Padder), "padder must be of type Padder."
            padder = deepcopy(padder)
        self.set_padder(padder)
        self.ignore_type = ignore_type
        
        self.BASIC_TYPES = (int, float, str)  # content中可接受的Python基本类型，这里没有np.array
        
        self.pytype = None
        self.dtype = None
        self._is_input = None
        self._is_target = None
        
        if is_input is not None or is_target is not None:
            self.is_input = is_input
            self.is_target = is_target
    
    def _set_dtype(self):
        if self.ignore_type is False:
            self.pytype = self._type_detection(self.content)
            self.dtype = self._map_to_np_type(self.pytype)
    
    @property
    def is_input(self):
        return self._is_input
    
    @is_input.setter
    def is_input(self, value):
        """
            当 field_array.is_input = True / False 时被调用
        """
        if value is True:
            self._set_dtype()
        self._is_input = value
    
    @property
    def is_target(self):
        return self._is_target
    
    @is_target.setter
    def is_target(self, value):
        """
        当 field_array.is_target = True / False 时被调用
        """
        if value is True:
            self._set_dtype()
        self._is_target = value
    
    def _type_detection(self, content):
        """
        当该field被设置为is_input或者is_target时被调用

        """
        if len(content) == 0:
            raise RuntimeError("Empty list in Field {}.".format(self.name))
        
        type_set = set([type(item) for item in content])
        
        if list in type_set:
            if len(type_set) > 1:
                # list 跟 非list 混在一起
                raise RuntimeError("Mixed data types in Field {}: {}".format(self.name, list(type_set)))
            # >1维list
            inner_type_set = set()
            for l in content:
                [inner_type_set.add(type(obj)) for obj in l]
            if list not in inner_type_set:
                # 二维list
                self.content_dim = 2
                return self._basic_type_detection(inner_type_set)
            else:
                if len(inner_type_set) == 1:
                    # >2维list
                    inner_inner_type_set = set()
                    for _2d_list in content:
                        for _1d_list in _2d_list:
                            [inner_inner_type_set.add(type(obj)) for obj in _1d_list]
                    if list in inner_inner_type_set:
                        raise RuntimeError("FieldArray cannot handle 4-D or more-D list.")
                    # 3维list
                    self.content_dim = 3
                    return self._basic_type_detection(inner_inner_type_set)
                else:
                    # list 跟 非list 混在一起
                    raise RuntimeError("Mixed data types in Field {}: {}".format(self.name, list(inner_type_set)))
        else:
            # 一维list
            for content_type in type_set:
                if content_type not in self.BASIC_TYPES:
                    raise RuntimeError("Unexpected data type in Field '{}'. Expect one of {}. Got {}.".format(
                        self.name, self.BASIC_TYPES, content_type))
            self.content_dim = 1
            return self._basic_type_detection(type_set)
    
    def _basic_type_detection(self, type_set):
        """
        :param type_set: a set of Python types
        :return: one of self.BASIC_TYPES
        """
        if len(type_set) == 1:
            return type_set.pop()
        elif len(type_set) == 2:
            # 有多个basic type; 可能需要up-cast
            if float in type_set and int in type_set:
                # up-cast int to float
                return float
            else:
                # str 跟 int 或者 float 混在一起
                raise RuntimeError("Mixed data types in Field {}: {}".format(self.name, list(type_set)))
        else:
            # str, int, float混在一起
            raise RuntimeError("Mixed data types in Field {}: {}".format(self.name, list(type_set)))
    
    def _1d_list_check(self, val):
        """如果不是1D list就报错
        """
        type_set = set((type(obj) for obj in val))
        if any(obj not in self.BASIC_TYPES for obj in type_set):
            raise ValueError("Mixed data types in Field {}: {}".format(self.name, list(type_set)))
        self._basic_type_detection(type_set)
        # otherwise: _basic_type_detection will raise error
        return True
    
    def _2d_list_check(self, val):
        """如果不是2D list 就报错
        """
        type_set = set(type(obj) for obj in val)
        if list(type_set) != [list]:
            raise ValueError("Mixed data types in Field {}: {}".format(self.name, type_set))
        inner_type_set = set()
        for l in val:
            for obj in l:
                inner_type_set.add(type(obj))
        self._basic_type_detection(inner_type_set)
        return True
    
    @staticmethod
    def _map_to_np_type(basic_type):
        type_mapping = {int: np.int64, float: np.float64, str: np.str, np.ndarray: np.ndarray}
        return type_mapping[basic_type]
    
    def __repr__(self):
        return "FieldArray {}: {}".format(self.name, self.content.__repr__())
    
    def append(self, val):
        """将val append到这个field的尾部。如果这个field已经被设置为input或者target，则在append之前会检查该类型是否与已有
        的内容是匹配的。

        :param Any val: 需要append的值。
        """
        if self.ignore_type is False:
            if isinstance(val, list):
                pass
            elif isinstance(val, tuple):  # 确保最外层是list
                val = list(val)
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            elif any((isinstance(val, t) for t in self.BASIC_TYPES)):
                pass
            else:
                raise RuntimeError(
                    "Unexpected data type {}. Should be list, np.array, or {}".format(type(val), self.BASIC_TYPES))
            
            if self.is_input is True or self.is_target is True:
                if type(val) == list:
                    if len(val) == 0:
                        raise ValueError("Cannot append an empty list.")
                    if self.content_dim == 2 and self._1d_list_check(val):
                        # 1维list检查
                        pass
                    elif self.content_dim == 3 and self._2d_list_check(val):
                        # 2维list检查
                        pass
                    else:
                        raise RuntimeError(
                            "Dimension not matched: expect dim={}, got {}.".format(self.content_dim - 1, val))
                elif type(val) in self.BASIC_TYPES and self.content_dim == 1:
                    # scalar检查
                    if type(val) == float and self.pytype == int:
                        self.pytype = float
                        self.dtype = self._map_to_np_type(self.pytype)
                else:
                    raise RuntimeError(
                        "Unexpected data type {}. Should be list, np.array, or {}".format(type(val), self.BASIC_TYPES))
        self.content.append(val)
    
    def __getitem__(self, indices):
        return self.get(indices, pad=False)
    
    def __setitem__(self, idx, val):
        assert isinstance(idx, int)
        self.content[idx] = val
    
    def get(self, indices, pad=True):
        """
        根据给定的indices返回内容

        :param int,List[int] indices: 获取indices对应的内容。
        :param bool pad:  是否对返回的结果进行padding。仅对indices为List[int]时有效
        :return: 根据给定的indices返回的内容，可能是单个值或List
        """
        if isinstance(indices, int):
            return self.content[indices]
        if self.is_input is False and self.is_target is False:
            raise RuntimeError("Please specify either is_input or is_target is True for {}".format(self.name))
        
        contents = [self.content[i] for i in indices]
        if self.padder is None or pad is False:
            return np.array(contents)
        else:
            return self.padder(contents, field_name=self.name, field_ele_dtype=self.dtype)
    
    def set_padder(self, padder):
        """
        设置padder，在这个field进行pad的时候用这个padder进行pad，如果为None则不进行pad。

        :param padder: :class:`~fastNLP.Padder` 类型，设置为None即删除padder。
        """
        if padder is not None:
            assert isinstance(padder, Padder), "padder must be of type Padder."
            self.padder = deepcopy(padder)
        else:
            self.padder = None
    
    def set_pad_val(self, pad_val):
        """
        修改padder的pad_val.

        :param int pad_val: 该field的pad值设置为该值。
        """
        if self.padder is not None:
            self.padder.set_pad_val(pad_val)
        return self
    
    def __len__(self):
        """
        Returns the size of FieldArray.

        :return int length:
        """
        return len(self.content)
    
    def to(self, other):
        """
        将other的属性复制给本FieldArray(other必须为FieldArray类型).
        属性包括 is_input, is_target, padder, ignore_type

        :param  other: :class:`~fastNLP.FieldArray` 从哪个field拷贝属性
        :return: :class:`~fastNLP.FieldArray`
        """
        assert isinstance(other, FieldArray), "Only support FieldArray type, not {}.".format(type(other))
        
        self.is_input = other.is_input
        self.is_target = other.is_target
        self.padder = other.padder
        self.ignore_type = other.ignore_type
        
        return self


def _is_iterable(content):
    try:
        _ = (e for e in content)
    except TypeError:
        return False
    return True


class Padder:
    """
    别名：:class:`fastNLP.Padder` :class:`fastNLP.core.field.Padder`

    所有padder都需要继承这个类，并覆盖__call__方法。
    用于对batch进行padding操作。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前deepcopy一份。
    
    .. py:function:: __call__(self, contents, field_name, field_ele_dtype):
        传入的是List内容。假设有以下的DataSet。
        
        :param list(Any) contents: 传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param str, field_name: field的名称。
        :param np.int64,np.float64,np.str,None, field_ele_dtype: 该field的内层元素的类型。如果该field的ignore_type为True，该这个值为None。
        :return: np.array([padded_element])
    
    """
    
    def __init__(self, pad_val=0, **kwargs):
        self.pad_val = pad_val
    
    def set_pad_val(self, pad_val):
        self.pad_val = pad_val
    
    def __call__(self, contents, field_name, field_ele_dtype):
        """
        传入的是List内容。假设有以下的DataSet。

        :param list(Any) contents: 传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param str, field_name: field的名称。
        :param np.int64,np.float64,np.str,None, field_ele_dtype: 该field的内层元素的类型。如果该field的ignore_type为True，该这个值为None。
        :return: np.array([padded_element])

        Example::

            from fastNLP import DataSet
            from fastNLP import Instance
            dataset = DataSet()
            dataset.append(Instance(sent='this is a demo', length=4,
                                    chars=[['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']]))
            dataset.append(Instance(sent='another one', length=2,
                                    chars=[['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]))
            如果调用
            batch = dataset.get([0,1], pad=True)
            sent这个field的padder的__call__会接收到的内容会是
                [
                    'this is a demo',
                    'another one'
                ]

            length这个field的padder的__call__会接收到的内容会是
                [4, 2]

            chars这个field的padder的__call__会接收到的内容会是
                [
                    [['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']],
                    [['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]
                ]

        即把每个instance中某个field的内容合成一个List传入

        """
        raise NotImplementedError


class AutoPadder(Padder):
    """
    别名：:class:`fastNLP.AutoPadder` :class:`fastNLP.core.field.AutoPadder`

    根据contents的数据自动判定是否需要做padding。

    1 如果元素类型(元素类型是指field中最里层元素的数据类型, 可以通过FieldArray.dtype查看，比如['This', 'is', ...]的元素类
    型为np.str, [[1,2], ...]的元素类型为np.int64)的数据不为(np.int64, np.float64)则不会进行pad

    2 如果元素类型为(np.int64, np.float64),

        2.1 如果该field的内容为(np.int64, np.float64)，比如为seq_len, 则不进行padding

        2.2 如果该field的内容为List, 那么会将Batch中的List pad为一样长。若该List下还有里层的List需要padding，请使用其它padder。
        即如果Instance中field形如[1, 2, 3, ...]，则可以pad；若为[[1,2], [3,4, ...]]则不能进行pad
    """
    
    def __init__(self, pad_val=0):
        """
        :param pad_val: int, padding的位置使用该index
        """
        super().__init__(pad_val=pad_val)
    
    def _is_two_dimension(self, contents):
        """
        判断contents是不是只有两个维度。[[1,2], [3]]是两个维度. [[[1,2], [3, 4, 5]], [[4,5]]]有三个维度
        :param contents:
        :return:
        """
        value = contents[0]
        if isinstance(value, (np.ndarray, list)):
            value = value[0]
            if isinstance(value, (np.ndarray, list)):
                return False
            return True
        return False
    
    def __call__(self, contents, field_name, field_ele_dtype):
        
        if not _is_iterable(contents[0]):
            array = np.array([content for content in contents], dtype=field_ele_dtype)
        elif field_ele_dtype in (np.int64, np.float64) and self._is_two_dimension(contents):
            max_len = max([len(content) for content in contents])
            array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
            for i, content in enumerate(contents):
                array[i][:len(content)] = content
        elif field_ele_dtype is None:
            array = np.array(contents)  # 当ignore_type=True时，直接返回contents
        else:  # should only be str
            array = np.array([content for content in contents])
        return array


class EngChar2DPadder(Padder):
    """
    别名：:class:`fastNLP.EngChar2DPadder` :class:`fastNLP.core.field.EngChar2DPadder`

    用于为英语执行character级别的2D padding操作。对应的field内容应该类似[['T', 'h', 'i', 's'], ['a'], ['d', 'e', 'm', 'o']]，
    但这个Padder只能处理index为int的情况。

    padded过后的batch内容，形状为(batch_size, max_sentence_length, max_word_length). max_sentence_length为这个batch中最大句
    子长度；max_word_length为这个batch中最长的word的长度::

        from fastNLP import DataSet
        from fastNLP import EngChar2DPadder
        from fastNLP import Vocabulary
        dataset = DataSet({'sent': ['This is the first demo', 'This is the second demo']})
        dataset.apply(lambda ins:[list(word) for word in ins['sent'].split()], new_field_name='chars')
        vocab = Vocabulary()
        vocab.from_dataset(dataset, field_name='chars')
        vocab.index_dataset(dataset, field_name='chars')
        dataset.set_input('chars')
        padder = EngChar2DPadder()
        dataset.set_padder('chars', padder)  # chars这个field的设置为了EnChar2DPadder

    """
    
    def __init__(self, pad_val=0, pad_length=0):
        """
        :param pad_val: int, pad的位置使用该index
        :param pad_length: int, 如果为0则取一个batch中最大的单词长度作为padding长度。如果为大于0的数，则将所有单词的长度
            都pad或截取到该长度.
        """
        super().__init__(pad_val=pad_val)
        
        self.pad_length = pad_length
    
    def _exactly_three_dims(self, contents, field_name):
        """
        检查传入的contents是否刚好是3维，如果不是3维就报错。理论上，第一个维度是batch，第二个维度是word，第三个维度是character
        :param contents:
        :param field_name: str
        :return:
        """
        if not isinstance(contents, list):
            raise TypeError("contents should be a list, not {}.".format(type(contents)))
        value = contents[0]
        try:
            value = value[0]
        except:
            raise ValueError("Field:{} only has one dimension.".format(field_name))
        try:
            value = value[0]
        except:
            raise ValueError("Field:{} only has two dimensions.".format(field_name))
        
        if _is_iterable(value):
            raise ValueError("Field:{} has more than 3 dimension.".format(field_name))
    
    def __call__(self, contents, field_name, field_ele_dtype):
        """
        期望输入类似于
        [
            [[0, 2], [2, 3, 4], ..],
            [[9, 8, 2, 4], [1, 2,], ...],
            ....
        ]

        :param contents:
        :param field_name:
        :param field_ele_dtype
        :return:
        """
        if field_ele_dtype not in (np.int64, np.float64):
            raise TypeError('dtype of Field:{} should be np.int64 or np.float64 to do 2D padding, get {}.'.format(
                field_name, field_ele_dtype
            ))
        self._exactly_three_dims(contents, field_name)
        if self.pad_length < 1:
            max_char_length = max(max([[len(char_lst) for char_lst in word_lst] for word_lst in contents]))
        else:
            max_char_length = self.pad_length
        max_sent_length = max(len(word_lst) for word_lst in contents)
        batch_size = len(contents)
        dtype = type(contents[0][0][0])
        
        padded_array = np.full((batch_size, max_sent_length, max_char_length), fill_value=self.pad_val,
                               dtype=dtype)
        for b_idx, word_lst in enumerate(contents):
            for c_idx, char_lst in enumerate(word_lst):
                chars = char_lst[:max_char_length]
                padded_array[b_idx, c_idx, :len(chars)] = chars
        
        return padded_array
