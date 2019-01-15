import numpy as np


class PadderBase:
    """
        所有padder都需要继承这个类，并覆盖__call__()方法。
        用于对batch进行padding操作。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前deepcopy一份。
    """
    def __init__(self, pad_val=0, **kwargs):
        self.pad_val = pad_val

    def set_pad_val(self, pad_val):
        self.pad_val = pad_val

    def __call__(self, contents, field_name, field_ele_dtype):
        """
        传入的是List内容。假设有以下的DataSet。
        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        dataset.append(Instance(word='this is a demo', length=4,
                                    chars=[['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']]))
        dataset.append(Instance(word='another one', length=2,
                                    chars=[['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]))
        # 如果batch_size=2, 下面只是用str的方式看起来更直观一点，但实际上可能word和chars在pad时都已经为index了。
        word这个field的pad_func会接收到的内容会是
            [
                'this is a demo',
                'another one'
            ]
        length这个field的pad_func会接收到的内容会是
            [4, 2]
        chars这个field的pad_func会接收到的内容会是
            [
                [['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']],
                [['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]
            ]
        即把每个instance中某个field的内容合成一个List传入
        :param contents: List[element]。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param field_name: str, field的名称，帮助定位错误
        :param field_ele_dtype: np.int64, np.float64, np.str. 该field的内层list元素的类型。辅助判断是否pad，大多数情况用不上
        :return: List[padded_element]或np.array([padded_element])
        """
        raise NotImplementedError


class AutoPadder(PadderBase):
    """
    根据contents的数据自动判定是否需要做padding。
    (1) 如果元素类型(元素类型是指field中最里层List的元素的数据类型, 可以通过FieldArray.dtype查看，比如['This', 'is', ...]的元素类
        型为np.str, [[1,2], ...]的元素类型为np.int64)的数据不为(np.int64, np.float64)则不会进行padding
    (2) 如果元素类型为(np.int64, np.float64),
        (2.1) 如果该field的内容只有一个，比如为sequence_length, 则不进行padding
        (2.2) 如果该field的内容为List, 那么会将Batch中的List pad为一样长。若该List下还有里层的List需要padding，请使用其它padder。
            如果某个instance中field为[1, 2, 3]，则可以pad； 若为[[1,2], [3,4, ...]]则不能进行pad
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
        if isinstance(value , (np.ndarray, list)):
            value = value[0]
            if isinstance(value, (np.ndarray, list)):
                return False
            return True
        return False

    def __call__(self, contents, field_name, field_ele_dtype):
        if not is_iterable(contents[0]):
            array = np.array([content for content in contents], dtype=field_ele_dtype)
        elif field_ele_dtype in (np.int64, np.float64) and self._is_two_dimension(contents):
            max_len = max([len(content) for content in contents])
            array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
            for i, content in enumerate(contents):
                array[i][:len(content)] = content
        else:  # should only be str
            array = np.array([content for content in contents])
        return array


class FieldArray(object):
    """``FieldArray`` is the collection of ``Instance``s of the same field.
    It is the basic element of ``DataSet`` class.

    :param str name: the name of the FieldArray
    :param list content: a list of int, float, str or np.ndarray, or a list of list of one, or a np.ndarray.
    :param bool is_target: If True, this FieldArray is used to compute loss.
    :param bool is_input: If True, this FieldArray is used to the model input.
    :param padder: PadderBase类型。大多数情况下都不需要设置该值，除非需要在多个维度上进行padding(比如英文中对character进行padding)
    """

    def __init__(self, name, content, is_target=None, is_input=None, padder=AutoPadder(pad_val=0)):
        self.name = name
        if isinstance(content, list):
            content = content
        elif isinstance(content, np.ndarray):
            content = content.tolist()  # convert np.ndarray into 2-D list
        else:
            raise TypeError("content in FieldArray can only be list or numpy.ndarray, got {}.".format(type(content)))
        self.content = content
        self.set_padder(padder)

        self._is_target = None
        self._is_input = None

        self.BASIC_TYPES = (int, float, str, np.ndarray)
        self.is_2d_list = False
        self.pytype = None  # int, float, str, or np.ndarray
        self.dtype = None  # np.int64, np.float64, np.str

        if is_input is not None:
            self.is_input = is_input
        if is_target is not None:
            self.is_target = is_target

    @property
    def is_input(self):
        return self._is_input

    @is_input.setter
    def is_input(self, value):
        if value is True:
            self.pytype = self._type_detection(self.content)
            self.dtype = self._map_to_np_type(self.pytype)
        self._is_input = value

    @property
    def is_target(self):
        return self._is_target

    @is_target.setter
    def is_target(self, value):
        if value is True:
            self.pytype = self._type_detection(self.content)
            self.dtype = self._map_to_np_type(self.pytype)
        self._is_target = value

    def _type_detection(self, content):
        """

        :param content: a list of int, float, str or np.ndarray, or a list of list of one.
        :return type: one of int, float, str, np.ndarray

        """
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], list):
            # content is a 2-D list
            if not all(isinstance(_, list) for _ in content):  # strict check 2-D list
                raise TypeError("Please provide 2-D list.")
            type_set = set([self._type_detection(x) for x in content])
            if len(type_set) == 2 and int in type_set and float in type_set:
                type_set = {float}
            elif len(type_set) > 1:
                raise TypeError("Cannot create FieldArray with more than one type. Provided {}".format(type_set))
            self.is_2d_list = True
            return type_set.pop()

        elif isinstance(content, list):
            # content is a 1-D list
            if len(content) == 0:
                # the old error is not informative enough.
                raise RuntimeError("Cannot create FieldArray with an empty list. Or one element in the list is empty.")
            type_set = set([type(item) for item in content])

            if len(type_set) == 1 and tuple(type_set)[0] in self.BASIC_TYPES:
                return type_set.pop()
            elif len(type_set) == 2 and float in type_set and int in type_set:
                # up-cast int to float
                return float
            else:
                raise TypeError("Cannot create FieldArray with type {}".format(*type_set))
        else:
            raise TypeError("Cannot create FieldArray with type {}".format(type(content)))

    @staticmethod
    def _map_to_np_type(basic_type):
        type_mapping = {int: np.int64, float: np.float64, str: np.str, np.ndarray: np.ndarray}
        return type_mapping[basic_type]

    def __repr__(self):
        return "FieldArray {}: {}".format(self.name, self.content.__repr__())

    def append(self, val):
        """Add a new item to the tail of FieldArray.

        :param val: int, float, str, or a list of one.
        """
        if self.is_target is True or self.is_input is True:
            # only check type when used as target or input

            val_type = type(val)
            if val_type == list:  # shape check
                if self.is_2d_list is False:
                    raise RuntimeError("Cannot append a list into a 1-D FieldArray. Please provide an element.")
                if len(val) == 0:
                    raise RuntimeError("Cannot append an empty list.")
                val_list_type = set([type(_) for _ in val])  # type check
                if len(val_list_type) == 2 and int in val_list_type and float in val_list_type:
                    # up-cast int to float
                    val_type = float
                elif len(val_list_type) == 1:
                    val_type = val_list_type.pop()
                else:
                    raise TypeError("Cannot append a list of {}".format(val_list_type))
            else:
                if self.is_2d_list is True:
                    raise RuntimeError("Cannot append a non-list into a 2-D list. Please provide a list.")

            if val_type == float and self.pytype == int:
                # up-cast
                self.pytype = float
                self.dtype = self._map_to_np_type(self.pytype)
            elif val_type == int and self.pytype == float:
                pass
            elif val_type == self.pytype:
                pass
            else:
                raise TypeError("Cannot append type {} into type {}".format(val_type, self.pytype))

        self.content.append(val)

    def __getitem__(self, indices):
        return self.get(indices)

    def __setitem__(self, idx, val):
        assert isinstance(idx, int)
        self.content[idx] = val

    def get(self, indices, pad=True):
        """Fetch instances based on indices.

        :param indices: an int, or a list of int.
        :param pad: bool, 是否对返回的结果进行padding。
        :return:
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
        设置padding方式

        :param padder: PadderBase类型或None. 设置为None即删除padder.
        :return:
        """
        if padder is not None:
            assert isinstance(padder, PadderBase), "padder must be of type PadderBase."
        self.padder = padder

    def set_pad_val(self, pad_val):
        """
        修改padder的pad_val.
        :param pad_val: int。
        :return:
        """
        if self.padder is not None:
            self.padder.set_pad_val(pad_val)


    def __len__(self):
        """Returns the size of FieldArray.

        :return int length:
        """
        return len(self.content)


def is_iterable(content):
    try:
        _ = (e for e in content)
    except TypeError:
        return False
    return True


class EngChar2DPadder(PadderBase):
    """
    用于为英语执行character级别的2D padding操作。对应的field内容应该为[['T', 'h', 'i', 's'], ['a'], ['d', 'e', 'm', 'o']](这里为
        了更直观，把它们写为str，但实际使用时它们应该是character的index)。
    padded过后的batch内容，形状为(batch_size, max_sentence_length, max_word_length). max_sentence_length最大句子长度。
        max_word_length最长的word的长度

    """
    def __init__(self, pad_val=0, pad_length=0):
        """
        :param pad_val: int, padding的位置使用该index
        :param pad_length: int, 如果为0则取一个batch中最大的单词长度作为padding长度。如果为大于0的数，则将所有单词的长度都pad或截
            取到该长度.
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
            value = value[1]
        except:
            raise ValueError("Field:{} only has two dimensions.".format(field_name))

        if is_iterable(value):
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