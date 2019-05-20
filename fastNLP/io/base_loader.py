__all__ = [
    "BaseLoader"
]

import _pickle as pickle
import os


class BaseLoader(object):
    """
    各个 Loader 的基类，提供了 API 的参考。

    """
    
    def __init__(self):
        super(BaseLoader, self).__init__()
    
    @staticmethod
    def load_lines(data_path):
        """
        按行读取，舍弃每行两侧空白字符，返回list of str

        :param data_path: 读取数据的路径
        """
        with open(data_path, "r", encoding="utf=8") as f:
            text = f.readlines()
        return [line.strip() for line in text]
    
    @classmethod
    def load(cls, data_path):
        """
        先按行读取，去除一行两侧空白，再提取每行的字符。返回list of list of str
        
        :param data_path:
        """
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.readlines()
        return [[word for word in sent.strip()] for sent in text]
    
    @classmethod
    def load_with_cache(cls, data_path, cache_path):
        """缓存版的load
        """
        if os.path.isfile(cache_path) and os.path.getmtime(data_path) < os.path.getmtime(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            obj = cls.load(data_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
            return obj


class DataLoaderRegister:
    _readers = {}
    
    @classmethod
    def set_reader(cls, reader_cls, read_fn_name):
        # def wrapper(reader_cls):
        if read_fn_name in cls._readers:
            raise KeyError(
                'duplicate reader: {} and {} for read_func: {}'.format(cls._readers[read_fn_name], reader_cls,
                                                                       read_fn_name))
        if hasattr(reader_cls, 'load'):
            cls._readers[read_fn_name] = reader_cls().load
        return reader_cls
    
    @classmethod
    def get_reader(cls, read_fn_name):
        if read_fn_name in cls._readers:
            return cls._readers[read_fn_name]
        raise AttributeError('no read function: {}'.format(read_fn_name))
    
    # TODO 这个类使用在何处？
