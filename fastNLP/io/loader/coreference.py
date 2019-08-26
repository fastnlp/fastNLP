from ...core.dataset import DataSet
from ..file_reader import _read_json
from ...core.instance import Instance
from .json import JsonLoader


class CRLoader(JsonLoader):
    def __init__(self, fields=None, dropna=False):
        super().__init__(fields, dropna)

    def _load(self, path):
        """
        加载数据
        :param path:
        :return:
        """
        dataset = DataSet()
        for idx, d in _read_json(path, fields=self.fields_list, dropna=self.dropna):
            if self.fields:
                ins = {self.fields[k]: v for k, v in d.items()}
            else:
                ins = d
            dataset.append(Instance(**ins))
        return dataset