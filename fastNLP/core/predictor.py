r"""undocumented"""

__all__ = [
    "Predictor"
]

from collections import defaultdict

import torch

from . import DataSet
from . import DataSetIter
from . import SequentialSampler
from .utils import _build_args, _move_dict_value_to_device, _get_model_device


class Predictor(object):
    r"""
    一个根据训练模型预测输出的预测器（Predictor）

    与测试器（Tester）不同的是，predictor不关心模型性能的评价指标，只做inference。
    这是一个fastNLP调用的高级模型包装器。它与Trainer、Tester不共享任何操作。
    """
    
    def __init__(self, network):
        r"""
        
        :param torch.nn.Module network: 用来完成预测任务的模型
        """
        if not isinstance(network, torch.nn.Module):
            raise ValueError(
                "Only fastNLP.models.BaseModel or torch.nn,Module is allowed, not {}".format(type(network)))
        self.network = network
        self.batch_size = 1
        self.batch_output = []
    
    def predict(self, data: DataSet, seq_len_field_name=None):
        r"""用已经训练好的模型进行inference.

        :param fastNLP.DataSet data: 待预测的数据集
        :param str seq_len_field_name: 表示序列长度信息的field名字
        :return: dict dict里面的内容为模型预测的结果
        """
        if not isinstance(data, DataSet):
            raise ValueError("Only Dataset class is allowed, not {}.".format(type(data)))
        if seq_len_field_name is not None and seq_len_field_name not in data.field_arrays:
            raise ValueError("Field name {} not found in DataSet {}.".format(seq_len_field_name, data))
        
        prev_training = self.network.training
        self.network.eval()
        network_device = _get_model_device(self.network)
        batch_output = defaultdict(list)
        data_iterator = DataSetIter(data, batch_size=self.batch_size, sampler=SequentialSampler(), as_numpy=False)
        
        if hasattr(self.network, "predict"):
            predict_func = self.network.predict
        else:
            predict_func = self.network.forward
        
        with torch.no_grad():
            for batch_x, _ in data_iterator:
                _move_dict_value_to_device(batch_x, _, device=network_device)
                refined_batch_x = _build_args(predict_func, **batch_x)
                prediction = predict_func(**refined_batch_x)
                
                if seq_len_field_name is not None:
                    seq_lens = batch_x[seq_len_field_name].tolist()
                
                for key, value in prediction.items():
                    value = value.cpu().numpy()
                    if len(value.shape) == 1 or (len(value.shape) == 2 and value.shape[1] == 1):
                        batch_output[key].extend(value.tolist())
                    else:
                        if seq_len_field_name is not None:
                            tmp_batch = []
                            for idx, seq_len in enumerate(seq_lens):
                                tmp_batch.append(value[idx, :seq_len])
                            batch_output[key].extend(tmp_batch)
                        else:
                            batch_output[key].append(value)
        
        self.network.train(prev_training)
        return batch_output
