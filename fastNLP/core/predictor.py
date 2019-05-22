"""
    ..todo::
        检查这个类是否需要
"""
from collections import defaultdict

import torch

from . import Batch
from . import DataSet
from . import SequentialSampler
from .utils import _build_args


class Predictor(object):
    """
    An interface for predicting outputs based on trained models.

    It does not care about evaluations of the model, which is different from Tester.
    This is a high-level model wrapper to be called by FastNLP.
    This class does not share any operations with Trainer and Tester.
    Currently, Predictor does not support GPU.
    """

    def __init__(self, network):
        if not isinstance(network, torch.nn.Module):
            raise ValueError(
                "Only fastNLP.models.BaseModel or torch.nn,Module is allowed, not {}".format(type(network)))
        self.network = network
        self.batch_size = 1
        self.batch_output = []

    def predict(self, data, seq_len_field_name=None):
        """Perform inference using the trained model.

        :param data: a DataSet object.
        :param str seq_len_field_name: field name indicating sequence lengths
        :return: list of batch outputs
        """
        if not isinstance(data, DataSet):
            raise ValueError("Only Dataset class is allowed, not {}.".format(type(data)))
        if seq_len_field_name is not None and seq_len_field_name not in data.field_arrays:
            raise ValueError("Field name {} not found in DataSet {}.".format(seq_len_field_name, data))

        self.network.eval()
        batch_output = defaultdict(list)
        data_iterator = Batch(data, batch_size=self.batch_size, sampler=SequentialSampler(), as_numpy=False,
                              prefetch=False)

        if hasattr(self.network, "predict"):
            predict_func = self.network.predict
        else:
            predict_func = self.network.forward

        with torch.no_grad():
            for batch_x, _ in data_iterator:
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

        return batch_output
