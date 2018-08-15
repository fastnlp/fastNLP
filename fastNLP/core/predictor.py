import numpy as np
import torch

from fastNLP.core.action import Batchifier, SequentialSampler
from fastNLP.core.action import convert_to_torch_tensor
from fastNLP.loader.preprocess import load_pickle, DEFAULT_UNKNOWN_LABEL
from fastNLP.modules import utils


def make_batch(iterator, use_cuda, output_length=False, max_len=None, min_len=None):
    """Batch and Pad data, only for Inference.

    :param iterator: An iterable object that returns a list of indices representing a mini-batch of samples.
    :param use_cuda: bool, whether to use GPU
    :param output_length: bool, whether to output the original length of the sequence before padding. (default: False)
    :param max_len: int, maximum sequence length. Longer sequences will be clipped. (default: None)
    :param min_len: int, minimum sequence length. Shorter sequences will be padded. (default: None)
    :return:
    """
    for batch_x in iterator:
        batch_x = pad(batch_x)
        # convert list to tensor
        batch_x = convert_to_torch_tensor(batch_x, use_cuda)

        # trim data to max_len
        if max_len is not None and batch_x.size(1) > max_len:
            batch_x = batch_x[:, :max_len]
        if min_len is not None and batch_x.size(1) < min_len:
            pad_tensor = torch.zeros(batch_x.size(0), min_len - batch_x.size(1)).to(batch_x)
            batch_x = torch.cat((batch_x, pad_tensor), 1)

        if output_length:
            seq_len = [len(x) for x in batch_x]
            yield tuple([batch_x, seq_len])
        else:
            yield batch_x


def pad(batch, fill=0):
    """ Pad a mini-batch of sequence samples to maximum length of this batch.

    :param batch: list of list
    :param fill: word index to pad, default 0.
    :return batch: a padded mini-batch
    """
    max_length = max([len(x) for x in batch])
    for idx, sample in enumerate(batch):
        if len(sample) < max_length:
            batch[idx] = sample + ([fill] * (max_length - len(sample)))
    return batch


class Predictor(object):
    """An interface for predicting outputs based on trained models.

    It does not care about evaluations of the model, which is different from Tester.
    This is a high-level model wrapper to be called by FastNLP.
    This class does not share any operations with Trainer and Tester.
    Currently, Predictor does not support GPU.
    """

    def __init__(self, pickle_path):
        self.batch_size = 1
        self.batch_output = []
        self.iterator = None
        self.pickle_path = pickle_path
        self.index2label = load_pickle(self.pickle_path, "id2class.pkl")
        self.word2index = load_pickle(self.pickle_path, "word2id.pkl")

    def predict(self, network, data):
        """Perform inference using the trained model.

        :param network: a PyTorch model
        :param data: list of list of strings
        :return: list of list of strings, [num_examples, tag_seq_length]
        """
        # transform strings into indices
        data = self.prepare_input(data)

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)
        self.batch_output.clear()

        data_iterator = iter(Batchifier(SequentialSampler(data), self.batch_size, drop_last=False))

        for batch_x in self.make_batch(data_iterator, use_cuda=False):

            prediction = self.data_forward(network, batch_x)

            self.batch_output.append(prediction)

        return self.prepare_output(self.batch_output)

    def mode(self, network, test=True):
        if test:
            network.eval()
        else:
            network.train()

    def data_forward(self, network, x):
        """Forward through network."""
        raise NotImplementedError

    def make_batch(self, iterator, use_cuda):
        raise NotImplementedError

    def prepare_input(self, data):
        """Transform two-level list of strings into that of index.

        :param data:
                [
                    [word_11, word_12, ...],
                    [word_21, word_22, ...],
                    ...
                ]
        :return data_index: list of list of int.
        """
        assert isinstance(data, list)
        data_index = []
        default_unknown_index = self.word2index[DEFAULT_UNKNOWN_LABEL]
        for example in data:
            data_index.append([self.word2index.get(w, default_unknown_index) for w in example])
        return data_index

    def prepare_output(self, data):
        """Transform list of batch outputs into strings."""
        raise NotImplementedError


class SeqLabelInfer(Predictor):
    """
    Inference on sequence labeling models.
    """

    def __init__(self, pickle_path):
        super(SeqLabelInfer, self).__init__(pickle_path)

    def data_forward(self, network, inputs):
        """
        This is only for sequence labeling with CRF decoder.
        :param network: a PyTorch model
        :param inputs: tuple of (x, seq_len)
                        x: Tensor of shape [batch_size, max_len], where max_len is the maximum length of the mini-batch
                            after padding.
                        seq_len: list of int, the lengths of sequences before padding.
        :return prediction: Tensor of shape [batch_size, max_len]
        """
        if not isinstance(inputs[1], list) and isinstance(inputs[0], list):
            raise RuntimeError("output_length must be true for sequence modeling.")
        # unpack the returned value from make_batch
        x, seq_len = inputs[0], inputs[1]
        batch_size, max_len = x.size(0), x.size(1)
        mask = utils.seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)
        y = network(x)
        prediction = network.prediction(y, mask)
        return torch.Tensor(prediction)

    def make_batch(self, iterator, use_cuda):
        return make_batch(iterator, use_cuda, output_length=True)

    def prepare_output(self, batch_outputs):
        """Transform list of batch outputs into strings.

        :param batch_outputs: list of 2-D Tensor, shape [num_batch, batch-size, tag_seq_length].
        :return results: 2-D list of strings, shape [num_examples, tag_seq_length]
        """
        results = []
        for batch in batch_outputs:
            for example in np.array(batch):
                results.append([self.index2label[int(x)] for x in example])
        return results


class ClassificationInfer(Predictor):
    """
    Inference on Classification models.
    """

    def __init__(self, pickle_path):
        super(ClassificationInfer, self).__init__(pickle_path)

    def data_forward(self, network, x):
        """Forward through network."""
        logits = network(x)
        return logits

    def make_batch(self, iterator, use_cuda):
        return make_batch(iterator, use_cuda, output_length=False, min_len=5)

    def prepare_output(self, batch_outputs):
        """
        Transform list of batch outputs into strings.
        :param batch_outputs: list of 2-D Tensor, of shape [num_batch, batch-size, num_classes].
        :return results: list of strings
        """
        results = []
        for batch_out in batch_outputs:
            idx = np.argmax(batch_out.detach().numpy())
            results.append(self.index2label[idx])
        return results
