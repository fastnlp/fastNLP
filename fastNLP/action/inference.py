import torch

from fastNLP.action.action import Batchifier, SequentialSampler
from fastNLP.loader.preprocess import load_pickle, DEFAULT_UNKNOWN_LABEL


class Inference(object):
    """
    This is an interface focusing on predicting output based on trained models.
    It does not care about evaluations of the model, which is different from Tester.
    This is a high-level model wrapper to be called by FastNLP.

    """

    def __init__(self, pickle_path):
        self.batch_size = 1
        self.batch_output = []
        self.iterator = None
        self.pickle_path = pickle_path
        self.index2label = load_pickle(self.pickle_path, "id2class.pkl")
        self.word2index = load_pickle(self.pickle_path, "word2id.pkl")

    def predict(self, network, data):
        """
        Perform inference.
        :param network:
        :param data: multi-level lists of strings
        :return result: the model outputs
        """
        # transform strings into indices
        data = self.prepare_input(data)

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)

        self.iterator = iter(Batchifier(SequentialSampler(data), self.batch_size, drop_last=False))

        num_iter = len(data) // self.batch_size

        for step in range(num_iter):
            batch_x = self.make_batch(data)

            prediction = self.data_forward(network, batch_x)

            self.batch_output.append(prediction)

        return self.prepare_output(self.batch_output)

    def mode(self, network, test=True):
        if test:
            network.eval()
        else:
            network.train()
        self.batch_output.clear()

    def data_forward(self, network, x):
        """
        This is only for sequence labeling with CRF decoder. To do: more general ?
        :param network:
        :param x:
        :return:
        """
        seq_len = [len(seq) for seq in x]
        x = torch.Tensor(x).long()
        y = network(x)
        prediction = network.prediction(y, seq_len)
        # To do: hide framework
        results = torch.Tensor(prediction).view(-1, )
        return list(results.data)

    def make_batch(self, data):
        indices = next(self.iterator)
        batch_x = [data[idx] for idx in indices]
        if self.batch_size > 1:
            batch_x = self.pad(batch_x)
        return batch_x

    @staticmethod
    def pad(batch, fill=0):
        """
        Pad a batch of samples to maximum length.
        :param batch: list of list
        :param fill: word index to pad, default 0.
        :return: a padded batch
        """
        max_length = max([len(x) for x in batch])
        for idx, sample in enumerate(batch):
            if len(sample) < max_length:
                batch[idx] = sample + [fill * (max_length - len(sample))]
        return batch

    def prepare_input(self, data):
        """
        Transform three-level list of strings into that of index.
        :param data:
        [
            [word_11, word_12, ...],
            [word_21, word_22, ...],
            ...
        ]
        """
        assert isinstance(data, list)
        data_index = []
        default_unknown_index = self.word2index[DEFAULT_UNKNOWN_LABEL]
        for example in data:
            data_index.append([self.word2index.get(w, default_unknown_index) for w in example])
        return data_index

    def prepare_output(self, batch_outputs):
        """
        Transform list of batch outputs into strings.
        :param batch_outputs: list of list, of shape [num_batch, tag_seq_length]. Element type is Tensor.
        :return:
        """
        results = []
        for batch in batch_outputs:
            results.append([self.index2label[int(x.data)] for x in batch])
        return results
