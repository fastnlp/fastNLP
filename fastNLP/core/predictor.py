import numpy as np
import torch

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import create_dataset_from_lists
from fastNLP.core.preprocess import load_pickle
from fastNLP.core.sampler import SequentialSampler


class Predictor(object):
    """An interface for predicting outputs based on trained models.

    It does not care about evaluations of the model, which is different from Tester.
    This is a high-level model wrapper to be called by FastNLP.
    This class does not share any operations with Trainer and Tester.
    Currently, Predictor does not support GPU.
    """

    def __init__(self, pickle_path, post_processor):
        """

        :param pickle_path: str, the path to the pickle files.
        :param post_processor: a function or callable object, that takes list of batch outputs as input

        """
        self.batch_size = 1
        self.batch_output = []
        self.pickle_path = pickle_path
        self._post_processor = post_processor
        self.label_vocab = load_pickle(self.pickle_path, "label2id.pkl")
        self.word_vocab = load_pickle(self.pickle_path, "word2id.pkl")

    def predict(self, network, data):
        """Perform inference using the trained model.

        :param network: a PyTorch model (cpu)
        :param data: a DataSet object.
        :return: list of list of strings, [num_examples, tag_seq_length]
        """
        # transform strings into DataSet object
        # data = self.prepare_input(data)

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)
        batch_output = []

        data_iterator = Batch(data, batch_size=self.batch_size, sampler=SequentialSampler(), use_cuda=False)

        for batch_x, _ in data_iterator:
            with torch.no_grad():
                prediction = self.data_forward(network, batch_x)
            batch_output.append(prediction)

        return self._post_processor(batch_output, self.label_vocab)

    def mode(self, network, test=True):
        if test:
            network.eval()
        else:
            network.train()

    def data_forward(self, network, x):
        """Forward through network."""
        y = network(**x)
        return y

    def prepare_input(self, data):
        """Transform two-level list of strings into an DataSet object.
        In the training pipeline, this is done by Preprocessor. But in inference time, we do not call Preprocessor.

        :param data: list of list of strings.
                ::
                [
                    [word_11, word_12, ...],
                    [word_21, word_22, ...],
                    ...
                ]

        :return data_set: a DataSet instance.
        """
        assert isinstance(data, list)
        return create_dataset_from_lists(data, self.word_vocab, has_target=False)


class SeqLabelInfer(Predictor):
    def __init__(self, pickle_path):
        print(
            "[FastNLP Warning] SeqLabelInfer will be deprecated. Please use Predictor directly.")
        super(SeqLabelInfer, self).__init__(pickle_path, seq_label_post_processor)


class ClassificationInfer(Predictor):
    def __init__(self, pickle_path):
        print(
            "[FastNLP Warning] ClassificationInfer will be deprecated. Please use Predictor directly.")
        super(ClassificationInfer, self).__init__(pickle_path, text_classify_post_processor)


def seq_label_post_processor(batch_outputs, label_vocab):
    results = []
    for batch in batch_outputs:
        for example in np.array(batch):
            results.append([label_vocab.to_word(int(x)) for x in example])
    return results


def text_classify_post_processor(batch_outputs, label_vocab):
    results = []
    for batch_out in batch_outputs:
        idx = np.argmax(batch_out.detach().numpy(), axis=-1)
        results.extend([label_vocab.to_word(i) for i in idx])
    return results
