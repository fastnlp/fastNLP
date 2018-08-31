import numpy as np
import torch

from fastNLP.core.action import Action
from fastNLP.core.action import RandomSampler, Batchifier
from fastNLP.modules import utils
from fastNLP.saver.logger import create_logger

logger = create_logger(__name__, "./train_test.log")


class BaseTester(object):
    """An collection of model inference and evaluation of performance, used over validation/dev set and test set. """

    def __init__(self, **kwargs):
        """
        :param kwargs: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(BaseTester, self).__init__()
        """
            "default_args" provides default value for important settings. 
            The initialization arguments "kwargs" with the same key (name) will override the default value. 
            "kwargs" must have the same type as "default_args" on corresponding keys. 
            Otherwise, error will raise.
        """
        default_args = {"save_output": False,  # collect outputs of validation set
                        "save_loss": False,  # collect losses in validation
                        "save_best_dev": False,  # save best model during validation
                        "batch_size": 8,
                        "use_cuda": True,
                        "pickle_path": "./save/",
                        "model_name": "dev_best_model.pkl",
                        "print_every_step": 1,
                        }
        """
            "required_args" is the collection of arguments that users must pass to Trainer explicitly. 
            This is used to warn users of essential settings in the training. 
            Obviously, "required_args" is the subset of "default_args". 
            The value in "default_args" to the keys in "required_args" is simply for type check. 
        """
        # TODO: required arguments
        required_args = {}

        for req_key in required_args:
            if req_key not in kwargs:
                logger.error("Tester lacks argument {}".format(req_key))
                raise ValueError("Tester lacks argument {}".format(req_key))

        for key in default_args:
            if key in kwargs:
                if isinstance(kwargs[key], type(default_args[key])):
                    default_args[key] = kwargs[key]
                else:
                    msg = "Argument %s type mismatch: expected %s while get %s" % (
                        key, type(default_args[key]), type(kwargs[key]))
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                # BeseTester doesn't care about extra arguments
                pass
        print(default_args)

        self.save_output = default_args["save_output"]
        self.save_best_dev = default_args["save_best_dev"]
        self.save_loss = default_args["save_loss"]
        self.batch_size = default_args["batch_size"]
        self.pickle_path = default_args["pickle_path"]
        self.use_cuda = default_args["use_cuda"]
        self.print_every_step = default_args["print_every_step"]

        self._model = None
        self.eval_history = []
        self.batch_output = []

    def test(self, network, dev_data):
        if torch.cuda.is_available() and self.use_cuda:
            self._model = network.cuda()
        else:
            self._model = network

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)
        self.eval_history.clear()
        self.batch_output.clear()

        iterator = iter(Batchifier(RandomSampler(dev_data), self.batch_size, drop_last=True))
        step = 0

        for batch_x, batch_y in self.make_batch(iterator, dev_data):
            with torch.no_grad():
                prediction = self.data_forward(network, batch_x)
                eval_results = self.evaluate(prediction, batch_y)

            if self.save_output:
                self.batch_output.append(prediction)
            if self.save_loss:
                self.eval_history.append(eval_results)

            print_output = "[test step {}] {}".format(step, eval_results)
            logger.info(print_output)
            if self.print_every_step > 0 and step % self.print_every_step == 0:
                print(print_output)
            step += 1

    def mode(self, model, test):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param test: bool, whether in test mode.
        """
        Action.mode(model, test)

    def data_forward(self, network, x):
        """A forward pass of the model. """
        raise NotImplementedError

    def evaluate(self, predict, truth):
        """Compute evaluation metrics for the model. """
        raise NotImplementedError

    @property
    def metrics(self):
        """Return a list of metrics. """
        raise NotImplementedError

    def show_matrices(self):
        """This is called by Trainer to print evaluation results on dev set during training.

        :return print_str: str
        """
        raise NotImplementedError

    def make_batch(self, iterator, data):
        raise NotImplementedError


class SeqLabelTester(BaseTester):
    """
    Tester for sequence labeling.
    """

    def __init__(self, **test_args):
        """
        :param test_args: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(SeqLabelTester, self).__init__(**test_args)
        self.max_len = None
        self.mask = None
        self.seq_len = None

    def data_forward(self, network, inputs):
        """This is only for sequence labeling with CRF decoder.

        :param network: a PyTorch model
        :param inputs: tuple of (x, seq_len)
                        x: Tensor of shape [batch_size, max_len], where max_len is the maximum length of the mini-batch
                            after padding.
                        seq_len: list of int, the lengths of sequences before padding.
        :return y: Tensor of shape [batch_size, max_len]
        """
        if not isinstance(inputs, tuple):
            raise RuntimeError("output_length must be true for sequence modeling.")
        # unpack the returned value from make_batch
        x, seq_len = inputs[0], inputs[1]
        batch_size, max_len = x.size(0), x.size(1)
        mask = utils.seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)
        if torch.cuda.is_available() and self.use_cuda:
            mask = mask.cuda()
        self.mask = mask
        self.seq_len = seq_len
        y = network(x)
        return y

    def evaluate(self, predict, truth):
        """Compute metrics (or loss).

        :param predict: Tensor, [batch_size, max_len, tag_size]
        :param truth: Tensor, [batch_size, max_len]
        :return:
        """
        batch_size, max_len = predict.size(0), predict.size(1)
        loss = self._model.loss(predict, truth, self.mask) / batch_size

        prediction = self._model.prediction(predict, self.mask)
        results = torch.Tensor(prediction).view(-1, )
        # make sure "results" is in the same device as "truth"
        results = results.to(truth)
        accuracy = torch.sum(results == truth.view((-1,))).to(torch.float) / results.shape[0]
        return [float(loss), float(accuracy)]

    def metrics(self):
        batch_loss = np.mean([x[0] for x in self.eval_history])
        batch_accuracy = np.mean([x[1] for x in self.eval_history])
        return batch_loss, batch_accuracy

    def show_matrices(self):
        """
        This is called by Trainer to print evaluation on dev set.
        :return print_str: str
        """
        loss, accuracy = self.metrics()
        return "dev loss={:.2f}, accuracy={:.2f}".format(loss, accuracy)

    def make_batch(self, iterator, data):
        return Action.make_batch(iterator, use_cuda=self.use_cuda, output_length=True)


class ClassificationTester(BaseTester):
    """Tester for classification."""

    def __init__(self, **test_args):
        """
        :param test_args: a dict-like object that has __getitem__ method, \
            can be accessed by "test_args["key_str"]"
        """
        super(ClassificationTester, self).__init__(**test_args)

    def make_batch(self, iterator, data, max_len=None):
        return Action.make_batch(iterator, use_cuda=self.use_cuda, max_len=max_len)

    def data_forward(self, network, x):
        """Forward through network."""
        logits = network(x)
        return logits

    def evaluate(self, y_logit, y_true):
        """Return y_pred and y_true."""
        y_prob = torch.nn.functional.softmax(y_logit, dim=-1)
        return [y_prob, y_true]

    def metrics(self):
        """Compute accuracy."""
        y_prob, y_true = zip(*self.eval_history)
        y_prob = torch.cat(y_prob, dim=0)
        y_pred = torch.argmax(y_prob, dim=-1)
        y_true = torch.cat(y_true, dim=0)
        acc = float(torch.sum(y_pred == y_true)) / len(y_true)
        return y_true.cpu().numpy(), y_prob.cpu().numpy(), acc
