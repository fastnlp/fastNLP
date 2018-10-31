import torch

from fastNLP.core.batch import Batch
from fastNLP.core.metrics import Evaluator
from fastNLP.core.sampler import RandomSampler
from fastNLP.saver.logger import create_logger

logger = create_logger(__name__, "./train_test.log")


class Tester(object):
    """An collection of model inference and evaluation of performance, used over validation/dev set and test set. """

    def __init__(self, **kwargs):
        """
        :param kwargs: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(Tester, self).__init__()
        """
            "default_args" provides default value for important settings.
            The initialization arguments "kwargs" with the same key (name) will override the default value.
            "kwargs" must have the same type as "default_args" on corresponding keys.
            Otherwise, error will raise.
        """
        default_args = {"batch_size": 8,
                        "use_cuda": False,
                        "pickle_path": "./save/",
                        "model_name": "dev_best_model.pkl",
                        "evaluator": Evaluator()
                        }
        """
            "required_args" is the collection of arguments that users must pass to Trainer explicitly.
            This is used to warn users of essential settings in the training.
            Specially, "required_args" does not have default value, so they have nothing to do with "default_args".
        """
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
                # Tester doesn't care about extra arguments
                pass
        print(default_args)

        self.batch_size = default_args["batch_size"]
        self.pickle_path = default_args["pickle_path"]
        self.use_cuda = default_args["use_cuda"]
        self._evaluator = default_args["evaluator"]

        self._model = None
        self.eval_history = []  # evaluation results of all batches

    def test(self, network, dev_data):
        if torch.cuda.is_available() and self.use_cuda:
            self._model = network.cuda()
        else:
            self._model = network

        # turn on the testing mode; clean up the history
        self.mode(network, is_test=True)
        self.eval_history.clear()
        output_list = []
        truth_list = []

        data_iterator = Batch(dev_data, self.batch_size, sampler=RandomSampler(), use_cuda=self.use_cuda)

        with torch.no_grad():
            for batch_x, batch_y in data_iterator:
                prediction = self.data_forward(network, batch_x)
                output_list.append(prediction)
                truth_list.append(batch_y)
            eval_results = self.evaluate(output_list, truth_list)
        print("[tester] {}".format(self.print_eval_results(eval_results)))
        logger.info("[tester] {}".format(self.print_eval_results(eval_results)))
        self.mode(network, is_test=False)
        self.metrics = eval_results
        return eval_results

    def mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def data_forward(self, network, x):
        """A forward pass of the model. """
        y = network(**x)
        return y

    def evaluate(self, predict, truth):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :return eval_results: can be anything. It will be stored in self.eval_history
        """
        return self._evaluator(predict, truth)

    def print_eval_results(self, results):
        """Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        return ", ".join([str(key) + "=" + str(value) for key, value in results.items()])


class SeqLabelTester(Tester):
    def __init__(self, **test_args):
        print(
            "[FastNLP Warning] SeqLabelTester will be deprecated. Please use Tester directly.")
        super(SeqLabelTester, self).__init__(**test_args)


class ClassificationTester(Tester):
    def __init__(self, **test_args):
        print(
            "[FastNLP Warning] ClassificationTester will be deprecated. Please use Tester directly.")
        super(ClassificationTester, self).__init__(**test_args)


class SNLITester(Tester):
    def __init__(self, **test_args):
        print(
            "[FastNLP Warning] SNLITester will be deprecated. Please use Tester directly.")
        super(SNLITester, self).__init__(**test_args)
