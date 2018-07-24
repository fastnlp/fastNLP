class Inference(object):
    """
    This is an interface focusing on predicting output based on trained models.
    It does not care about evaluations of the model.

    """

    def __init__(self):
        pass

    def predict(self, model, data):
        """
        this is actually a forward pass. shall be shared by Trainer/Tester
        :param model:
        :param data:
        :return result: the output results
        """
        raise NotImplementedError

    def prepare_input(self, data_path):
        """
        This can also be shared.
        :param data_path:
        :return:
        """
        raise NotImplementedError
