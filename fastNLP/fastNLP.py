from fastNLP.action.inference import Inference
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.model_loader import ModelLoader

"""
mapping from model name to [URL, file_name.class_name]
Notice that the class of the model should be in "models" directory.

Example:
    "zh_pos_tag_model": ["www.fudan.edu.cn", "sequence_modeling.SeqLabeling"]
"""
FastNLP_MODEL_COLLECTION = {
    "zh_pos_tag_model": ["www.fudan.edu.cn", "sequence_modeling.SeqLabeling"]
}


class FastNLP(object):
    """
    High-level interface for direct model inference.
    Usage:
        fastnlp = FastNLP()
        fastnlp.load("zh_pos_tag_model")
        text = "这是最好的基于深度学习的中文分词系统。"
        result = fastnlp.run(text)
        print(result)  # ["这", "是", "最好", "的", "基于", "深度学习", "的", "中文", "分词", "系统", "。"]
    """

    def __init__(self, model_dir="./"):
        self.model_dir = model_dir
        self.model = None

    def load(self, model_name):
        """
        Load a pre-trained FastNLP model together with additional data.
        :param model_name: str, the name of a FastNLP model.
        """
        assert type(model_name) is str
        if model_name not in FastNLP_MODEL_COLLECTION:
            raise ValueError("No FastNLP model named {}.".format(model_name))

        if not self.model_exist(model_dir=self.model_dir):
            self._download(model_name, FastNLP_MODEL_COLLECTION[model_name][0])

        model_class = self._get_model_class(FastNLP_MODEL_COLLECTION[model_name][1])

        model_args = ConfigSection()
        # To do: customized config file for model init parameters
        ConfigLoader.load_config(self.model_dir + "default.cfg", model_args)

        model = model_class(model_args)

        # To do: framework independent
        ModelLoader.load_pytorch(model, self.model_dir + model_name)

        self.model = model

        print("Model loaded. ")

    def run(self, infer_input):
        """
        Perform inference over given input using the loaded model.
        :param infer_input: str, raw text
        :return results:
        """
        infer = Inference()
        data = infer.prepare_input(infer_input)
        results = infer.predict(self.model, data)
        return results

    @staticmethod
    def _get_model_class(file_class_name):
        """
        Feature the class specified by <file_class_name>
        :param file_class_name: str, contains the name of the Python module followed by the name of the class.
                Example: "sequence_modeling.SeqLabeling"
        :return module: the model class
        """
        import_prefix = "fastNLP.models."
        parts = (import_prefix + file_class_name).split(".")
        from_module = ".".join(parts[:-1])
        module = __import__(from_module)
        for sub in parts[1:]:
            module = getattr(module, sub)
        return module

    def _load(self, model_dir, model_name):
        # To do
        return 0

    def _download(self, model_name, url):
        """
        Download the model weights from <url> and save in <self.model_dir>.
        :param model_name:
        :param url:
        """
        print("Downloading {} from {}".format(model_name, url))
        # To do

    def model_exist(self, model_dir):
        """
        Check whether the desired model is already in the directory.
        :param model_dir:
        """
        pass
