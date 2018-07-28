from fastNLP.core.inference import Inference
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.model_loader import ModelLoader

"""
mapping from model name to [URL, file_name.class_name, model_pickle_name]
Notice that the class of the model should be in "models" directory.

Example:
    "zh_pos_tag_model": ["www.fudan.edu.cn", "sequence_modeling.SeqLabeling", "saved_model.pkl"]
"""
FastNLP_MODEL_COLLECTION = {
    "zh_pos_tag_model": ["www.fudan.edu.cn", "sequence_modeling.SeqLabeling", "saved_model.pkl"]
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
        """
        :param model_dir: this directory should contain the following files:
            1. a pre-trained model
            2. a config file
            3. "id2class.pkl"
            4. "word2id.pkl"
        """
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
        ConfigLoader.load_config(self.model_dir + "config", {"POS_infer": model_args})

        # Construct the model
        model = model_class(model_args)

        # To do: framework independent
        ModelLoader.load_pytorch(model, self.model_dir + FastNLP_MODEL_COLLECTION[model_name][2])

        self.model = model

        print("Model loaded. ")

    def run(self, raw_input):
        """
        Perform inference over given input using the loaded model.
        :param raw_input: str, raw text
        :return results:
        """

        infer = Inference(self.model_dir)
        infer_input = self.string_to_list(raw_input)

        results = infer.predict(self.model, infer_input)

        outputs = self.make_output(results)
        return outputs

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
        return True

    def string_to_list(self, text, delimiter="\n"):
        """
        For word seg only, currently.
        This function is used to transform raw input to lists, which is done by DatasetLoader in training.
        Split text string into three-level lists.
        [
            [word_11, word_12, ...],
            [word_21, word_22, ...],
            ...
        ]
        :param text: string
        :param delimiter: str, character used to split text into sentences.
        :return data: three-level lists
        """
        data = []
        sents = text.strip().split(delimiter)
        for sent in sents:
            characters = []
            for ch in sent:
                characters.append(ch)
            data.append(characters)
        # To refactor: this is used in make_output
        self.data = data
        return data

    def make_output(self, results):
        """
        Transform model output into user-friendly contents.
        Example: In CWS, convert <BMES> labeling into segmented text.
        :param results:
        :return:
        """
        outputs = []
        for sent_char, sent_label in zip(self.data, results):
            words = []
            word = ""
            for char, label in zip(sent_char, sent_label):
                if label[0] == "B":
                    if word != "":
                        words.append(word)
                    word = char
                elif label[0] == "M":
                    word += char
                elif label[0] == "E":
                    word += char
                    words.append(word)
                    word = ""
                elif label[0] == "S":
                    if word != "":
                        words.append(word)
                    word = ""
                    words.append(char)
                else:
                    raise ValueError("invalid label")
            outputs.append(" ".join(words))
        return outputs
