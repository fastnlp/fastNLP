from fastNLP.core.inference import SeqLabelInfer, ClassificationInfer
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.model_loader import ModelLoader

"""
mapping from model name to [URL, file_name.class_name, model_pickle_name]
Notice that the class of the model should be in "models" directory.

Example:
    "zh_pos_tag_model": ["www.fudan.edu.cn", "sequence_modeling.SeqLabeling", "saved_model.pkl"]
"""
FastNLP_MODEL_COLLECTION = {
    "seq_label_model": {
        "url": "www.fudan.edu.cn",
        "class": "sequence_modeling.SeqLabeling",
        "pickle": "seq_label_model.pkl",
        "type": "seq_label"
    },
    "text_class_model": {
        "url": "www.fudan.edu.cn",
        "class": "cnn_text_classification.CNNText",
        "pickle": "text_class_model.pkl",
        "type": "text_class"
    }
}

CONFIG_FILE_NAME = "config"
SECTION_NAME = "text_class_model"


class FastNLP(object):
    """
    High-level interface for direct model inference.
    Example Usage:
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
        self.infer_type = None  # "seq_label"/"text_class"

    def load(self, model_name):
        """
        Load a pre-trained FastNLP model together with additional data.
        :param model_name: str, the name of a FastNLP model.
        """
        assert type(model_name) is str
        if model_name not in FastNLP_MODEL_COLLECTION:
            raise ValueError("No FastNLP model named {}.".format(model_name))

        if not self.model_exist(model_dir=self.model_dir):
            self._download(model_name, FastNLP_MODEL_COLLECTION[model_name]["url"])

        model_class = self._get_model_class(FastNLP_MODEL_COLLECTION[model_name]["class"])

        model_args = ConfigSection()
        ConfigLoader.load_config(self.model_dir + CONFIG_FILE_NAME, {SECTION_NAME: model_args})

        # Construct the model
        model = model_class(model_args)

        # To do: framework independent
        ModelLoader.load_pytorch(model, self.model_dir + FastNLP_MODEL_COLLECTION[model_name]["pickle"])

        self.model = model
        self.infer_type = FastNLP_MODEL_COLLECTION[model_name]["type"]

        print("Model loaded. ")

    def run(self, raw_input):
        """
        Perform inference over given input using the loaded model.
        :param raw_input: str, raw text
        :return results:
        """

        infer = self._create_inference(self.model_dir)

        # string ---> 2-D list of string
        infer_input = self.string_to_list(raw_input)

        # 2-D list of string ---> list of strings
        results = infer.predict(self.model, infer_input)

        # list of strings ---> final answers
        outputs = self._make_output(results, infer_input)
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

    def _create_inference(self, model_dir):
        if self.infer_type == "seq_label":
            return SeqLabelInfer(model_dir)
        elif self.infer_type == "text_class":
            return ClassificationInfer(model_dir)
        else:
            raise ValueError("fail to create inference instance")

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
        This function is used to transform raw input to lists, which is done by DatasetLoader in training.
        Split text string into three-level lists.
        [
            [word_11, word_12, ...],
            [word_21, word_22, ...],
            ...
        ]
        :param text: string
        :param delimiter: str, character used to split text into sentences.
        :return data: two-level lists
        """
        data = []
        sents = text.strip().split(delimiter)
        for sent in sents:
            characters = []
            for ch in sent:
                characters.append(ch)
            data.append(characters)
        return data

    def _make_output(self, results, infer_input):
        if self.infer_type == "seq_label":
            outputs = make_seq_label_output(results, infer_input)
        elif self.infer_type == "text_class":
            outputs = make_class_output(results, infer_input)
        else:
            raise ValueError("fail to make outputs with infer type {}".format(self.infer_type))
        return outputs


def make_seq_label_output(result, infer_input):
    """
     Transform model output into user-friendly contents.
    :param result: 1-D list of strings. (model output)
    :param infer_input: 2-D list of string (model input)
    :return outputs:
    """
    return result


def make_class_output(result, infer_input):
    return result


def interpret_word_seg_results(infer_input, results):
    """
    Transform model output into user-friendly contents.
    Example: In CWS, convert <BMES> labeling into segmented text.
    :param results: list of strings. (model output)
    :param infer_input: 2-D list of string (model input)
    :return output: list of strings
    """
    outputs = []
    for sent_char, sent_label in zip(infer_input, results):
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
