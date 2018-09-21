import os

from fastNLP.core.predictor import SeqLabelInfer, ClassificationInfer
from fastNLP.core.preprocess import load_pickle
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.model_loader import ModelLoader

"""
mapping from model name to [URL, file_name.class_name, model_pickle_name]
Notice that the class of the model should be in "models" directory.

Example:
    "seq_label_model": {
        "url": "www.fudan.edu.cn",
        "class": "sequence_modeling.SeqLabeling", # file_name.class_name in models/
        "pickle": "seq_label_model.pkl",
        "type": "seq_label",
        "config_file_name": "config",   # the name of the config file which stores model initialization parameters
        "config_section_name": "text_class_model" # the name of the section in the config file which stores model init params
    },
    "text_class_model": {
        "url": "www.fudan.edu.cn",
        "class": "cnn_text_classification.CNNText",
        "pickle": "text_class_model.pkl",
        "type": "text_class"
    }
"""
FastNLP_MODEL_COLLECTION = {
    "cws_basic_model": {
        "url": "",
        "class": "sequence_modeling.AdvSeqLabel",
        "pickle": "cws_basic_model_v_0.pkl",
        "type": "seq_label",
        "config_file_name": "config",
        "config_section_name": "text_class_model"
    },
    "pos_tag_model": {
        "url": "",
        "class": "sequence_modeling.AdvSeqLabel",
        "pickle": "pos_tag_model_v_0.pkl",
        "type": "seq_label",
        "config_file_name": "pos_tag.config",
        "config_section_name": "pos_tag_model"
    },
    "text_classify_model": {
        "url": "",
        "class": "cnn_text_classification.CNNText",
        "pickle": "text_class_model_v0.pkl",
        "type": "text_class",
        "config_file_name": "text_classify.cfg",
        "config_section_name": "model"
    }
}


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
            3. "class2id.pkl"
            4. "word2id.pkl"
        """
        self.model_dir = model_dir
        self.model = None
        self.infer_type = None  # "seq_label"/"text_class"

    def load(self, model_name, config_file="config", section_name="model"):
        """
        Load a pre-trained FastNLP model together with additional data.
        :param model_name: str, the name of a FastNLP model.
        :param config_file: str, the name of the config file which stores the initialization information of the model.
                (default: "config")
        :param section_name: str, the name of the corresponding section in the config file. (default: model)
        """
        assert type(model_name) is str
        if model_name not in FastNLP_MODEL_COLLECTION:
            raise ValueError("No FastNLP model named {}.".format(model_name))

        if not self.model_exist(model_dir=self.model_dir):
            self._download(model_name, FastNLP_MODEL_COLLECTION[model_name]["url"])

        model_class = self._get_model_class(FastNLP_MODEL_COLLECTION[model_name]["class"])
        print("Restore model class {}".format(str(model_class)))

        model_args = ConfigSection()
        ConfigLoader.load_config(os.path.join(self.model_dir, config_file), {section_name: model_args})
        print("Restore model hyper-parameters {}".format(str(model_args.data)))

        # fetch dictionary size and number of labels from pickle files
        word_vocab = load_pickle(self.model_dir, "word2id.pkl")
        model_args["vocab_size"] = len(word_vocab)
        label_vocab = load_pickle(self.model_dir, "class2id.pkl")
        model_args["num_classes"] = len(label_vocab)

        # Construct the model
        model = model_class(model_args)
        print("Model constructed.")

        # To do: framework independent
        ModelLoader.load_pytorch(model, os.path.join(self.model_dir, FastNLP_MODEL_COLLECTION[model_name]["pickle"]))
        print("Model weights loaded.")

        self.model = model
        self.infer_type = FastNLP_MODEL_COLLECTION[model_name]["type"]

        print("Inference ready.")

    def run(self, raw_input):
        """
        Perform inference over given input using the loaded model.
        :param raw_input: list of string. Each list is an input query.
        :return results:
        """

        infer = self._create_inference(self.model_dir)

        # tokenize: list of string ---> 2-D list of string
        infer_input = self.tokenize(raw_input, language="zh")

        # 2-D list of string ---> 2-D list of tags
        results = infer.predict(self.model, infer_input)

        # 2-D list of tags ---> list of final answers
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

    def tokenize(self, text, language):
        """Extract tokens from strings.
        For English, extract words separated by space.
        For Chinese, extract characters.
        TODO: more complex tokenization methods

        :param text: list of string
        :param language: str, one of ('zh', 'en'), Chinese or English.
        :return data: list of list of string, each string is a token.
        """
        assert language in ("zh", "en")
        data = []
        for sent in text:
            if language == "en":
                tokens = sent.strip().split()
            elif language == "zh":
                tokens = [char for char in sent]
            else:
                raise RuntimeError("Unknown language {}".format(language))
            data.append(tokens)
        return data

    def _make_output(self, results, infer_input):
        """Transform the infer output into user-friendly output.

        :param results: 1 or 2-D list of strings.
                If self.infer_type == "seq_label", it is of shape [num_examples, tag_seq_length]
                If self.infer_type == "text_class", it is of shape [num_examples]
        :param infer_input: 2-D list of string, the input query before inference.
        :return outputs: list. Each entry is a prediction.
        """
        if self.infer_type == "seq_label":
            outputs = make_seq_label_output(results, infer_input)
        elif self.infer_type == "text_class":
            outputs = make_class_output(results, infer_input)
        else:
            raise RuntimeError("fail to make outputs with infer type {}".format(self.infer_type))
        return outputs


def make_seq_label_output(result, infer_input):
    """Transform model output into user-friendly contents.

    :param result: 2-D list of strings. (model output)
    :param infer_input: 2-D list of string (model input)
    :return ret: list of list of tuples
        [
            [(word_11, label_11), (word_12, label_12), ...],
            [(word_21, label_21), (word_22, label_22), ...],
            ...
        ]
    """
    ret = []
    for example_x, example_y in zip(infer_input, result):
        ret.append([(x, y) for x, y in zip(example_x, example_y)])
    return ret

def make_class_output(result, infer_input):
    """Transform model output into user-friendly contents.

    :param result: 2-D list of strings. (model output)
    :param infer_input: 1-D list of string (model input)
    :return ret: the same as result, [label_1, label_2, ...]
    """
    return result


def interpret_word_seg_results(char_seq, label_seq):
    """Transform model output into user-friendly contents.

    Example: In CWS, convert <BMES> labeling into segmented text.
    :param char_seq: list of string,
    :param label_seq: list of string, the same length as char_seq
            Each entry is one of ('B', 'M', 'E', 'S').
    :return output: list of words
    """
    words = []
    word = ""
    for char, label in zip(char_seq, label_seq):
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
            raise ValueError("invalid label {}".format(label[0]))
    return words


def interpret_cws_pos_results(char_seq, label_seq):
    """Transform model output into user-friendly contents.

    :param char_seq: list of string
    :param label_seq: list of string, the same length as char_seq.
    :return outputs: list of tuple (words, pos_tag):
    """

    def pos_tag_check(seq):
        """check whether all entries are the same """
        return len(set(seq)) <= 1

    word = []
    word_pos = []
    outputs = []
    for char, label in zip(char_seq, label_seq):
        tmp = label.split("-")
        cws_label, pos_tag = tmp[0], tmp[1]

        if cws_label == "B" or cws_label == "M":
            word.append(char)
            word_pos.append(pos_tag)
        elif cws_label == "E":
            word.append(char)
            word_pos.append(pos_tag)
            if not pos_tag_check(word_pos):
                raise RuntimeError("character-wise pos tags inconsistent. ")
            outputs.append(("".join(word), word_pos[0]))
            word.clear()
            word_pos.clear()
        elif cws_label == "S":
            outputs.append((char, pos_tag))
    return outputs
