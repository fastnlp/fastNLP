import _pickle
import os

DEFAULT_PADDING_LABEL = '<pad>'  # dict index = 0
DEFAULT_UNKNOWN_LABEL = '<unk>'  # dict index = 1
DEFAULT_RESERVED_LABEL = ['<reserved-2>',
                          '<reserved-3>',
                          '<reserved-4>']  # dict index = 2~4


# the first vocab in dict with the index = 5


class BasePreprocess(object):

    def __init__(self, data, pickle_path):
        super(BasePreprocess, self).__init__()
        self.data = data
        self.pickle_path = pickle_path
        if not self.pickle_path.endswith('/'):
            self.pickle_path = self.pickle_path + '/'


class POSPreprocess(BasePreprocess):
    """
        This class are used to preprocess the pos datasets.
        In these datasets, each line are divided by '\t'
    while the first Col is the vocabulary and the second
    Col is the label.
        Different sentence are divided by an empty line.
        e.g:
        Tom label1
        and label2
        Jerry   label1
        .   label3
        Hello   label4
        world   label5
        !   label3
        In this file, there are two sentence "Tom and Jerry ."
    and "Hello world !". Each word has its own label from label1
    to label5.
    """


    def __init__(self, data, pickle_path):
        super(POSPreprocess, self).__init__(data, pickle_path)
        self.word_dict = {DEFAULT_PADDING_LABEL: 0, DEFAULT_UNKNOWN_LABEL: 1,
                          DEFAULT_RESERVED_LABEL[0]: 2, DEFAULT_RESERVED_LABEL[1]: 3,
                          DEFAULT_RESERVED_LABEL[2]: 4}

        self.label_dict = None
        self.data = data
        self.pickle_path = pickle_path

        self.build_dict(data)
        if not self.pickle_exist("word2id.pkl"):
            self.word_dict.update(self.word2id(data))
            file_name = os.path.join(self.pickle_path, "word2id.pkl")
            with open(file_name, "wb") as f:
                _pickle.dump(self.word_dict, f)

        self.vocab_size = self.id2word()
        self.class2id()
        self.num_classes = self.id2class()
        self.embedding()
        self.data_train()
        self.data_dev()
        self.data_test()

    def build_dict(self, data):
        """
        Add new words with indices into self.word_dict, new labels with indices into self.label_dict.
        :param data: list of list [word, label]
        """

        self.label_dict = {}
        for line in data:
            line = line.strip()
            if len(line) <= 1:
                continue
            tokens = line.split('\t')

            if tokens[0] not in self.word_dict:
                # add (word, index) into the dict
                self.word_dict[tokens[0]] = len(self.word_dict)

            # for label in tokens[1: ]:
            if tokens[1] not in self.label_dict:
                self.label_dict[tokens[1]] = len(self.label_dict)

    def pickle_exist(self, pickle_name):
        """
        :param pickle_name: the filename of target pickle file
        :return: True if file exists else False
        """
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        file_name = os.path.join(self.pickle_path, pickle_name)
        if os.path.exists(file_name):
            return True
        else:
            return False

    def word2id(self):
        if self.pickle_exist("word2id.pkl"):
            return
        # nothing will be done if word2id.pkl exists

        file_name = os.path.join(self.pickle_path, "word2id.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(self.word_dict, f)

    def id2word(self):
        if self.pickle_exist("id2word.pkl"):
            file_name = os.path.join(self.pickle_path, "id2word.pkl")
            id2word_dict = _pickle.load(open(file_name, "rb"))
            return len(id2word_dict)
        # nothing will be done if id2word.pkl exists

        id2word_dict = {}
        for word in self.word_dict:
            id2word_dict[self.word_dict[word]] = word
        file_name = os.path.join(self.pickle_path, "id2word.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(id2word_dict, f)
        return len(id2word_dict)

    def class2id(self):
        if self.pickle_exist("class2id.pkl"):
            return
        # nothing will be done if class2id.pkl exists

        file_name = os.path.join(self.pickle_path, "class2id.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(self.label_dict, f)

    def id2class(self):
        if self.pickle_exist("id2class.pkl"):
            file_name = os.path.join(self.pickle_path, "id2class.pkl")
            id2class_dict = _pickle.load(open(file_name, "rb"))
            return len(id2class_dict)
        # nothing will be done if id2class.pkl exists

        id2class_dict = {}
        for label in self.label_dict:
            id2class_dict[self.label_dict[label]] = label
        file_name = os.path.join(self.pickle_path, "id2class.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(id2class_dict, f)
        return len(id2class_dict)

    def embedding(self):
        if self.pickle_exist("embedding.pkl"):
            return
        # nothing will be done if embedding.pkl exists

    def data_train(self):
        if self.pickle_exist("data_train.pkl"):
            return
        # nothing will be done if data_train.pkl exists

        data_train = []
        sentence = []
        for w in self.data:
            w = w.strip()
            if len(w) <= 1:
                wid = []
                lid = []
                for i in range(len(sentence)):
                    # if sentence[i][0]=="":
                    #     print("")
                    wid.append(self.word_dict[sentence[i][0]])
                    lid.append(self.label_dict[sentence[i][1]])
                data_train.append((wid, lid))
                sentence = []
                continue
            sentence.append(w.split('\t'))

        file_name = os.path.join(self.pickle_path, "data_train.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(data_train, f)

    def data_dev(self):
        pass

    def data_test(self):
        pass


class ClassPreprocess(BasePreprocess):
    """
    Pre-process the classification datasets.

    Params:
        pickle_path - directory to save result of pre-processing
    Saves:
        word2id.pkl
        id2word.pkl
        class2id.pkl
        id2class.pkl
        embedding.pkl
        data_train.pkl
        data_dev.pkl
        data_test.pkl
    """

    def __init__(self, pickle_path):
        # super(ClassPreprocess, self).__init__(data, pickle_path)
        self.word_dict = None
        self.label_dict = None
        self.pickle_path = pickle_path  # save directory

    def process(self, data, save_name):
        """
        Process data.

        Params:
            data - nested list, data = [sample1, sample2, ...],
                sample = [sentence, label], sentence = [word1, word2, ...]
            save_name - name of processed data, such as data_train.pkl
        Returns:
            vocab_size - vocabulary size
            n_classes - number of classes
        """
        self.build_dict(data)
        self.word2id()
        vocab_size = self.id2word()
        self.class2id()
        num_classes = self.id2class()
        self.embedding()
        self.data_generate(data, save_name)

        return vocab_size, num_classes

    def build_dict(self, data):
        """Build vocabulary."""

        # just read if word2id.pkl and class2id.pkl exists
        if self.pickle_exist("word2id.pkl") and \
                self.pickle_exist("class2id.pkl"):
            file_name = os.path.join(self.pickle_path, "word2id.pkl")
            with open(file_name, 'rb') as f:
                self.word_dict = _pickle.load(f)
            file_name = os.path.join(self.pickle_path, "class2id.pkl")
            with open(file_name, 'rb') as f:
                self.label_dict = _pickle.load(f)
            return

        # build vocabulary from scratch if nothing exists
        self.word_dict = {
            DEFAULT_PADDING_LABEL: 0,
            DEFAULT_UNKNOWN_LABEL: 1,
            DEFAULT_RESERVED_LABEL[0]: 2,
            DEFAULT_RESERVED_LABEL[1]: 3,
            DEFAULT_RESERVED_LABEL[2]: 4}
        self.label_dict = {}

        # collect every word and label
        for sent, label in data:
            if len(sent) <= 1:
                continue

            if label not in self.label_dict:
                index = len(self.label_dict)
                self.label_dict[label] = index

            for word in sent:
                if word not in self.word_dict:
                    index = len(self.word_dict)
                    self.word_dict[word[0]] = index

    def pickle_exist(self, pickle_name):
        """
        Check whether a pickle file exists.

        Params
            pickle_name: the filename of target pickle file
        Return
            True if file exists else False
        """
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        file_name = os.path.join(self.pickle_path, pickle_name)
        if os.path.exists(file_name):
            return True
        else:
            return False

    def word2id(self):
        """Save vocabulary of {word:id} mapping format."""
        # nothing will be done if word2id.pkl exists
        if self.pickle_exist("word2id.pkl"):
            return

        file_name = os.path.join(self.pickle_path, "word2id.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(self.word_dict, f)

    def id2word(self):
        """Save vocabulary of {id:word} mapping format."""
        # nothing will be done if id2word.pkl exists
        if self.pickle_exist("id2word.pkl"):
            file_name = os.path.join(self.pickle_path, "id2word.pkl")
            with open(file_name, 'rb') as f:
                id2word_dict = _pickle.load(f)
            return len(id2word_dict)

        id2word_dict = {self.word_dict[w]: w for w in self.word_dict}
        file_name = os.path.join(self.pickle_path, "id2word.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(id2word_dict, f)
        return len(id2word_dict)

    def class2id(self):
        """Save mapping of {class:id}."""
        # nothing will be done if class2id.pkl exists
        if self.pickle_exist("class2id.pkl"):
            return

        file_name = os.path.join(self.pickle_path, "class2id.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(self.label_dict, f)

    def id2class(self):
        """Save mapping of {id:class}."""
        # nothing will be done if id2class.pkl exists
        if self.pickle_exist("id2class.pkl"):
            file_name = os.path.join(self.pickle_path, "id2class.pkl")
            with open(file_name, "rb") as f:
                id2class_dict = _pickle.load(f)
            return len(id2class_dict)

        id2class_dict = {self.label_dict[c]: c for c in self.label_dict}
        file_name = os.path.join(self.pickle_path, "id2class.pkl")
        with open(file_name, "wb") as f:
            _pickle.dump(id2class_dict, f)
        return len(id2class_dict)

    def embedding(self):
        """Save embedding lookup table corresponding to vocabulary."""
        # nothing will be done if embedding.pkl exists
        if self.pickle_exist("embedding.pkl"):
            return

        # retrieve vocabulary from pre-trained embedding (not implemented)

    def data_generate(self, data_src, save_name):
        """Convert dataset from text to digit."""

        # nothing will be done if file exists
        save_path = os.path.join(self.pickle_path, save_name)
        if os.path.exists(save_path):
            return

        data = []
        # for every sample
        for sent, label in data_src:
            if len(sent) <= 1:
                continue

            label_id = self.label_dict[label]  # label id
            sent_id = []  # sentence ids
            for word in sent:
                if word in self.word_dict:
                    sent_id.append(self.word_dict[word])
                else:
                    sent_id.append(self.word_dict[DEFAULT_UNKNOWN_LABEL])
            data.append([sent_id, label_id])

        # save data
        with open(save_path, "wb") as f:
            _pickle.dump(data, f)


class LMPreprocess(BasePreprocess):
    def __init__(self, data, pickle_path):
        super(LMPreprocess, self).__init__(data, pickle_path)
