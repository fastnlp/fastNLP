import pickle
import _pickle
import os

from fastNLP.loader.base_preprocess import BasePreprocess

DEFAULT_PADDING_LABEL = '<pad>'             #dict index = 0
DEFAULT_UNKNOWN_LABEL = '<unk>'             #dict index = 1
DEFAULT_RESERVED_LABEL = ['<reserved-2>',
                          '<reserved-3>',
                          '<reserved-4>']   #dict index = 2~4
#the first vocab in dict with the index = 5



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
        super(POSPreprocess, self).__init(data, pickle_path)
        self.build_dict()
        self.word2id()
        self.id2word()
        self.class2id()
        self.id2class()
        self.embedding()
        self.data_train()
        self.data_dev()
        self.data_test()
        #...


    def build_dict(self):
        self.word_dict = {DEFAULT_PADDING_LABEL: 0, DEFAULT_UNKNOWN_LABEL: 1,
                          DEFAULT_RESERVED_LABEL[0]: 2, DEFAULT_RESERVED_LABEL[1]: 3,
                          DEFAULT_RESERVED_LABEL[2]: 4}
        self.label_dict = {}
        for w in self.data:
            if len(w) == 0:
                continue
            word = w.split('\t')

            if word[0] not in self.word_dict:
                index = len(self.word_dict)
                self.word_dict[word[0]] = index

            for label in word[1: ]:
                if label not in self.label_dict:
                    index = len(self.label_dict)
                    self.label_dict[label] = index


    def pickle_exist(self, pickle_name):
        """
        :param pickle_name: the filename of target pickle file
        :return: True if file exists else False
        """
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        file_name = self.pickle_path + pickle_name
        if os.path.exists(file_name):
            return True
        else:
            return False


    def word2id(self):
        if self.pickle_exist("word2id.pkl"):
            return
        # nothing will be done if word2id.pkl exists

        file_name = self.pickle_path + "word2id.pkl"
        with open(file_name, "wb", encoding='utf-8') as f:
            _pickle.dump(self.word_dict, f)


    def id2word(self):
        if self.pickle_exist("id2word.pkl"):
            return
        #nothing will be done if id2word.pkl exists

        id2word_dict = {}
        for word in self.word_dict:
            id2word_dict[self.word_dict[word]] = word
        file_name = self.pickle_path + "id2word.pkl"
        with open(file_name, "wb", encoding='utf-8') as f:
            _pickle.dump(id2word_dict, f)


    def class2id(self):
        if self.pickle_exist("class2id.pkl"):
            return
        # nothing will be done if class2id.pkl exists

        file_name = self.pickle_path + "class2id.pkl"
        with open(file_name, "wb", encoding='utf-8') as f:
            _pickle.dump(self.label_dict, f)


    def id2class(self):
        if self.pickle_exist("id2class.pkl"):
            return
        #nothing will be done if id2class.pkl exists

        id2class_dict = {}
        for label in self.label_dict:
            id2class_dict[self.label_dict[label]] = label
        file_name = self.pickle_path + "id2class.pkl"
        with open(file_name, "wb", encoding='utf-8') as f:
            _pickle.dump(id2class_dict, f)


    def embedding(self):
        if self.pickle_exist("embedding.pkl"):
            return
        #nothing will be done if embedding.pkl exists


    def data_train(self):
        if self.pickle_exist("data_train.pkl"):
            return
        #nothing will be done if data_train.pkl exists

        data_train = []
        sentence = []
        for w in self.data:
            if len(w) == 0:
                wid = []
                lid = []
                for i in range(len(sentence)):
                    wid.append(self.word_dict[sentence[i][0]])
                    lid.append(self.label_dict[sentence[i][1]])
                data_train.append((wid, lid))
                sentence = []
            sentence.append(w.split('\t'))

        file_name = self.pickle_path + "data_train.pkl"
        with open(file_name, "wb", encoding='utf-8') as f:
            _pickle.dump(data_train, f)

    def data_dev(self):
        pass

    def data_test(self):
        pass
