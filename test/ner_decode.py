import _pickle
import os

import torch

from fastNLP.core.predictor import SeqLabelInfer
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.models.sequence_modeling import AdvSeqLabel


class Decode(SeqLabelTrainer):
    def __init__(self, args):
        super(Decode, self).__init__(args)

    def decoder(self, network, sents, model_path):
        self.model = network
        self.model.load_state_dict(torch.load(model_path))
        out_put = []
        self.mode(network, test=True)
        for batch_x in sents:
            prediction = self.data_forward(self.model, batch_x)

            seq_tag = self.model.prediction(prediction, batch_x[1])

            out_put.append(list(seq_tag)[0])
        return out_put


def process_sent(sents, word2id):
    sents_num = []
    for s in sents:
        sent_num = []
        for c in s:
            if c in word2id:
                sent_num.append(word2id[c])
            else:
                sent_num.append(word2id["<unk>"])
        sents_num.append(([sent_num], [len(sent_num)]))  # batch_size is 1

    return sents_num


def process_tag(sents, tags, id2class):
    Tags = []
    for ttt in tags:
        Tags.append([id2class[t] for t in ttt])

    Segs = []
    PosNers = []
    for sent, tag in zip(sents, tags):
        word__ = []
        lll__ = []
        for c, t in zip(sent, tag):

            t = id2class[t]
            l = t.split("-")
            split_ = l[0]
            pn = l[1]

            if split_ == "S":
                word__.append(c)
                lll__.append(pn)
                word_1 = ""
            elif split_ == "E":
                word_1 += c
                word__.append(word_1)
                lll__.append(pn)
                word_1 = ""
            elif split_ == "B":
                word_1 = ""
                word_1 += c
            else:
                word_1 += c
        Segs.append(word__)
        PosNers.append(lll__)
    return Segs, PosNers


pickle_path = "data_for_tests/"
model_path = "data_for_tests/model_best_dev.pkl"
if __name__ == "__main__":

    with open(os.path.join(pickle_path, "id2word.pkl"), "rb") as f:
        id2word = _pickle.load(f)
    with open(os.path.join(pickle_path, "word2id.pkl"), "rb") as f:
        word2id = _pickle.load(f)
    with open(os.path.join(pickle_path, "id2class.pkl"), "rb") as f:
        id2class = _pickle.load(f)

    sent = ["中共中央总书记、国家主席江泽民",
            "逆向处理输入序列并返回逆序后的序列"]  # here is input

    args = {"epochs": 1,
            "batch_size": 1,
            "pickle_path": "data_for_tests/",
            "validate": True,
            "save_best_dev": True,
            "model_saved_path": "data_for_tests/",
            "use_cuda": False,

            "vocab_size": len(word2id),
            "num_classes": len(id2class),
            "word_emb_dim": 50,
            "rnn_hidden_units": 100,
            }
    """
    network = AdvSeqLabel(args, None)
    decoder_ = Decode(args)
    tags_num = decoder_.decoder(network, process_sent(sent, word2id), model_path=model_path)
    output_seg, output_pn = process_tag(sent, tags_num, id2class)  # here is output
    print(output_seg)
    print(output_pn)
    """
    # Define the same model
    model = AdvSeqLabel(args, None)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, "./data_for_tests/model_best_dev.pkl")
    print("model loaded!")

    # Inference interface
    infer = SeqLabelInfer(pickle_path)
    sent = [[ch for ch in s] for s in sent]
    results = infer.predict(model, sent)

    for res in results:
        print(res)
    print("Inference finished!")
