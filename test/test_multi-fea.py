import sys
sys.path.append("../")
import codecs
from fastNLP.models.event_extraction import Argument_extraction
from fastNLP.core.preprocess import MultiFeaSeqpreprocess
from fastNLP.core.trainer import AdvSeqLabelTrainer
def data_load(data_file):
    """
    :param data_file:
    :return: [[sent, label,seq_fea,...,single_fea,...],...]
    """
    with codecs.open(data_file, "r", encoding="utf-8") as f:
        all_data = []
        sent = []
        label = []
        en_ph=[]
        tr_ph=[]
        i=0
        for line in f:
            context=line
            line = line.strip().split("\t")
            if len(line) == 4:
                sent.append(line[0])
                label.append(line[1])
                en_ph.append(line[2])
                tr_ph.append(int(line[3]))
                if int(line[3])==-1:
                    print(line,i)
            elif len(line)==1:

                if sent and label and en_ph and tr_ph:
                    all_data.append([sent, label, en_ph, tr_ph[0]])
                sent = []
                label = []
                en_ph = []
                tr_ph = []
            else:
                print("wrong",context,i)
            i+=1
    return all_data

data_path = "test/data_for_tests/argument.txt"
pick_path = "test/data_for_tests/"


if __name__ == "__main__":
    data = data_load(data_path)
    preprocess = MultiFeaSeqpreprocess()
    print(len(data))
    data_train,data_dev= preprocess.run(data, pickle_path=pick_path,train_dev_split=0.1)

    print("train",len(data_train))
    dev=data_load("data/argument_dev.txt")
    data_dev=preprocess.to_index(dev)
    print("dev",len(data_dev))
    # if os.path.exists(os.path.join(pick_path, "embedding.pkl")):
    #     emb=load_pickle(pick_path,"embedding.pkl")
    # else:
    #     emb = embedding_process(emb_path, preprocess.all_dicts["sent"], 50, os.path.join(pick_path, "embedding.pkl"))

    args = {
            "vocab_size": len(preprocess.all_dicts["sent"]),
            "num_classes": 2,
            "word_emb_dim": 50,
            "rnn_hidden_units": 50
            }
    print(preprocess.label2index)
    emb = None#torch.Tensor(emb).float().cuda()
    networks = Argument_extraction(args, emb)
    trainer = AdvSeqLabelTrainer(epochs=20,batch_size=256,validate=True,pickle_path=pick_path,model_name="2liner.pkl")
    trainer.train(networks, data_train, data_dev)
    print("Training finished!")
