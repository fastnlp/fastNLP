class Config():
    def __init__(self):
        self.is_training = True
        # path
        self.glove = 'data/glove.840B.300d.txt.filtered'
        self.turian = 'data/turian.50d.txt'
        self.train_path = "data/train.english.jsonlines"
        self.dev_path = "data/dev.english.jsonlines"
        self.test_path = "data/test.english.jsonlines"
        self.char_path = "data/char_vocab.english.txt"

        self.cuda = "0"
        self.max_word = 1500
        self.epoch = 200

        # config
        # self.use_glove = True
        # self.use_turian = True      #No
        self.use_elmo = False
        self.use_CNN = True
        self.model_heads = True     #Yes
        self.use_width = True       # Yes
        self.use_distance = True    #Yes
        self.use_metadata = True   #Yes

        self.mention_ratio = 0.4
        self.max_sentences = 50
        self.span_width = 10
        self.feature_size = 20          #宽度信息emb的size
        self.lr = 0.001
        self.lr_decay = 1e-3
        self.max_antecedents = 100    # 这个参数在mention detection中没有用
        self.atten_hidden_size = 150
        self.mention_hidden_size = 150
        self.sa_hidden_size = 150

        self.char_emb_size = 8
        self.filter = [3,4,5]


    # decay = 1e-5

    def __str__(self):
        d = self.__dict__
        out = 'config==============\n'
        for i in list(d):
            out += i+":"
            out += str(d[i])+"\n"
        out+="config==============\n"
        return out

if __name__=="__main__":
    config = Config()
    print(config)
