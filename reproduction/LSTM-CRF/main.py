from process import *


def prepare_data():
    ## load the data from the textfile
    datasets = load_data(Conll2003Loader(), \[
                    "/remote-home/nndl/data/CONLL2003/train.txt",
                    "/remote-home/nndl/data/CONLL2003/valid.txt",
                    "/remote-home/nndl/data/CONLL2003/test.txt"
                  ])
    train_data = datasets[0]
    valid_data = datasets[1]
    test_data = datasets[2]
    
    #Lower case the words in the sentences
    loser_case([train_data, valid_data, test_data], "token_list")
    
    ## Build vocab
    vocab = build_vocab([train_data, valid_data, test_data], "token_list")
    speech_vocab = build_vocab([train_data, valid_data, test_data], "label0_list")
    
    ## Build index
    build_index([train_data, valid_data, test_data], "token_list", 'token_index_list', vocab)
    build_index([train_data, valid_data, test_data], "label0_list", 'speech_index_list', speech_vocab)
    
    return train_data, valid_data, test_data, vocab, speech_vocab


def workflow():
    
    train_data, valid_data, test_data, vocab, speech_vocab = prepare_data()
    
    config = {
        "vocab_size": len(vocab),
        "word_emb_dim": 200, 
        "rnn_hidden_units": 600,
        "num_classes": len(speech_vocab),
        "bi_direction": True
    }
    
    ## Set the corresponding tags for each dataset, which will be used in the Trainer
    train_data.set_input("token_index_list", "speech_index_list")
    test_data.set_input("token_index_list", "speech_index_list")
    valid_data.set_input("token_index_list")

    train_data.set_target("speech_index_list")
    test_data.set_target("speech_index_list")
    valid_data.set_target("speech_index_list")
    
    ## Build the model
    model = BiLSTMCRF(config)
    
    
    ## Train the model
    trainer = Trainer(
        model=model, 
        train_data=train_data, 
        dev_data=valid_data,
        use_cuda=True,
        metrics=PosMetric(pred='pred', target='speech_index_list'),
        optimizer=SGD(lr=0.1),
        n_epochs=100, 
        batch_size=1000,
        save_path="./"
    )
    trainer.train()
    
    ## Test the model
    tester = Tester(data=test_data, 
                  model=model, 
                  metrics=PosMetric(pred='pred', target='speech_index_list')
           )
    tester.test()



if __name__ == "__main__":
    workflow()
    