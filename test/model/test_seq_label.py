import os

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.loader.dataset_loader import TokenizeDataSetLoader
from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.preprocess import save_pickle
from fastNLP.core.tester import SeqLabelTester
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.models.sequence_modeling import SeqLabeling
from fastNLP.saver.model_saver import ModelSaver

pickle_path = "./seq_label/"
model_name = "seq_label_model.pkl"
config_dir = "test/data_for_tests/config"
data_path = "test/data_for_tests/people.txt"
data_infer_path = "test/data_for_tests/people_infer.txt"


def test_training():
    # Config Loader
    trainer_args = ConfigSection()
    model_args = ConfigSection()
    ConfigLoader().load_config(config_dir, {
        "test_seq_label_trainer": trainer_args, "test_seq_label_model": model_args})

    data_set = TokenizeDataSetLoader().load(data_path)
    word_vocab = Vocabulary()
    label_vocab = Vocabulary()
    data_set.update_vocab(word_seq=word_vocab, label_seq=label_vocab)
    data_set.index_field("word_seq", word_vocab).index_field("label_seq", label_vocab)
    data_set.set_origin_len("word_seq")
    data_set.rename_field("label_seq", "truth").set_target(truth=False)
    data_train, data_dev = data_set.split(0.3, shuffle=True)
    model_args["vocab_size"] = len(word_vocab)
    model_args["num_classes"] = len(label_vocab)

    save_pickle(word_vocab, pickle_path, "word2id.pkl")
    save_pickle(label_vocab, pickle_path, "label2id.pkl")

    trainer = SeqLabelTrainer(
        epochs=trainer_args["epochs"],
        batch_size=trainer_args["batch_size"],
        validate=False,
        use_cuda=False,
        pickle_path=pickle_path,
        save_best_dev=trainer_args["save_best_dev"],
        model_name=model_name,
        optimizer=Optimizer("SGD", lr=0.01, momentum=0.9),
    )

    # Model
    model = SeqLabeling(model_args)

    # Start training
    trainer.train(model, data_train, data_dev)

    # Saver
    saver = ModelSaver(os.path.join(pickle_path, model_name))
    saver.save_pytorch(model)

    del model, trainer

    # Define the same model
    model = SeqLabeling(model_args)

    # Dump trained parameters into the model
    ModelLoader.load_pytorch(model, os.path.join(pickle_path, model_name))

    # Load test configuration
    tester_args = ConfigSection()
    ConfigLoader().load_config(config_dir, {"test_seq_label_tester": tester_args})

    # Tester
    tester = SeqLabelTester(batch_size=4,
                            use_cuda=False,
                            pickle_path=pickle_path,
                            model_name="seq_label_in_test.pkl",
                            evaluator=SeqLabelEvaluator()
                            )

    # Start testing with validation data
    data_dev.set_target(truth=True)
    tester.test(model, data_dev)
