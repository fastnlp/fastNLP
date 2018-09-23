from fastNLP.core.loss import Loss
from fastNLP.core.preprocess import Preprocessor
from fastNLP.core.trainer import Trainer
from fastNLP.loader.dataset_loader import LMDatasetLoader
from fastNLP.models.char_language_model import CharLM

PICKLE = "./save/"


def train():
    loader = LMDatasetLoader("./train.txt")
    train_data = loader.load()

    pre = Preprocessor(label_is_seq=True, share_vocab=True)
    train_set = pre.run(train_data, pickle_path=PICKLE)

    model = CharLM(50, 50, pre.vocab_size, pre.char_vocab_size)

    trainer = Trainer(task="language_model", loss=Loss("cross_entropy"))

    trainer.train(model, train_set)


if __name__ == "__main__":
    train()
