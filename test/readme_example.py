from fastNLP.core.loss import Loss
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.predictor import ClassificationInfer
from fastNLP.core.preprocess import ClassPreprocess
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.loader.dataset_loader import ClassDatasetLoader
from fastNLP.models.base_model import BaseModel
from fastNLP.modules import aggregation
from fastNLP.modules import decoder
from fastNLP.modules import encoder


class ClassificationModel(BaseModel):
    """
    Simple text classification model based on CNN.
    """

    def __init__(self, num_classes, vocab_size):
        super(ClassificationModel, self).__init__()

        self.emb = encoder.Embedding(nums=vocab_size, dims=300)
        self.enc = encoder.Conv(
            in_channels=300, out_channels=100, kernel_size=3)
        self.agg = aggregation.MaxPool()
        self.dec = decoder.MLP(size_layer=[100, num_classes])

    def forward(self, x):
        x = self.emb(x)  # [N,L] -> [N,L,C]
        x = self.enc(x)  # [N,L,C_in] -> [N,L,C_out]
        x = self.agg(x)  # [N,L,C] -> [N,C]
        x = self.dec(x)  # [N,C] -> [N, N_class]
        return x


data_dir = 'save/'  # directory to save data and model
train_path = './data_for_tests/text_classify.txt'  # training set file

# load dataset
ds_loader = ClassDatasetLoader(train_path)
data = ds_loader.load()

# pre-process dataset
pre = ClassPreprocess()
train_set, dev_set = pre.run(data, train_dev_split=0.3, pickle_path=data_dir)
n_classes, vocab_size = pre.num_classes, pre.vocab_size

# construct model
model_args = {
    'num_classes': n_classes,
    'vocab_size': vocab_size
}
model = ClassificationModel(num_classes=n_classes, vocab_size=vocab_size)

# construct trainer
train_args = {
    "epochs": 3,
    "batch_size": 16,
    "pickle_path": data_dir,
    "validate": False,
    "save_best_dev": False,
    "model_saved_path": None,
    "use_cuda": True,
    "loss": Loss("cross_entropy"),
    "optimizer": Optimizer("Adam", lr=0.001)
}
trainer = ClassificationTrainer(**train_args)

# start training
trainer.train(model, train_data=train_set, dev_data=dev_set)

# predict using model
data_infer = [x[0] for x in data]
infer = ClassificationInfer(data_dir)
labels_pred = infer.predict(model.cpu(), data_infer)
print(labels_pred)
