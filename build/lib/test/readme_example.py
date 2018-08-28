# python: 3.5
# pytorch: 0.4

################
# Test cross validation.
################

from fastNLP.loader.preprocess import ClassPreprocess

from fastNLP.core.predictor import ClassificationInfer
from fastNLP.core.trainer import ClassificationTrainer
from fastNLP.loader.dataset_loader import ClassDatasetLoader
from fastNLP.models.base_model import BaseModel
from fastNLP.modules import aggregation
from fastNLP.modules import encoder
from fastNLP.modules import decoder


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
        self.dec = decoder.MLP(100, num_classes=num_classes)

    def forward(self, x):
        x = self.emb(x)  # [N,L] -> [N,L,C]
        x = self.enc(x)  # [N,L,C_in] -> [N,L,C_out]
        x = self.agg(x)  # [N,L,C] -> [N,C]
        x = self.dec(x)  # [N,C] -> [N, N_class]
        return x


data_dir = 'data'  # directory to save data and model
train_path = 'test/data_for_tests/text_classify.txt'  # training set file

# load dataset
ds_loader = ClassDatasetLoader("train", train_path)
data = ds_loader.load()

# pre-process dataset
pre = ClassPreprocess(data, data_dir, cross_val=True, n_fold=5)
# pre = ClassPreprocess(data, data_dir)
n_classes = pre.num_classes
vocab_size = pre.vocab_size

# construct model
model_args = {
    'num_classes': n_classes,
    'vocab_size': vocab_size
}
model = ClassificationModel(num_classes=n_classes, vocab_size=vocab_size)

# train model
train_args = {
    "epochs": 10,
    "batch_size": 50,
    "pickle_path": data_dir,
    "validate": False,
    "save_best_dev": False,
    "model_saved_path": None,
    "use_cuda": True,
    "learn_rate": 1e-3,
    "momentum": 0.9}
trainer = ClassificationTrainer(train_args)
# trainer.train(model, ['data_train.pkl', 'data_dev.pkl'])
trainer.cross_validate(model)

# predict using model
data_infer = [x[0] for x in data]
infer = ClassificationInfer(data_dir)
labels_pred = infer.predict(model, data_infer)