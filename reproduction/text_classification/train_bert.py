import sys
sys.path.append('../../')

from reproduction.text_classification.data.IMDBLoader import IMDBLoader
from fastNLP.embeddings import BertEmbedding
from reproduction.text_classification.model.lstm import BiLSTMSentiment
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP import cache_results
from fastNLP import Tester

# 对返回结果进行缓存，下一次运行就会自动跳过预处理
@cache_results('imdb.pkl')
def get_data():
    data_bundle = IMDBLoader().process('imdb/')
    return data_bundle
data_bundle = get_data()

print(data_bundle)

# 删除超过512, 但由于英语中会把word进行word piece处理，所以截取的时候做一点的裕量
data_bundle.datasets['train'].drop(lambda x:len(x['words'])>400)
data_bundle.datasets['dev'].drop(lambda x:len(x['words'])>400)
data_bundle.datasets['test'].drop(lambda x:len(x['words'])>400)
bert_embed = BertEmbedding(data_bundle.vocabs['words'], requires_grad=False,
                           model_dir_or_name="en-base-uncased")
model = BiLSTMSentiment(bert_embed, len(data_bundle.vocabs['target']))

Trainer(data_bundle.datasets['train'], model, optimizer=None, loss=CrossEntropyLoss(), device=0,
                 batch_size=10, dev_data=data_bundle.datasets['dev'], metrics=AccuracyMetric()).train()

# 在测试集上测试一下效果
Tester(data_bundle.datasets['test'], model, batch_size=32, metrics=AccuracyMetric()).test()