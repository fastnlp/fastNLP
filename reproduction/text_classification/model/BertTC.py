from fastNLP.embeddings import BertEmbedding
import torch
import torch.nn as nn
from fastNLP.core.const import Const as C

class BertTC(nn.Module):
    def __init__(self, vocab,num_class,bert_model_dir_or_name,fine_tune=False):
        super(BertTC, self).__init__()
        self.embed=BertEmbedding(vocab, requires_grad=fine_tune,
                           model_dir_or_name=bert_model_dir_or_name,include_cls_sep=True)
        self.classifier = nn.Linear(self.embed.embedding_dim, num_class)

    def forward(self, words):
        embedding_cls=self.embed(words)[:,0]
        output=self.classifier(embedding_cls)
        return {C.OUTPUT: output}

    def predict(self,words):
        return self.forward(words)

if __name__=="__main__":
    ta=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    tb=ta[:,0]
    print(tb)
