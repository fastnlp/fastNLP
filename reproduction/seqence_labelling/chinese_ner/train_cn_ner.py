


from reproduction.seqence_labelling.chinese_ner.data.ChineseNER import ChineseNERLoader
from fastNLP.embeddings import StaticEmbedding

from torch import nn
import torch
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules import LSTM
from fastNLP.modules import ConditionalRandomField
from fastNLP.modules import allowed_transitions
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from fastNLP.core.const import Const as C
from fastNLP import SpanFPreRecMetric, Trainer
from fastNLP import cache_results

class CNBiLSTMCRFNER(nn.Module):
    def __init__(self, char_embed, num_classes, bigram_embed=None, trigram_embed=None, num_layers=1, hidden_size=100,
                 dropout=0.5, target_vocab=None, encoding_type=None):
        super().__init__()

        self.char_embed = get_embeddings(char_embed)
        embed_size = self.char_embed.embedding_dim
        if bigram_embed:
            self.bigram_embed = get_embeddings(bigram_embed)
            embed_size += self.bigram_embed.embedding_dim
        if trigram_embed:
            self.trigram_ebmbed = get_embeddings(trigram_embed)
            embed_size += self.bigram_embed.embedding_dim

        if num_layers>1:
            self.lstm = LSTM(embed_size, num_layers=num_layers, hidden_size=hidden_size//2, bidirectional=True,
                             batch_first=True, dropout=dropout)
        else:
            self.lstm = LSTM(embed_size, num_layers=num_layers, hidden_size=hidden_size//2, bidirectional=True,
                             batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        trans = None
        if target_vocab is not None and encoding_type is not None:
            trans = allowed_transitions(target_vocab.idx2word, encoding_type=encoding_type, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, chars, bigrams=None, trigrams=None, seq_len=None, target=None):
        chars = self.char_embed(chars)
        if hasattr(self, 'bigram_embed'):
            bigrams = self.bigram_embed(bigrams)
            chars = torch.cat((chars, bigrams), dim=-1)
        if hasattr(self, 'trigram_embed'):
            trigrams = self.trigram_embed(trigrams)
            chars = torch.cat((chars, trigrams), dim=-1)
        feats, _ = self.lstm(chars, seq_len=seq_len)
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {C.OUTPUT: pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {C.LOSS:loss}

    def forward(self, chars, target, bigrams=None, trigrams=None, seq_len=None):
        return self._forward(chars, bigrams, trigrams, seq_len, target)

    def predict(self, chars, seq_len=None, bigrams=None, trigrams=None):
        return self._forward(chars, bigrams, trigrams, seq_len)

# data_bundle = pickle.load(open('caches/msra.pkl', 'rb'))
@cache_results('caches/msra.pkl', _refresh=True)
def get_data():
    data_bundle = ChineseNERLoader().process('MSRA-NER/', bigrams=True)
    char_embed = StaticEmbedding(data_bundle.vocabs['chars'],
                                 model_dir_or_name='cn-char')
    bigram_embed = StaticEmbedding(data_bundle.vocabs['bigrams'],
                                   model_dir_or_name='cn-bigram')
    return data_bundle, char_embed, bigram_embed
data_bundle, char_embed, bigram_embed = get_data()
print(data_bundle)
# exit(0)
data_bundle.datasets['train'].set_input('target')
data_bundle.datasets['dev'].set_input('target')
model = CNBiLSTMCRFNER(char_embed, num_classes=len(data_bundle.vocabs['target']), bigram_embed=bigram_embed)

Trainer(data_bundle.datasets['train'], model, batch_size=640,
     metrics=SpanFPreRecMetric(data_bundle.vocabs['target'], encoding_type='bioes'),
     num_workers=2, dev_data=data_bundle. datasets['dev'], device=3).train()

