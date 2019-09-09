import sys
sys.path.append('../../..')

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
from fastNLP import cache_results, Vocabulary
from fastNLP.io.pipe.utils import _add_chars_field, _indexize

from fastNLP.io.pipe import Pipe
from fastNLP.core.utils import iob2bioes, iob2
from fastNLP.io import MsraNERLoader, WeiboNERLoader

class ChineseNERPipe(Pipe):
    def __init__(self, encoding_type: str = 'bio', target_pad_val=0, bigram=False):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        else:
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        self.target_pad_val = int(target_pad_val)
        self.bigram = bigram

    def process(self, data_bundle):
        data_bundle.copy_field(C.RAW_CHAR, C.CHAR_INPUT)
        input_fields = [C.TARGET, C.CHAR_INPUT, C.INPUT_LEN]
        target_fields = [C.TARGET, C.INPUT_LEN]
        if self.bigram:
            for dataset in data_bundle.datasets.values():
                dataset.apply_field(lambda chars:[c1+c2 for c1, c2 in zip(chars, chars[1:]+['<eos>'])],
                                    field_name=C.CHAR_INPUT, new_field_name='bigrams')
            bigram_vocab = Vocabulary()
            bigram_vocab.from_dataset(data_bundle.get_dataset('train'),field_name='bigrams',
                            no_create_entry_dataset=[ds for name, ds in data_bundle.datasets.items() if name!='train'])
            bigram_vocab.index_dataset(*data_bundle.datasets.values(), field_name='bigrams')
            data_bundle.set_vocab(bigram_vocab, field_name='bigrams')
            input_fields.append('bigrams')

        _add_chars_field(data_bundle, lower=False)

        # index
        _indexize(data_bundle, input_field_names=C.CHAR_INPUT, target_field_names=C.TARGET)

        for name, dataset in data_bundle.datasets.items():
            dataset.set_pad_val(C.TARGET, self.target_pad_val)
            dataset.add_seq_len(C.CHAR_INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle


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
@cache_results('caches/weibo-lstm.pkl', _refresh=False)
def get_data():
    data_bundle = WeiboNERLoader().load()
    data_bundle = ChineseNERPipe(encoding_type='bioes', bigram=True).process(data_bundle)
    char_embed = StaticEmbedding(data_bundle.get_vocab(C.CHAR_INPUT), model_dir_or_name='cn-fasttext')
    bigram_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'), embedding_dim=100, min_freq=3)
    return data_bundle, char_embed, bigram_embed
data_bundle, char_embed, bigram_embed = get_data()
# data_bundle = get_data()
print(data_bundle)

# exit(0)
model = CNBiLSTMCRFNER(char_embed, num_classes=len(data_bundle.vocabs['target']), bigram_embed=bigram_embed)

Trainer(data_bundle.datasets['train'], model, batch_size=20,
     metrics=SpanFPreRecMetric(data_bundle.vocabs['target'], encoding_type='bioes'),
     num_workers=2, dev_data=data_bundle. datasets['dev'], device=0).train()

