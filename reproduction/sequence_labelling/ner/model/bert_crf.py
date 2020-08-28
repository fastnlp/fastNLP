

from torch import nn
from fastNLP.modules import ConditionalRandomField, allowed_transitions
import torch.nn.functional as F

class BertCRF(nn.Module):
    def __init__(self, embed, tag_vocab, encoding_type='bio'):
        super().__init__()
        self.embed = embed
        self.fc = nn.Linear(self.embed.embed_size, len(tag_vocab))
        trans = allowed_transitions(tag_vocab, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, words, target):
        mask = words.ne(0)
        words = self.embed(words)
        words = self.fc(words)
        logits = F.log_softmax(words, dim=-1)
        if target is not None:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}
        else:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}

    def forward(self, words, target):
        return self._forward(words, target)

    def predict(self, words):
        return self._forward(words, None)
