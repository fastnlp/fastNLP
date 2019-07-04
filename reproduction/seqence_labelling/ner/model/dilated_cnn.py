import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules.decoder import ConditionalRandomField
from fastNLP.modules.encoder import Embedding
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.core.const import Const as C


class IDCNN(nn.Module):
    def __init__(self, init_embed, char_embed,
                 num_cls,
                 repeats, num_layers, num_filters, kernel_size,
                 use_crf=False, use_projection=False, block_loss=False,
                 input_dropout=0.3, hidden_dropout=0.2, inner_dropout=0.0):
        super(IDCNN, self).__init__()
        self.word_embeddings = Embedding(init_embed)
        self.char_embeddings = Embedding(char_embed)
        embedding_size = self.word_embeddings.embedding_dim + \
            self.char_embeddings.embedding_dim

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=num_filters,
                      kernel_size=kernel_size,
                      stride=1, dilation=1,
                      padding=kernel_size//2,
                      bias=True),
            nn.ReLU(),
        )

        block = []
        for layer_i in range(num_layers):
            dilated = 2 ** layer_i
            block.append(nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1, dilation=dilated,
                padding=(kernel_size//2) * dilated,
                bias=True))
            block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

        if use_projection:
            self.projection = nn.Sequential(
                nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters//2,
                    kernel_size=1,
                    bias=True),
                nn.ReLU(),)
            encode_dim = num_filters // 2
        else:
            self.projection = None
            encode_dim = num_filters

        self.input_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.inner_drop = nn.Dropout(inner_dropout)
        self.repeats = repeats
        self.out_fc = nn.Conv1d(
            in_channels=encode_dim,
            out_channels=num_cls,
            kernel_size=1,
            bias=True)
        self.crf = ConditionalRandomField(
            num_tags=num_cls) if use_crf else None
        self.block_loss = block_loss

    def forward(self, words, chars, seq_len, target=None):
        e1 = self.word_embeddings(words)
        e2 = self.char_embeddings(chars)
        x = torch.cat((e1, e2), dim=-1) # b,l,h
        mask = seq_len_to_mask(seq_len)

        x = x.transpose(1, 2) # b,h,l
        last_output = self.conv0(x)
        output = []
        for repeat in range(self.repeats):
            last_output = self.block(last_output)
            hidden = self.projection(last_output) if self.projection is not None else last_output
            output.append(self.out_fc(hidden))

        def compute_loss(y, t, mask):
            if self.crf is not None and target is not None:
                loss = self.crf(y, t, mask)
            else:
                t.masked_fill_(mask == 0, -100)
                loss = F.cross_entropy(y, t, ignore_index=-100)
            return loss

        if self.block_loss:
            losses = [compute_loss(o, target, mask) for o in output]
            loss = sum(losses)
        else:
            loss = compute_loss(output[-1], target, mask)

        scores = output[-1]
        if self.crf is not None:
            pred = self.crf.viterbi_decode(scores, target, mask)
        else:
            pred = scores.max(1)[1] * mask.long()

        return {
            C.LOSS: loss,
            C.OUTPUT: pred,
        }

    def predict(self, words, chars, seq_len):
        return self.forward(words, chars, seq_len)[C.OUTPUT]
