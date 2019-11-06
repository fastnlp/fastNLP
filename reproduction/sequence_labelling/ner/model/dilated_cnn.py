import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules.decoder import ConditionalRandomField
from fastNLP.embeddings import Embedding
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.core.const import Const as C


class IDCNN(nn.Module):
    def __init__(self,
                 init_embed,
                 char_embed,
                 num_cls,
                 repeats, num_layers, num_filters, kernel_size,
                 use_crf=False, use_projection=False, block_loss=False,
                 input_dropout=0.3, hidden_dropout=0.2, inner_dropout=0.0):
        super(IDCNN, self).__init__()
        self.word_embeddings = Embedding(init_embed)

        if char_embed is None:
            self.char_embeddings = None
            embedding_size = self.word_embeddings.embedding_dim
        else:
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
            dilated = 2 ** layer_i if layer_i+1 < num_layers else 1
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
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)

    def forward(self, words, seq_len, target=None, chars=None):
        if self.char_embeddings is None:
            x = self.word_embeddings(words)
        else:
            if chars is None:
                raise ValueError('must provide chars for model with char embedding')
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
                loss = self.crf(y.transpose(1, 2), t, mask)
            else:
                y.masked_fill_((mask.eq(False))[:,None,:], -100)
                # f_mask = mask.float()
                # t = f_mask * t + (1-f_mask) * -100
                loss = F.cross_entropy(y, t, ignore_index=-100)
            return loss

        if target is not None:
            if self.block_loss:
                losses = [compute_loss(o, target, mask) for o in output]
                loss = sum(losses)
            else:
                loss = compute_loss(output[-1], target, mask)
        else:
            loss = None

        scores = output[-1]
        if self.crf is not None:
            pred, _ = self.crf.viterbi_decode(scores.transpose(1, 2), mask)
        else:
            pred = scores.max(1)[1] * mask.long()

        return {
            C.LOSS: loss,
            C.OUTPUT: pred,
        }

