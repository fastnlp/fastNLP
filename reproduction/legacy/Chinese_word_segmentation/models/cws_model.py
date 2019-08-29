
import torch
from torch import nn

from fastNLP.models.base_model import BaseModel
from fastNLP.modules.decoder.mlp import MLP
from reproduction.legacy.Chinese_word_segmentation.utils import seq_lens_to_mask


class CWSBiLSTMEncoder(BaseModel):
    def __init__(self, vocab_num, embed_dim=100, bigram_vocab_num=None, bigram_embed_dim=100, num_bigram_per_char=None,
                 hidden_size=200, bidirectional=True, embed_drop_p=0.2, num_layers=1):
        super().__init__()

        self.input_size = 0
        self.num_bigram_per_char = num_bigram_per_char
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embed_drop_p = embed_drop_p
        if self.bidirectional:
            self.hidden_size = hidden_size//2
            self.num_directions = 2
        else:
            self.hidden_size = hidden_size
            self.num_directions = 1

        if not bigram_vocab_num is None:
            assert not bigram_vocab_num is None, "Specify num_bigram_per_char."

        if vocab_num is not None:
            self.char_embedding = nn.Embedding(num_embeddings=vocab_num, embedding_dim=embed_dim)
            self.input_size += embed_dim

        if bigram_vocab_num is not None:
            self.bigram_embedding = nn.Embedding(num_embeddings=bigram_vocab_num, embedding_dim=bigram_embed_dim)
            self.input_size += self.num_bigram_per_char*bigram_embed_dim

        if not self.embed_drop_p is None:
            self.embedding_drop = nn.Dropout(p=self.embed_drop_p)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=self.bidirectional,
                    batch_first=True, num_layers=self.num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias_hh' in name:
                nn.init.constant_(param, 0)
            elif 'bias_ih' in name:
                nn.init.constant_(param, 1)
            else:
                nn.init.xavier_uniform_(param)

    def init_embedding(self, embedding, embed_name):
        if embed_name == 'bigram':
            self.bigram_embedding.weight.data = torch.from_numpy(embedding)
        elif embed_name == 'char':
            self.char_embedding.weight.data = torch.from_numpy(embedding)


    def forward(self, chars, bigrams=None, seq_lens=None):

        batch_size, max_len = chars.size()

        x_tensor = self.char_embedding(chars)

        if hasattr(self, 'bigram_embedding'):
            bigram_tensor = self.bigram_embedding(bigrams).view(batch_size, max_len, -1)
            x_tensor = torch.cat([x_tensor, bigram_tensor], dim=2)
        x_tensor = self.embedding_drop(x_tensor)
        sorted_lens, sorted_indices = torch.sort(seq_lens, descending=True)
        packed_x = nn.utils.rnn.pack_padded_sequence(x_tensor[sorted_indices], sorted_lens, batch_first=True)

        outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        _, desorted_indices = torch.sort(sorted_indices, descending=False)
        outputs = outputs[desorted_indices]

        return outputs


class CWSBiLSTMSegApp(BaseModel):
    def __init__(self, vocab_num, embed_dim=100, bigram_vocab_num=None, bigram_embed_dim=100, num_bigram_per_char=None,
                 hidden_size=200, bidirectional=True, embed_drop_p=None, num_layers=1, tag_size=2):
        super(CWSBiLSTMSegApp, self).__init__()

        self.tag_size = tag_size

        self.encoder_model = CWSBiLSTMEncoder(vocab_num, embed_dim, bigram_vocab_num, bigram_embed_dim, num_bigram_per_char,
                 hidden_size, bidirectional, embed_drop_p, num_layers)

        size_layer = [hidden_size, 200, tag_size]
        self.decoder_model = MLP(size_layer)


    def forward(self, chars, seq_lens, bigrams=None):
        device = self.parameters().__next__().device
        chars = chars.to(device).long()
        if not bigrams is None:
            bigrams = bigrams.to(device).long()
        else:
            bigrams = None
        seq_lens = seq_lens.to(device).long()

        feats = self.encoder_model(chars, bigrams, seq_lens)
        probs = self.decoder_model(feats)

        pred_dict = {}
        pred_dict['seq_lens'] = seq_lens
        pred_dict['pred_probs'] = probs

        return pred_dict

    def predict(self, chars, seq_lens, bigrams=None):
        pred_dict = self.forward(chars, seq_lens, bigrams)
        pred_probs = pred_dict['pred_probs']
        _, pred_tags = pred_probs.max(dim=-1)
        return {'pred_tags': pred_tags}


from fastNLP.modules.decoder.crf import ConditionalRandomField
from fastNLP.modules.decoder.crf import allowed_transitions

class CWSBiLSTMCRF(BaseModel):
    def __init__(self, vocab_num, embed_dim=100, bigram_vocab_num=None, bigram_embed_dim=100, num_bigram_per_char=None,
                 hidden_size=200, bidirectional=True, embed_drop_p=0.2, num_layers=1, tag_size=4):
        """
        默认使用BMES的标注方式
        :param vocab_num:
        :param embed_dim:
        :param bigram_vocab_num:
        :param bigram_embed_dim:
        :param num_bigram_per_char:
        :param hidden_size:
        :param bidirectional:
        :param embed_drop_p:
        :param num_layers:
        :param tag_size:
        """
        super(CWSBiLSTMCRF, self).__init__()

        self.tag_size = tag_size

        self.encoder_model = CWSBiLSTMEncoder(vocab_num, embed_dim, bigram_vocab_num, bigram_embed_dim, num_bigram_per_char,
                 hidden_size, bidirectional, embed_drop_p, num_layers)

        size_layer = [hidden_size, 200, tag_size]
        self.decoder_model = MLP(size_layer)
        allowed_trans = allowed_transitions({0:'b', 1:'m', 2:'e', 3:'s'}, encoding_type='bmes')
        self.crf = ConditionalRandomField(num_tags=tag_size, include_start_end_trans=False,
                                          allowed_transitions=allowed_trans)


    def forward(self, chars, target, seq_lens, bigrams=None):
        device = self.parameters().__next__().device
        chars = chars.to(device).long()
        if not bigrams is None:
            bigrams = bigrams.to(device).long()
        else:
            bigrams = None
        seq_lens = seq_lens.to(device).long()
        masks = seq_lens_to_mask(seq_lens)
        feats = self.encoder_model(chars, bigrams, seq_lens)
        feats = self.decoder_model(feats)
        losses = self.crf(feats, target, masks)

        pred_dict = {}
        pred_dict['seq_lens'] = seq_lens
        pred_dict['loss'] = torch.mean(losses)

        return pred_dict

    def predict(self, chars, seq_lens, bigrams=None):
        device = self.parameters().__next__().device
        chars = chars.to(device).long()
        if not bigrams is None:
            bigrams = bigrams.to(device).long()
        else:
            bigrams = None
        seq_lens = seq_lens.to(device).long()
        masks = seq_lens_to_mask(seq_lens)
        feats = self.encoder_model(chars, bigrams, seq_lens)
        feats = self.decoder_model(feats)
        paths, _ = self.crf.viterbi_decode(feats, masks)

        return {'pred': paths, 'seq_lens':seq_lens}

