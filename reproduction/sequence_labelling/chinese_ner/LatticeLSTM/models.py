import torch.nn as nn
from fastNLP.embeddings import StaticEmbedding
from fastNLP.modules import LSTM, ConditionalRandomField
import torch
from fastNLP import seq_len_to_mask
from utils import better_init_rnn,print_info


class LatticeLSTM_SeqLabel(nn.Module):
    def __init__(self, char_embed, bigram_embed, word_embed, hidden_size, label_size, bias=True, bidirectional=False,
                 device=None, embed_dropout=0, output_dropout=0, skip_batch_first=True,debug=False,
                 skip_before_head=False,use_bigram=True,vocabs=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        from modules import LatticeLSTMLayer_sup_back_V0
        super().__init__()
        self.debug = debug
        self.skip_batch_first = skip_batch_first
        self.char_embed_size = char_embed.embedding.weight.size(1)
        self.bigram_embed_size = bigram_embed.embedding.weight.size(1)
        self.word_embed_size = word_embed.embedding.weight.size(1)
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.bidirectional = bidirectional
        self.use_bigram = use_bigram
        self.vocabs = vocabs

        if self.use_bigram:
            self.input_size = self.char_embed_size + self.bigram_embed_size
        else:
            self.input_size = self.char_embed_size

        self.char_embed = char_embed
        self.bigram_embed = bigram_embed
        self.word_embed = word_embed
        self.encoder = LatticeLSTMLayer_sup_back_V0(self.input_size,self.word_embed_size,
                                                 self.hidden_size,
                                                 left2right=True,
                                                 bias=bias,
                                                 device=self.device,
                                                 debug=self.debug,
                                                 skip_before_head=skip_before_head)
        if self.bidirectional:
            self.encoder_back = LatticeLSTMLayer_sup_back_V0(self.input_size,
                                                          self.word_embed_size, self.hidden_size,
                                                          left2right=False,
                                                          bias=bias,
                                                          device=self.device,
                                                          debug=self.debug,
                                                          skip_before_head=skip_before_head)

        self.output = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.label_size)
        self.crf = ConditionalRandomField(label_size, True)

        self.crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size],requires_grad=True))
        if self.crf.include_start_end_trans:
            self.crf.start_scores = nn.Parameter(torch.zeros(size=[label_size],requires_grad=True))
            self.crf.end_scores = nn.Parameter(torch.zeros(size=[label_size],requires_grad=True))

        self.loss_func = nn.CrossEntropyLoss()
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, chars, bigrams, seq_len, target,
                skips_l2r_source, skips_l2r_word, lexicon_count,
                skips_r2l_source=None, skips_r2l_word=None, lexicon_count_back=None):
        # print('skips_l2r_word_id:{}'.format(skips_l2r_word.size()))
        batch = chars.size(0)
        max_seq_len = chars.size(1)
        # max_lexicon_count = skips_l2r_word.size(2)


        embed_char = self.char_embed(chars)
        if self.use_bigram:

            embed_bigram = self.bigram_embed(bigrams)

            embedding = torch.cat([embed_char, embed_bigram], dim=-1)
        else:

            embedding = embed_char


        embed_nonword = self.embed_dropout(embedding)

        # skips_l2r_word = torch.reshape(skips_l2r_word,shape=[batch,-1])
        embed_word = self.word_embed(skips_l2r_word)
        embed_word = self.embed_dropout(embed_word)
        # embed_word = torch.reshape(embed_word,shape=[batch,max_seq_len,max_lexicon_count,-1])


        encoded_h, encoded_c = self.encoder(embed_nonword, seq_len, skips_l2r_source, embed_word, lexicon_count)

        if self.bidirectional:
            embed_word_back = self.word_embed(skips_r2l_word)
            embed_word_back = self.embed_dropout(embed_word_back)
            encoded_h_back, encoded_c_back = self.encoder_back(embed_nonword, seq_len, skips_r2l_source,
                                                               embed_word_back, lexicon_count_back)
            encoded_h = torch.cat([encoded_h, encoded_h_back], dim=-1)

        encoded_h = self.output_dropout(encoded_h)

        pred = self.output(encoded_h)

        mask = seq_len_to_mask(seq_len)

        if self.training:
            loss = self.crf(pred, target, mask)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            return {'pred': pred}

        # batch_size, sent_len = pred.shape[0], pred.shape[1]
        # loss = self.loss_func(pred.reshape(batch_size * sent_len, -1), target.reshape(batch_size * sent_len))
        # return {'pred':pred,'loss':loss}

class LatticeLSTM_SeqLabel_V1(nn.Module):
    def __init__(self, char_embed, bigram_embed, word_embed, hidden_size, label_size, bias=True, bidirectional=False,
                 device=None, embed_dropout=0, output_dropout=0, skip_batch_first=True,debug=False,
                 skip_before_head=False,use_bigram=True,vocabs=None,gaz_dropout=0):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        from modules import LatticeLSTMLayer_sup_back_V1
        super().__init__()
        self.count = 0
        self.debug = debug
        self.skip_batch_first = skip_batch_first
        self.char_embed_size = char_embed.embedding.weight.size(1)
        self.bigram_embed_size = bigram_embed.embedding.weight.size(1)
        self.word_embed_size = word_embed.embedding.weight.size(1)
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.bidirectional = bidirectional
        self.use_bigram = use_bigram
        self.vocabs = vocabs

        if self.use_bigram:
            self.input_size = self.char_embed_size + self.bigram_embed_size
        else:
            self.input_size = self.char_embed_size

        self.char_embed = char_embed
        self.bigram_embed = bigram_embed
        self.word_embed = word_embed
        self.encoder = LatticeLSTMLayer_sup_back_V1(self.input_size,self.word_embed_size,
                                                 self.hidden_size,
                                                 left2right=True,
                                                 bias=bias,
                                                 device=self.device,
                                                 debug=self.debug,
                                                 skip_before_head=skip_before_head)
        if self.bidirectional:
            self.encoder_back = LatticeLSTMLayer_sup_back_V1(self.input_size,
                                                          self.word_embed_size, self.hidden_size,
                                                          left2right=False,
                                                          bias=bias,
                                                          device=self.device,
                                                          debug=self.debug,
                                                          skip_before_head=skip_before_head)

        self.output = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.label_size)
        self.crf = ConditionalRandomField(label_size, True)

        self.crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size],requires_grad=True))
        if self.crf.include_start_end_trans:
            self.crf.start_scores = nn.Parameter(torch.zeros(size=[label_size],requires_grad=True))
            self.crf.end_scores = nn.Parameter(torch.zeros(size=[label_size],requires_grad=True))

        self.loss_func = nn.CrossEntropyLoss()
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.gaz_dropout = nn.Dropout(gaz_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, chars, bigrams, seq_len, target,
                skips_l2r_source, skips_l2r_word, lexicon_count,
                skips_r2l_source=None, skips_r2l_word=None, lexicon_count_back=None):

        batch = chars.size(0)
        max_seq_len = chars.size(1)



        embed_char = self.char_embed(chars)
        if self.use_bigram:

            embed_bigram = self.bigram_embed(bigrams)

            embedding = torch.cat([embed_char, embed_bigram], dim=-1)
        else:

            embedding = embed_char


        embed_nonword = self.embed_dropout(embedding)

        # skips_l2r_word = torch.reshape(skips_l2r_word,shape=[batch,-1])
        embed_word = self.word_embed(skips_l2r_word)
        embed_word = self.embed_dropout(embed_word)



        encoded_h, encoded_c = self.encoder(embed_nonword, seq_len, skips_l2r_source, embed_word, lexicon_count)

        if self.bidirectional:
            embed_word_back = self.word_embed(skips_r2l_word)
            embed_word_back = self.embed_dropout(embed_word_back)
            encoded_h_back, encoded_c_back = self.encoder_back(embed_nonword, seq_len, skips_r2l_source,
                                                               embed_word_back, lexicon_count_back)
            encoded_h = torch.cat([encoded_h, encoded_h_back], dim=-1)

        encoded_h = self.output_dropout(encoded_h)

        pred = self.output(encoded_h)

        mask = seq_len_to_mask(seq_len)

        if self.training:
            loss = self.crf(pred, target, mask)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            return {'pred': pred}


class LSTM_SeqLabel(nn.Module):
    def __init__(self, char_embed, bigram_embed, word_embed, hidden_size, label_size, bias=True,
                 bidirectional=False, device=None, embed_dropout=0, output_dropout=0,use_bigram=True):

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        super().__init__()
        self.char_embed_size = char_embed.embedding.weight.size(1)
        self.bigram_embed_size = bigram_embed.embedding.weight.size(1)
        self.word_embed_size = word_embed.embedding.weight.size(1)
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.bidirectional = bidirectional
        self.use_bigram = use_bigram

        self.char_embed = char_embed
        self.bigram_embed = bigram_embed
        self.word_embed = word_embed

        if self.use_bigram:
            self.input_size = self.char_embed_size + self.bigram_embed_size
        else:
            self.input_size = self.char_embed_size

        self.encoder = LSTM(self.input_size, self.hidden_size,
                            bidirectional=self.bidirectional)

        better_init_rnn(self.encoder.lstm)


        self.output = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.label_size)

        self.debug = True
        self.loss_func = nn.CrossEntropyLoss()
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        self.crf = ConditionalRandomField(label_size, True)

    def forward(self, chars, bigrams, seq_len, target):
        if self.debug:

            print_info('chars:{}'.format(chars.size()))
            print_info('bigrams:{}'.format(bigrams.size()))
            print_info('seq_len:{}'.format(seq_len.size()))
            print_info('target:{}'.format(target.size()))
        embed_char = self.char_embed(chars)

        if self.use_bigram:

            embed_bigram = self.bigram_embed(bigrams)

            embedding = torch.cat([embed_char, embed_bigram], dim=-1)
        else:

            embedding = embed_char

        embedding = self.embed_dropout(embedding)

        encoded_h, encoded_c = self.encoder(embedding, seq_len)

        encoded_h = self.output_dropout(encoded_h)

        pred = self.output(encoded_h)

        mask = seq_len_to_mask(seq_len)

        # pred = self.crf(pred)

        # batch_size, sent_len = pred.shape[0], pred.shape[1]
        # loss = self.loss_func(pred.reshape(batch_size * sent_len, -1), target.reshape(batch_size * sent_len))
        if self.debug:
            print('debug mode:finish')
            exit(1208)
        if self.training:
            loss = self.crf(pred, target, mask)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            return {'pred': pred}
