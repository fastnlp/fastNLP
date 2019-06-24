from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.embed_loader import EmbedLoader

# from tools.logger import *
from tools.PositionEmbedding import get_sinusoid_encoding_table

WORD_PAD = "[PAD]"

class Encoder(nn.Module):
    def __init__(self, hps, embed):
        """
        
        :param hps: 
                word_emb_dim: word embedding dimension
                sent_max_len: max token number in the sentence
                output_channel: output channel for cnn
                min_kernel_size: min kernel size for cnn
                max_kernel_size: max kernel size for cnn
                word_embedding: bool, use word embedding or not
                embedding_path: word embedding path
                embed_train: bool, whether to train word embedding
                cuda: bool, use cuda or not
        :param vocab: FastNLP.Vocabulary
        """
        super(Encoder, self).__init__()

        self._hps = hps
        self.sent_max_len = hps.sent_max_len
        embed_size = hps.word_emb_dim

        sent_max_len = hps.sent_max_len

        input_channels = 1
        out_channels = hps.output_channel
        min_kernel_size = hps.min_kernel_size
        max_kernel_size = hps.max_kernel_size
        width = embed_size

        # word embedding
        self.embed = embed

        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(sent_max_len + 1, embed_size, padding_idx=0), freeze=True)

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size = (height, width)) for height in range(min_kernel_size, max_kernel_size+1)])
        print("[INFO] Initing W for CNN.......")
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))
            fan_in, fan_out = Encoder.calculate_fan_in_and_fan_out(conv.weight.data)
            std = np.sqrt(init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            print("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")
            raise ValueError("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, input):
        # input: a batch of Example object [batch_size, N, seq_len]

        batch_size, N, _ = input.size()
        input = input.view(-1, input.size(2))   # [batch_size*N, L]
        input_sent_len = ((input!=0).sum(dim=1)).int()  # [batch_size*N, 1]
        enc_embed_input = self.embed(input) # [batch_size*N, L, D]

        input_pos = torch.Tensor([np.hstack((np.arange(1, sentlen + 1), np.zeros(self.sent_max_len - sentlen))) for sentlen in input_sent_len])
        if self._hps.cuda:
            input_pos = input_pos.cuda()
        enc_pos_embed_input = self.position_embedding(input_pos.long()) # [batch_size*N, D]
        enc_conv_input = enc_embed_input + enc_pos_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1) # (batch * N,Ci,L,D)
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs] # kernel_sizes * (batch*N, Co, W)
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output] # kernel_sizes * (batch*N, Co)
        sent_embedding = torch.cat(enc_maxpool_output, 1) # (batch*N, Co * kernel_sizes)
        sent_embedding = sent_embedding.view(batch_size, N, -1)
        return sent_embedding

class DomainEncoder(Encoder):
    def __init__(self, hps, vocab, domaindict):
        super(DomainEncoder, self).__init__(hps, vocab)

        # domain embedding
        self.domain_embedding = nn.Embedding(domaindict.size(), hps.domain_emb_dim)
        self.domain_embedding.weight.requires_grad = True

    def forward(self, input, domain):
        """
        :param input: [batch_size, N, seq_len], N sentence number, seq_len token number
        :param domain: [batch_size]
        :return: sent_embedding: [batch_size, N, Co * kernel_sizes]
        """

        batch_size, N, _ = input.size()

        sent_embedding = super().forward(input)
        enc_domain_input = self.domain_embedding(domain)  # [batch, D]
        enc_domain_input = enc_domain_input.unsqueeze(1).expand(batch_size, N, -1)  # [batch, N, D]
        sent_embedding = torch.cat((sent_embedding, enc_domain_input), dim=2)
        return sent_embedding

class MultiDomainEncoder(Encoder):
    def __init__(self, hps, vocab, domaindict):
        super(MultiDomainEncoder, self).__init__(hps, vocab)

        self.domain_size = domaindict.size()

        # domain embedding
        self.domain_embedding = nn.Embedding(self.domain_size, hps.domain_emb_dim)
        self.domain_embedding.weight.requires_grad = True

    def forward(self, input, domain):
        """
        :param input: [batch_size, N, seq_len], N sentence number, seq_len token number
        :param domain: [batch_size, domain_size]
        :return: sent_embedding: [batch_size, N, Co * kernel_sizes]
        """

        batch_size, N, _ = input.size()

        # logger.info(domain[:5, :])

        sent_embedding = super().forward(input)
        domain_padding = torch.arange(self.domain_size).unsqueeze(0).expand(batch_size, -1)
        domain_padding = domain_padding.cuda().view(-1) if self._hps.cuda else domain_padding.view(-1) # [batch * domain_size]

        enc_domain_input = self.domain_embedding(domain_padding)  # [batch * domain_size, D]
        enc_domain_input = enc_domain_input.view(batch_size, self.domain_size, -1) * domain.unsqueeze(-1).float()   # [batch, domain_size, D]

        # logger.info(enc_domain_input[:5,:])   # [batch, domain_size, D]

        enc_domain_input = enc_domain_input.sum(1) / domain.sum(1).float().unsqueeze(-1) # [batch, D]
        enc_domain_input = enc_domain_input.unsqueeze(1).expand(batch_size, N, -1)  # [batch, N, D]
        sent_embedding = torch.cat((sent_embedding, enc_domain_input), dim=2)
        return sent_embedding


class BertEncoder(nn.Module):
    def __init__(self, hps):
        super(BertEncoder, self).__init__()

        from pytorch_pretrained_bert.modeling import BertModel

        self._hps = hps
        self.sent_max_len = hps.sent_max_len
        self._cuda = hps.cuda

        embed_size = hps.word_emb_dim
        sent_max_len = hps.sent_max_len

        input_channels = 1
        out_channels = hps.output_channel
        min_kernel_size = hps.min_kernel_size
        max_kernel_size = hps.max_kernel_size
        width = embed_size

        # word embedding
        self._bert = BertModel.from_pretrained("/remote-home/dqwang/BERT/pre-train/uncased_L-24_H-1024_A-16")
        self._bert.eval()
        for p in self._bert.parameters():
            p.requires_grad = False

        self.word_embedding_proj = nn.Linear(4096, embed_size)

        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(sent_max_len + 1, embed_size, padding_idx=0), freeze=True)

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size = (height, width)) for height in range(min_kernel_size, max_kernel_size+1)])
        logger.info("[INFO] Initing W for CNN.......")
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))
            fan_in, fan_out = Encoder.calculate_fan_in_and_fan_out(conv.weight.data)
            std = np.sqrt(init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            logger.error("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")
            raise ValueError("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def pad_encoder_input(self, input_list):
        """
        :param input_list: N [seq_len, hidden_state]
        :return: enc_sent_input_pad: list, N [max_len, hidden_state]
        """
        max_len = self.sent_max_len
        enc_sent_input_pad = []
        _, hidden_size = input_list[0].size()
        for i in range(len(input_list)):
            article_words = input_list[i]  # [seq_len, hidden_size]
            seq_len = article_words.size(0)
            if seq_len > max_len:
                pad_words = article_words[:max_len, :]
            else:
                pad_tensor = torch.zeros(max_len - seq_len, hidden_size).cuda() if self._cuda else torch.zeros(max_len - seq_len, hidden_size)
                pad_words = torch.cat([article_words, pad_tensor], dim=0)
            enc_sent_input_pad.append(pad_words)
        return enc_sent_input_pad

    def forward(self, inputs, input_masks, enc_sent_len):
        """
        
        :param inputs: a batch of Example object [batch_size, doc_len=512]
        :param input_masks: 0 or 1, [batch, doc_len=512]
        :param enc_sent_len: sentence original length [batch, N]
        :return: 
        """


        # Use Bert to get word embedding
        batch_size, N = enc_sent_len.size()
        input_pad_list = []
        for i in range(batch_size):
            tokens_id = inputs[i]
            input_mask = input_masks[i]
            sent_len = enc_sent_len[i]
            input_ids = tokens_id.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)

            out, _ = self._bert(input_ids, token_type_ids=None, attention_mask=input_mask)
            out = torch.cat(out[-4:], dim=-1).squeeze(0)  # [doc_len=512, hidden_state=4096]

            _, hidden_size = out.size()

            # restore the sentence
            last_end = 1
            enc_sent_input = []
            for length in sent_len:
                if length != 0 and last_end < 511:
                    enc_sent_input.append(out[last_end: min(511, last_end + length), :])
                    last_end += length
                else:
                    pad_tensor = torch.zeros(self.sent_max_len, hidden_size).cuda() if self._hps.cuda else torch.zeros(self.sent_max_len, hidden_size)
                    enc_sent_input.append(pad_tensor)


            # pad the sentence
            enc_sent_input_pad = self.pad_encoder_input(enc_sent_input) # [N, seq_len, hidden_state=4096]
            input_pad_list.append(torch.stack(enc_sent_input_pad))

        input_pad = torch.stack(input_pad_list)

        input_pad = input_pad.view(batch_size*N, self.sent_max_len, -1)
        enc_sent_len = enc_sent_len.view(-1)   # [batch_size*N]
        enc_embed_input = self.word_embedding_proj(input_pad)           # [batch_size * N, L, D]

        sent_pos_list = []
        for sentlen in enc_sent_len:
            sent_pos = list(range(1, min(self.sent_max_len, sentlen) + 1))
            for k in range(self.sent_max_len - sentlen):
                sent_pos.append(0)
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long()

        if self._hps.cuda:
            input_pos = input_pos.cuda()
        enc_pos_embed_input = self.position_embedding(input_pos.long()) # [batch_size*N, D]
        enc_conv_input = enc_embed_input + enc_pos_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1) # (batch * N,Ci,L,D)
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs] # kernel_sizes * (batch*N, Co, W)
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output] # kernel_sizes * (batch*N, Co)
        sent_embedding = torch.cat(enc_maxpool_output, 1) # (batch*N, Co * kernel_sizes)
        sent_embedding = sent_embedding.view(batch_size, N, -1)
        return sent_embedding

class BertTagEncoder(BertEncoder):
    def __init__(self, hps, domaindict):
        super(BertTagEncoder, self).__init__(hps)

        # domain embedding
        self.domain_embedding = nn.Embedding(domaindict.size(), hps.domain_emb_dim)
        self.domain_embedding.weight.requires_grad = True

    def forward(self, inputs, input_masks, enc_sent_len, domain):
        sent_embedding = super().forward(inputs, input_masks, enc_sent_len)

        batch_size, N = enc_sent_len.size()

        enc_domain_input = self.domain_embedding(domain)  # [batch, D]
        enc_domain_input = enc_domain_input.unsqueeze(1).expand(batch_size, N, -1)  # [batch, N, D]
        sent_embedding = torch.cat((sent_embedding, enc_domain_input), dim=2)

        return sent_embedding

class ELMoEndoer(nn.Module):
    def __init__(self, hps):
        super(ELMoEndoer, self).__init__()

        self._hps = hps
        self.sent_max_len = hps.sent_max_len

        from allennlp.modules.elmo import Elmo

        elmo_dim = 1024
        options_file = "/remote-home/dqwang/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        weight_file = "/remote-home/dqwang/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

        # elmo_dim = 512
        # options_file = "/remote-home/dqwang/ELMo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        # weight_file = "/remote-home/dqwang/ELMo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

        embed_size = hps.word_emb_dim
        sent_max_len = hps.sent_max_len

        input_channels = 1
        out_channels = hps.output_channel
        min_kernel_size = hps.min_kernel_size
        max_kernel_size = hps.max_kernel_size
        width = embed_size

        # elmo embedding
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        self.embed_proj = nn.Linear(elmo_dim, embed_size)

        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(sent_max_len + 1, embed_size, padding_idx=0), freeze=True)

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size = (height, width)) for height in range(min_kernel_size, max_kernel_size+1)])
        logger.info("[INFO] Initing W for CNN.......")
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))
            fan_in, fan_out = Encoder.calculate_fan_in_and_fan_out(conv.weight.data)
            std = np.sqrt(init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            logger.error("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")
            raise ValueError("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, input):
        # input: a batch of Example object [batch_size, N, seq_len, character_len]

        batch_size, N, seq_len, _ = input.size()
        input = input.view(batch_size * N, seq_len, -1)   # [batch_size*N, seq_len, character_len]
        input_sent_len = ((input.sum(-1)!=0).sum(dim=1)).int()  # [batch_size*N, 1]
        # logger.debug(input_sent_len.view(batch_size, -1))
        enc_embed_input = self.elmo(input)['elmo_representations'][0] # [batch_size*N, L, D]
        enc_embed_input = self.embed_proj(enc_embed_input)

        # input_pos = torch.Tensor([np.hstack((np.arange(1, sentlen + 1), np.zeros(self.sent_max_len - sentlen))) for sentlen in input_sent_len])

        sent_pos_list = []
        for sentlen in input_sent_len:
            sent_pos = list(range(1, min(self.sent_max_len, sentlen) + 1))
            for k in range(self.sent_max_len - sentlen):
                sent_pos.append(0)
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long()

        if self._hps.cuda:
            input_pos = input_pos.cuda()
        enc_pos_embed_input = self.position_embedding(input_pos.long()) # [batch_size*N, D]
        enc_conv_input = enc_embed_input + enc_pos_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1) # (batch * N,Ci,L,D)
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs] # kernel_sizes * (batch*N, Co, W)
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output] # kernel_sizes * (batch*N, Co)
        sent_embedding = torch.cat(enc_maxpool_output, 1) # (batch*N, Co * kernel_sizes)
        sent_embedding = sent_embedding.view(batch_size, N, -1)
        return sent_embedding

class ELMoEndoer2(nn.Module):
    def __init__(self, hps):
        super(ELMoEndoer2, self).__init__()

        self._hps = hps
        self._cuda = hps.cuda
        self.sent_max_len = hps.sent_max_len

        from allennlp.modules.elmo import Elmo

        elmo_dim = 1024
        options_file = "/remote-home/dqwang/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        weight_file = "/remote-home/dqwang/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

        # elmo_dim = 512
        # options_file = "/remote-home/dqwang/ELMo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        # weight_file = "/remote-home/dqwang/ELMo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

        embed_size = hps.word_emb_dim
        sent_max_len = hps.sent_max_len

        input_channels = 1
        out_channels = hps.output_channel
        min_kernel_size = hps.min_kernel_size
        max_kernel_size = hps.max_kernel_size
        width = embed_size

        # elmo embedding
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        self.embed_proj = nn.Linear(elmo_dim, embed_size)

        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(sent_max_len + 1, embed_size, padding_idx=0), freeze=True)

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size = (height, width)) for height in range(min_kernel_size, max_kernel_size+1)])
        logger.info("[INFO] Initing W for CNN.......")
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))
            fan_in, fan_out = Encoder.calculate_fan_in_and_fan_out(conv.weight.data)
            std = np.sqrt(init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            logger.error("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")
            raise ValueError("[Error] Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def pad_encoder_input(self, input_list):
        """
        :param input_list: N [seq_len, hidden_state]
        :return: enc_sent_input_pad: list, N [max_len, hidden_state]
        """
        max_len = self.sent_max_len
        enc_sent_input_pad = []
        _, hidden_size = input_list[0].size()
        for i in range(len(input_list)):
            article_words = input_list[i]  # [seq_len, hidden_size]
            seq_len = article_words.size(0)
            if seq_len > max_len:
                pad_words = article_words[:max_len, :]
            else:
                pad_tensor = torch.zeros(max_len - seq_len, hidden_size).cuda() if self._cuda else torch.zeros(max_len - seq_len, hidden_size)
                pad_words = torch.cat([article_words, pad_tensor], dim=0)
            enc_sent_input_pad.append(pad_words)
        return enc_sent_input_pad

    def forward(self, inputs, input_masks, enc_sent_len):
        """

        :param inputs: a batch of Example object [batch_size, doc_len=512, character_len=50]
        :param input_masks: 0 or 1, [batch, doc_len=512]
        :param enc_sent_len: sentence original length [batch, N]
        :return: 
            sent_embedding: [batch, N, D]
        """

        # Use Bert to get word embedding
        batch_size, N = enc_sent_len.size()
        input_pad_list = []

        elmo_output = self.elmo(inputs)['elmo_representations'][0]  # [batch_size, 512, D]
        elmo_output = elmo_output * input_masks.unsqueeze(-1).float()
        # print("END elmo")

        for i in range(batch_size):
            sent_len = enc_sent_len[i]              # [1, N]
            out = elmo_output[i]

            _, hidden_size = out.size()

            # restore the sentence
            last_end = 0
            enc_sent_input = []
            for length in sent_len:
                if length != 0 and last_end < 512:
                    enc_sent_input.append(out[last_end : min(512, last_end + length), :])
                    last_end += length
                else:
                    pad_tensor = torch.zeros(self.sent_max_len, hidden_size).cuda() if self._hps.cuda else torch.zeros(self.sent_max_len, hidden_size)
                    enc_sent_input.append(pad_tensor)

            # pad the sentence
            enc_sent_input_pad = self.pad_encoder_input(enc_sent_input)  # [N, seq_len, hidden_state=4096]
            input_pad_list.append(torch.stack(enc_sent_input_pad))  # batch * [N, max_len, hidden_state]

        input_pad = torch.stack(input_pad_list)

        input_pad = input_pad.view(batch_size * N, self.sent_max_len, -1)
        enc_sent_len = enc_sent_len.view(-1)  # [batch_size*N]
        enc_embed_input = self.embed_proj(input_pad)  # [batch_size * N, L, D]

        # input_pos = torch.Tensor([np.hstack((np.arange(1, sentlen + 1), np.zeros(self.sent_max_len - sentlen))) for sentlen in input_sent_len])

        sent_pos_list = []
        for sentlen in enc_sent_len:
            sent_pos = list(range(1, min(self.sent_max_len, sentlen) + 1))
            for k in range(self.sent_max_len - sentlen):
                sent_pos.append(0)
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long()

        if self._hps.cuda:
            input_pos = input_pos.cuda()
        enc_pos_embed_input = self.position_embedding(input_pos.long()) # [batch_size*N, D]
        enc_conv_input = enc_embed_input + enc_pos_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1) # (batch * N,Ci,L,D)
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs] # kernel_sizes * (batch*N, Co, W)
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output] # kernel_sizes * (batch*N, Co)
        sent_embedding = torch.cat(enc_maxpool_output, 1) # (batch*N, Co * kernel_sizes)
        sent_embedding = sent_embedding.view(batch_size, N, -1)
        return sent_embedding