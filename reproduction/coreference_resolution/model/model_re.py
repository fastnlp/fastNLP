import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from allennlp.commands.elmo import ElmoEmbedder
from fastNLP.models.base_model import BaseModel
from fastNLP.modules.encoder.variational_rnn import VarLSTM
from reproduction.coreference_resolution.model import preprocess
from fastNLP.io.embed_loader import EmbedLoader
from fastNLP.core.const import Const
import random

# 设置seed
torch.manual_seed(0)  # cpu
torch.cuda.manual_seed(0)  # gpu
np.random.seed(0) # numpy
random.seed(0)


class ffnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ffnn, self).__init__()

        self.f = nn.Sequential(
            # 多少层数
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, output_size)
        )
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
                # param.data = torch.tensor(np.random.randn(*param.shape)).float()
            else:
                nn.init.zeros_(param)

    def forward(self, input):
        return self.f(input).squeeze()


class Model(BaseModel):
    def __init__(self, vocab, config):
        word2id = vocab.word2idx
        super(Model, self).__init__()
        vocab_num = len(word2id)
        self.word2id = word2id
        self.config = config
        self.char_dict = preprocess.get_char_dict('data/char_vocab.english.txt')
        self.genres = {g: i for i, g in enumerate(["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}
        self.device = torch.device("cuda:" + config.cuda)

        self.emb = nn.Embedding(vocab_num, 350)

        emb1 = EmbedLoader().load_with_vocab(config.glove, vocab,normalize=False)
        emb2 = EmbedLoader().load_with_vocab(config.turian,  vocab ,normalize=False)
        pre_emb = np.concatenate((emb1, emb2), axis=1)
        pre_emb /= (np.linalg.norm(pre_emb, axis=1, keepdims=True) + 1e-12)

        if pre_emb is not None:
            self.emb.weight = nn.Parameter(torch.from_numpy(pre_emb).float())
            for param in self.emb.parameters():
                param.requires_grad = False
        self.emb_dropout = nn.Dropout(inplace=True)


        if config.use_elmo:
            self.elmo = ElmoEmbedder(options_file='data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                                     weight_file='data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                                     cuda_device=int(config.cuda))
            print("elmo load over.")
            self.elmo_args = torch.randn((3), requires_grad=True).to(self.device)

        self.char_emb = nn.Embedding(len(self.char_dict), config.char_emb_size)
        self.conv1 = nn.Conv1d(config.char_emb_size, 50, 3)
        self.conv2 = nn.Conv1d(config.char_emb_size, 50, 4)
        self.conv3 = nn.Conv1d(config.char_emb_size, 50, 5)

        self.feature_emb = nn.Embedding(config.span_width, config.feature_size)
        self.feature_emb_dropout = nn.Dropout(p=0.2, inplace=True)

        self.mention_distance_emb = nn.Embedding(10, config.feature_size)
        self.distance_drop = nn.Dropout(p=0.2, inplace=True)

        self.genre_emb = nn.Embedding(7, config.feature_size)
        self.speaker_emb = nn.Embedding(2, config.feature_size)

        self.bilstm = VarLSTM(input_size=350+150*config.use_CNN+config.use_elmo*1024,hidden_size=200,bidirectional=True,batch_first=True,hidden_dropout=0.2)
        # self.bilstm = nn.LSTM(input_size=500, hidden_size=200, bidirectional=True, batch_first=True)
        self.h0 = nn.init.orthogonal_(torch.empty(2, 1, 200)).to(self.device)
        self.c0 = nn.init.orthogonal_(torch.empty(2, 1, 200)).to(self.device)
        self.bilstm_drop = nn.Dropout(p=0.2, inplace=True)

        self.atten = ffnn(input_size=400, hidden_size=config.atten_hidden_size, output_size=1)
        self.mention_score = ffnn(input_size=1320, hidden_size=config.mention_hidden_size, output_size=1)
        self.sa = ffnn(input_size=3980+40*config.use_metadata, hidden_size=config.sa_hidden_size, output_size=1)
        self.mention_start_np = None
        self.mention_end_np = None

    def _reorder_lstm(self, word_emb, seq_lens):
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
        seq_lens_re = [seq_lens[i] for i in sort_ind]
        emb_seq = self.reorder_sequence(word_emb, sort_ind, batch_first=True)
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_seq, seq_lens_re, batch_first=True)

        h0 = self.h0.repeat(1, len(seq_lens), 1)
        c0 = self.c0.repeat(1, len(seq_lens), 1)
        packed_out, final_states = self.bilstm(packed_seq, (h0, c0))

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens_re))]
        lstm_out = self.reorder_sequence(lstm_out, reorder_ind, batch_first=True)
        return lstm_out

    def reorder_sequence(self, sequence_emb, order, batch_first=True):
        """
        sequence_emb: [T, B, D] if not batch_first
        order: list of sequence length
        """
        batch_dim = 0 if batch_first else 1
        assert len(order) == sequence_emb.size()[batch_dim]

        order = torch.LongTensor(order)
        order = order.to(sequence_emb).long()

        sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

        del order
        return sorted_

    def flat_lstm(self, lstm_out, seq_lens):
        batch = lstm_out.shape[0]
        seq = lstm_out.shape[1]
        dim = lstm_out.shape[2]
        l = [j + i * seq for i, seq_len in enumerate(seq_lens) for j in range(seq_len)]
        flatted = torch.index_select(lstm_out.view(batch * seq, dim), 0, torch.LongTensor(l).to(self.device))
        return flatted

    def potential_mention_index(self, word_index, max_sent_len):
        # get mention index [3,2]:the first sentence is 3 and secend 2
        # [0,0,0,1,1] --> [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [3, 3], [3, 4], [4, 4]] (max =2)
        potential_mention = []
        for i in range(len(word_index)):
            for j in range(i, i + max_sent_len):
                if (j < len(word_index) and word_index[i] == word_index[j]):
                    potential_mention.append([i, j])
        return potential_mention

    def get_mention_start_end(self, seq_lens):
        # 序列长度转换成mention
        # [3,2] --> [0,0,0,1,1]
        word_index = [0] * sum(seq_lens)
        sent_index = 0
        index = 0
        for length in seq_lens:
            for l in range(length):
                word_index[index] = sent_index
                index += 1
            sent_index += 1

        # [0,0,0,1,1]-->[[0,0],[0,1],[0,2]....]
        mention_id = self.potential_mention_index(word_index, self.config.span_width)
        mention_start = np.array(mention_id, dtype=int)[:, 0]
        mention_end = np.array(mention_id, dtype=int)[:, 1]
        return mention_start, mention_end

    def get_mention_emb(self, flatten_lstm, mention_start, mention_end):
        mention_start_tensor = torch.from_numpy(mention_start).to(self.device)
        mention_end_tensor = torch.from_numpy(mention_end).to(self.device)
        emb_start = flatten_lstm.index_select(dim=0, index=mention_start_tensor)  # [mention_num,embed]
        emb_end = flatten_lstm.index_select(dim=0, index=mention_end_tensor)  # [mention_num,embed]
        return emb_start, emb_end

    def get_mask(self, mention_start, mention_end):
        # big mask for attention
        mention_num = mention_start.shape[0]
        mask = np.zeros((mention_num, self.config.span_width))  # [mention_num,span_width]
        for i in range(mention_num):
            start = mention_start[i]
            end = mention_end[i]
            # 实际上是宽度
            for j in range(end - start + 1):
                mask[i][j] = 1
        mask = torch.from_numpy(mask)  # [mention_num,max_mention]
        # 0-->-inf  1-->0
        log_mask = torch.log(mask)
        return log_mask

    def get_mention_index(self, mention_start, max_mention):
        # TODO 后面可能要改
        assert len(mention_start.shape) == 1
        mention_start_tensor = torch.from_numpy(mention_start)
        num_mention = mention_start_tensor.shape[0]
        mention_index = mention_start_tensor.expand(max_mention, num_mention).transpose(0,
                                                                                        1)  # [num_mention,max_mention]
        assert mention_index.shape[0] == num_mention
        assert mention_index.shape[1] == max_mention
        range_add = torch.arange(0, max_mention).expand(num_mention, max_mention).long()  # [num_mention,max_mention]
        mention_index = mention_index + range_add
        mention_index = torch.min(mention_index, torch.LongTensor([mention_start[-1]]).expand(num_mention, max_mention))
        return mention_index.to(self.device)

    def sort_mention(self, mention_start, mention_end, candidate_mention_emb, candidate_mention_score, seq_lens):
        # 排序记录,高分段在前面
        mention_score, mention_ids = torch.sort(candidate_mention_score, descending=True)
        preserve_mention_num = int(self.config.mention_ratio * sum(seq_lens))
        mention_ids = mention_ids[0:preserve_mention_num]
        mention_score = mention_score[0:preserve_mention_num]

        mention_start_tensor = torch.from_numpy(mention_start).to(self.device).index_select(dim=0,
                                                                                            index=mention_ids)  # [lamda*word_num]
        mention_end_tensor = torch.from_numpy(mention_end).to(self.device).index_select(dim=0,
                                                                                        index=mention_ids)  # [lamda*word_num]
        mention_emb = candidate_mention_emb.index_select(index=mention_ids, dim=0)  # [lamda*word_num,emb]
        assert mention_score.shape[0] == preserve_mention_num
        assert mention_start_tensor.shape[0] == preserve_mention_num
        assert mention_end_tensor.shape[0] == preserve_mention_num
        assert mention_emb.shape[0] == preserve_mention_num
        # TODO 不交叉没做处理

        # 对start进行再排序，实际位置在前面
        # TODO 这里只考虑了start没有考虑end
        mention_start_tensor, temp_index = torch.sort(mention_start_tensor)
        mention_end_tensor = mention_end_tensor.index_select(dim=0, index=temp_index)
        mention_emb = mention_emb.index_select(dim=0, index=temp_index)
        mention_score = mention_score.index_select(dim=0, index=temp_index)
        return mention_start_tensor, mention_end_tensor, mention_score, mention_emb

    def get_antecedents(self, mention_starts, max_antecedents):
        num_mention = mention_starts.shape[0]
        max_antecedents = min(max_antecedents, num_mention)
        # mention和它是第几个mention之间的对应关系
        antecedents = np.zeros((num_mention, max_antecedents), dtype=int)  # [num_mention,max_an]
        # 记录长度
        antecedents_len = [0] * num_mention
        for i in range(num_mention):
            ante_count = 0
            for j in range(max(0, i - max_antecedents), i):
                antecedents[i, ante_count] = j
                ante_count += 1
            # 补位操作
            for j in range(ante_count, max_antecedents):
                antecedents[i, j] = 0
            antecedents_len[i] = ante_count
        assert antecedents.shape[1] == max_antecedents
        return antecedents, antecedents_len

    def get_antecedents_score(self, span_represent, mention_score, antecedents, antecedents_len, mention_speakers_ids,
                              genre):
        num_mention = mention_score.shape[0]
        max_antecedent = antecedents.shape[1]

        pair_emb = self.get_pair_emb(span_represent, antecedents, mention_speakers_ids, genre)  # [span_num,max_ant,emb]
        antecedent_scores = self.sa(pair_emb)
        mask01 = self.sequence_mask(antecedents_len, max_antecedent)
        maskinf = torch.log(mask01).to(self.device)
        assert maskinf.shape[1] <= max_antecedent
        assert antecedent_scores.shape[0] == num_mention
        antecedent_scores = antecedent_scores + maskinf
        antecedents = torch.from_numpy(antecedents).to(self.device)
        mention_scoreij = mention_score.unsqueeze(1) + torch.gather(
            mention_score.unsqueeze(0).expand(num_mention, num_mention), dim=1, index=antecedents)
        antecedent_scores += mention_scoreij

        antecedent_scores = torch.cat([torch.zeros([mention_score.shape[0], 1]).to(self.device), antecedent_scores],
                                      1)  # [num_mentions, max_ant + 1]
        return antecedent_scores

    ##############################
    def distance_bin(self, mention_distance):
        bins = torch.zeros(mention_distance.size()).byte().to(self.device)
        rg = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 7], [8, 15], [16, 31], [32, 63], [64, 300]]
        for t, k in enumerate(rg):
            i, j = k[0], k[1]
            b = torch.LongTensor([i]).unsqueeze(-1).expand(mention_distance.size()).to(self.device)
            m1 = torch.ge(mention_distance, b)
            e = torch.LongTensor([j]).unsqueeze(-1).expand(mention_distance.size()).to(self.device)
            m2 = torch.le(mention_distance, e)
            bins = bins + (t + 1) * (m1 & m2)
        return bins.long()

    def get_distance_emb(self, antecedents_tensor):
        num_mention = antecedents_tensor.shape[0]
        max_ant = antecedents_tensor.shape[1]

        assert max_ant <= self.config.max_antecedents
        source = torch.arange(0, num_mention).expand(max_ant, num_mention).transpose(0,1).to(self.device)  # [num_mention,max_ant]
        mention_distance = source - antecedents_tensor
        mention_distance_bin = self.distance_bin(mention_distance)
        distance_emb = self.mention_distance_emb(mention_distance_bin)
        distance_emb = self.distance_drop(distance_emb)
        return distance_emb

    def get_pair_emb(self, span_emb, antecedents, mention_speakers_ids, genre):
        emb_dim = span_emb.shape[1]
        num_span = span_emb.shape[0]
        max_ant = antecedents.shape[1]
        assert span_emb.shape[0] == antecedents.shape[0]
        antecedents = torch.from_numpy(antecedents).to(self.device)

        # [num_span,max_ant,emb]
        antecedent_emb = torch.gather(span_emb.unsqueeze(0).expand(num_span, num_span, emb_dim), dim=1,
                                      index=antecedents.unsqueeze(2).expand(num_span, max_ant, emb_dim))
        # [num_span,max_ant,emb]
        target_emb_tiled = span_emb.expand((max_ant, num_span, emb_dim))
        target_emb_tiled = target_emb_tiled.transpose(0, 1)

        similarity_emb = antecedent_emb * target_emb_tiled

        pair_emb_list = [target_emb_tiled, antecedent_emb, similarity_emb]

        # get speakers and genre
        if self.config.use_metadata:
            antecedent_speaker_ids = mention_speakers_ids.unsqueeze(0).expand(num_span, num_span).gather(dim=1,
                                                                                                         index=antecedents)
            same_speaker = torch.eq(mention_speakers_ids.unsqueeze(1).expand(num_span, max_ant),
                                    antecedent_speaker_ids)  # [num_mention,max_ant]
            speaker_embedding = self.speaker_emb(same_speaker.long().to(self.device))  # [mention_num.max_ant,emb]
            genre_embedding = self.genre_emb(
                torch.LongTensor([genre]).expand(num_span, max_ant).to(self.device))  # [mention_num,max_ant,emb]
            pair_emb_list.append(speaker_embedding)
            pair_emb_list.append(genre_embedding)

        # get distance emb
        if self.config.use_distance:
            distance_emb = self.get_distance_emb(antecedents)
            pair_emb_list.append(distance_emb)

        pair_emb = torch.cat(pair_emb_list, 2)
        return pair_emb

    def sequence_mask(self, len_list, max_len):
        x = np.zeros((len(len_list), max_len))
        for i in range(len(len_list)):
            l = len_list[i]
            for j in range(l):
                x[i][j] = 1
        return torch.from_numpy(x).float()

    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                           dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))

            return m + torch.log(sum_exp)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        antecedent_labels = torch.from_numpy(antecedent_labels * 1).to(self.device)
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())  # [num_mentions, max_ant + 1]
        marginalized_gold_scores = self.logsumexp(gold_scores, 1)  # [num_mentions]
        log_norm = self.logsumexp(antecedent_scores, 1)  # [num_mentions]
        return torch.sum(log_norm - marginalized_gold_scores)  # [num_mentions]reduce_logsumexp

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores.detach(), axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, mention_starts, mention_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(mention_starts[predicted_index]), int(mention_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(mention_starts[i]), int(mention_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, mention_starts, mention_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_starts, mention_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters


    def forward(self, words1 , words2, words3, words4, chars, seq_len):
        """
        实际输入都是tensor
        :param sentences: 句子，被fastNLP转化成了numpy，
        :param doc_np: 被fastNLP转化成了Tensor
        :param speaker_ids_np: 被fastNLP转化成了Tensor
        :param genre: 被fastNLP转化成了Tensor
        :param char_index: 被fastNLP转化成了Tensor
        :param seq_len: 被fastNLP转化成了Tensor
        :return:
        """

        sentences = words3
        doc_np = words4
        speaker_ids_np = words2
        genre = words1
        char_index = chars


        # change for fastNLP
        sentences = sentences[0].tolist()
        doc_tensor = doc_np[0]
        speakers_tensor = speaker_ids_np[0]
        genre = genre[0].item()
        char_index = char_index[0]
        seq_len = seq_len[0].cpu().numpy()

        # 类型

        # doc_tensor = torch.from_numpy(doc_np).to(self.device)
        # speakers_tensor = torch.from_numpy(speaker_ids_np).to(self.device)
        mention_emb_list = []

        word_emb = self.emb(doc_tensor)
        word_emb_list = [word_emb]
        if self.config.use_CNN:
            # [batch, length, char_length, char_dim]
            char = self.char_emb(char_index)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)

            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char_over_cnn, _ = self.conv1(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char_over_cnn = torch.tanh(char_over_cnn).view(char_size[0], char_size[1], -1)
            word_emb_list.append(char_over_cnn)

            char_over_cnn, _ = self.conv2(char).max(dim=2)
            char_over_cnn = torch.tanh(char_over_cnn).view(char_size[0], char_size[1], -1)
            word_emb_list.append(char_over_cnn)

            char_over_cnn, _ = self.conv3(char).max(dim=2)
            char_over_cnn = torch.tanh(char_over_cnn).view(char_size[0], char_size[1], -1)
            word_emb_list.append(char_over_cnn)

        # word_emb = torch.cat(word_emb_list, dim=2)

        # use elmo or not
        if self.config.use_elmo:
            # 如果确实被截断了
            if doc_tensor.shape[0] == 50 and len(sentences) > 50:
                sentences = sentences[0:50]
            elmo_embedding, elmo_mask = self.elmo.batch_to_embeddings(sentences)
            elmo_embedding = elmo_embedding.to(
                self.device)  # [sentence_num,max_sent_len,3,1024]--[sentence_num,max_sent,1024]
            elmo_embedding = elmo_embedding[:, 0, :, :] * self.elmo_args[0] + elmo_embedding[:, 1, :, :] * \
                             self.elmo_args[1] + elmo_embedding[:, 2, :, :] * self.elmo_args[2]
            word_emb_list.append(elmo_embedding)
        # print(word_emb_list[0].shape)
        # print(word_emb_list[1].shape)
        # print(word_emb_list[2].shape)
        # print(word_emb_list[3].shape)
        # print(word_emb_list[4].shape)

        word_emb = torch.cat(word_emb_list, dim=2)

        word_emb = self.emb_dropout(word_emb)
        # word_emb_elmo = self.emb_dropout(word_emb_elmo)
        lstm_out = self._reorder_lstm(word_emb, seq_len)
        flatten_lstm = self.flat_lstm(lstm_out, seq_len)  # [word_num,emb]
        flatten_lstm = self.bilstm_drop(flatten_lstm)
        # TODO 没有按照论文写
        flatten_word_emb = self.flat_lstm(word_emb, seq_len)  # [word_num,emb]

        mention_start, mention_end = self.get_mention_start_end(seq_len)  # [mention_num]
        self.mention_start_np = mention_start  # [mention_num] np
        self.mention_end_np = mention_end
        mention_num = mention_start.shape[0]
        emb_start, emb_end = self.get_mention_emb(flatten_lstm, mention_start, mention_end)  # [mention_num,emb]

        # list
        mention_emb_list.append(emb_start)
        mention_emb_list.append(emb_end)

        if self.config.use_width:
            mention_width_index = mention_end - mention_start
            mention_width_tensor = torch.from_numpy(mention_width_index).to(self.device)  # [mention_num]
            mention_width_emb = self.feature_emb(mention_width_tensor)
            mention_width_emb = self.feature_emb_dropout(mention_width_emb)
            mention_emb_list.append(mention_width_emb)

        if self.config.model_heads:
            mention_index = self.get_mention_index(mention_start, self.config.span_width)  # [mention_num,max_mention]
            log_mask_tensor = self.get_mask(mention_start, mention_end).float().to(
                self.device)  # [mention_num,max_mention]
            alpha = self.atten(flatten_lstm).to(self.device)  # [word_num]

            # 得到attention
            mention_head_score = torch.gather(alpha.expand(mention_num, -1), 1,
                                              mention_index).float().to(self.device)  # [mention_num,max_mention]
            mention_attention = F.softmax(mention_head_score + log_mask_tensor, dim=1)  # [mention_num,max_mention]

            # TODO flatte lstm
            word_num = flatten_lstm.shape[0]
            lstm_emb = flatten_lstm.shape[1]
            emb_num = flatten_word_emb.shape[1]

            # [num_mentions, max_mention_width, emb]
            mention_text_emb = torch.gather(
                flatten_word_emb.unsqueeze(1).expand(word_num, self.config.span_width, emb_num),
                0, mention_index.unsqueeze(2).expand(mention_num, self.config.span_width,
                                                     emb_num))
            # [mention_num,emb]
            mention_head_emb = torch.sum(
                mention_attention.unsqueeze(2).expand(mention_num, self.config.span_width, emb_num) * mention_text_emb,
                dim=1)
            mention_emb_list.append(mention_head_emb)

        candidate_mention_emb = torch.cat(mention_emb_list, 1)  # [candidate_mention_num,emb]
        candidate_mention_score = self.mention_score(candidate_mention_emb)  # [candidate_mention_num]

        antecedent_scores, antecedents, mention_start_tensor, mention_end_tensor = (None, None, None, None)
        mention_start_tensor, mention_end_tensor, mention_score, mention_emb = \
            self.sort_mention(mention_start, mention_end, candidate_mention_emb, candidate_mention_score, seq_len)
        mention_speakers_ids = speakers_tensor.index_select(dim=0, index=mention_start_tensor)  # num_mention

        antecedents, antecedents_len = self.get_antecedents(mention_start_tensor, self.config.max_antecedents)
        antecedent_scores = self.get_antecedents_score(mention_emb, mention_score, antecedents, antecedents_len,
                                                       mention_speakers_ids, genre)

        ans = {"candidate_mention_score": candidate_mention_score, "antecedent_scores": antecedent_scores,
               "antecedents": antecedents, "mention_start_tensor": mention_start_tensor,
               "mention_end_tensor": mention_end_tensor}

        return ans

    def predict(self, sentences, doc_np, speaker_ids_np, genre, char_index, seq_len):
        ans = self(sentences,
                   doc_np,
                   speaker_ids_np,
                   genre,
                   char_index,
                   seq_len)

        predicted_antecedents = self.get_predicted_antecedents(ans["antecedents"], ans["antecedent_scores"])
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(ans["mention_start_tensor"],
                                                                               ans["mention_end_tensor"],
                                                                               predicted_antecedents)

        return {'predicted':predicted_clusters,"mention_to_predicted":mention_to_predicted}


if __name__ == '__main__':
    pass
