r"""
.. todo::
    doc
"""

__all__ = [
    "ElmoEmbedding"
]

import codecs
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contextual_embedding import ContextualEmbedding
from ..core import logger
from ..core.vocabulary import Vocabulary
from ..io.file_utils import cached_path, _get_embedding_url, PRETRAINED_ELMO_MODEL_DIR
from ..modules.encoder._elmo import ElmobiLm, ConvTokenEmbedder


class ElmoEmbedding(ContextualEmbedding):
    r"""
    使用ELMo的embedding。初始化之后，只需要传入words就可以得到对应的embedding。
    当前支持的使用名称初始化的模型:
    
    .. code::
    
        en: 即en-medium hidden_size 1024; output_size 12
        en-medium: hidden_size 2048; output_size 256
        en-origial: hidden_size 4096; output_size 512
        en-original-5.5b: hidden_size 4096; output_size 512
        en-small: hidden_size 1024; output_size 128

    Example::
    
        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import ElmoEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> # 使用不同层的concat的结果
        >>> embed = ElmoEmbedding(vocab, model_dir_or_name='en', layers='1,2', requires_grad=False)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5, 2048])

        >>> # 使用不同层的weighted sum。
        >>> embed = ElmoEmbedding(vocab, model_dir_or_name='en', layers='mix', requires_grad=False)
        >>> embed.set_mix_weights_requires_grad()  # 使得weighted的权重是可以学习的，但ELMO的LSTM部分是不更新

    """
    
    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en', layers: str = '2', requires_grad: bool = True,
                 word_dropout=0.0, dropout=0.0, cache_word_reprs: bool = False):
        r"""
        
        :param vocab: 词表
        :param model_dir_or_name: 可以有两种方式调用预训练好的ELMo embedding：第一种是传入ELMo所在文件夹，该文件夹下面应该有两个文件，
            其中一个是以json为后缀的配置文件，另一个是以pkl为后缀的权重文件；第二种是传入ELMo版本的名称，将自动查看缓存中是否存在该模型，
            没有的话将自动下载并缓存。
        :param layers: str, 指定返回的层数(从0开始), 以,隔开不同的层。如果要返回第二层的结果'2', 返回后两层的结果'1,2'。不同的层的结果
            按照这个顺序concat起来，默认为'2'。'mix'会使用可学习的权重结合不同层的表示(权重是否可训练与requires_grad保持一致，
            初始化权重对三层结果进行mean-pooling, 可以通过ElmoEmbedding.set_mix_weights_requires_grad()方法只将mix weights设置为可学习。)
        :param requires_grad: bool, 该层是否需要gradient, 默认为False.
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param cache_word_reprs: 可以选择对word的表示进行cache; 设置为True的话，将在初始化的时候为每个word生成对应的embedding，
            并删除character encoder，之后将直接使用cache的embedding。默认为False。
        """
        super(ElmoEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        
        # 根据model_dir_or_name检查是否存在并下载
        if model_dir_or_name.lower() in PRETRAINED_ELMO_MODEL_DIR:
            model_url = _get_embedding_url('elmo', model_dir_or_name.lower())
            model_dir = cached_path(model_url, name='embedding')
            # 检查是否存在
        elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_dir = model_dir_or_name
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")
        self.model = _ElmoModel(model_dir, vocab, cache_word_reprs=cache_word_reprs)
        num_layers = self.model.encoder.num_layers
        
        if layers == 'mix':
            self.layer_weights = nn.Parameter(torch.zeros(self.model.config['lstm']['n_layers'] + 1),
                                              requires_grad=requires_grad)
            self.gamma = nn.Parameter(torch.ones(1), requires_grad=requires_grad)
            self._get_outputs = self._get_mixed_outputs
            self._embed_size = self.model.config['lstm']['projection_dim'] * 2
        else:
            layers = list(map(int, layers.split(',')))
            assert len(layers) > 0, "Must choose at least one output, but got None."
            for layer in layers:
                assert 0 <= layer <= num_layers, f"Layer index should be in range [0, {num_layers}], but got {layer}."
            self.layers = layers
            self._get_outputs = self._get_layer_outputs
            self._embed_size = len(self.layers) * self.model.config['lstm']['projection_dim'] * 2
        
        self.requires_grad = requires_grad
    
    def _get_mixed_outputs(self, outputs):
        # outputs: num_layers x batch_size x max_len x hidden_size
        # return: batch_size x max_len x hidden_size
        weights = F.softmax(self.layer_weights + 1 / len(outputs), dim=0).to(outputs)
        outputs = torch.einsum('l,lbij->bij', weights, outputs)
        return self.gamma.to(outputs) * outputs
    
    def set_mix_weights_requires_grad(self, flag=True):
        r"""
        当初始化ElmoEmbedding时layers被设置为mix时，可以通过调用该方法设置mix weights是否可训练。如果layers不是mix，调用
        该方法没有用。
        
        :param bool flag: 混合不同层表示的结果是否可以训练。
        :return:
        """
        if hasattr(self, 'layer_weights'):
            self.layer_weights.requires_grad = flag
            self.gamma.requires_grad = flag
    
    def _get_layer_outputs(self, outputs):
        if len(self.layers) == 1:
            outputs = outputs[self.layers[0]]
        else:
            outputs = torch.cat(tuple([*outputs[self.layers]]), dim=-1)
        
        return outputs
    
    def forward(self, words: torch.LongTensor):
        r"""
        计算words的elmo embedding表示。根据elmo文章中介绍的ELMO实际上是有2L+1层结果，但是为了让结果比较容易拆分，token的
        被重复了一次，使得实际上layer=0的结果是[token_embedding;token_embedding], 而layer=1的结果是[forward_hiddens;
        backward_hiddens].

        :param words: batch_size x max_len
        :return: torch.FloatTensor. batch_size x max_len x (512*len(self.layers))
        """
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = self._get_outputs(outputs)
        return self.dropout(outputs)
    
    def _delete_model_weights(self):
        for name in ['layers', 'model', 'layer_weights', 'gamma']:
            if hasattr(self, name):
                delattr(self, name)


class _ElmoModel(nn.Module):
    r"""
    该Module是ElmoEmbedding中进行所有的heavy lifting的地方。做的工作，包括
        (1) 根据配置，加载模型;
        (2) 根据vocab，对模型中的embedding进行调整. 并将其正确初始化
        (3) 保存一个words与chars的对应转换，获取时自动进行相应的转换
        (4) 设计一个保存token的embedding，允许缓存word的表示。

    """
    
    def __init__(self, model_dir: str, vocab: Vocabulary = None, cache_word_reprs: bool = False):
        super(_ElmoModel, self).__init__()
        self.model_dir = model_dir
        dir = os.walk(self.model_dir)
        config_file = None
        weight_file = None
        config_count = 0
        weight_count = 0
        for path, dir_list, file_list in dir:
            for file_name in file_list:
                if file_name.__contains__(".json"):
                    config_file = file_name
                    config_count += 1
                elif file_name.__contains__(".pkl"):
                    weight_file = file_name
                    weight_count += 1
        if config_count > 1 or weight_count > 1:
            raise Exception(f"Multiple config files(*.json) or weight files(*.hdf5) detected in {model_dir}.")
        elif config_count == 0 or weight_count == 0:
            raise Exception(f"No config file or weight file found in {model_dir}")
        with open(os.path.join(model_dir, config_file), 'r') as config_f:
            config = json.load(config_f)
        self.weight_file = os.path.join(model_dir, weight_file)
        self.config = config
        
        OOV_TAG = '<oov>'
        PAD_TAG = '<pad>'
        BOS_TAG = '<bos>'
        EOS_TAG = '<eos>'
        BOW_TAG = '<bow>'
        EOW_TAG = '<eow>'
        
        # For the model trained with character-based word encoder.
        char_lexicon = {}
        with codecs.open(os.path.join(model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                char_lexicon[token] = int(i)
        
        # 做一些sanity check
        for special_word in [PAD_TAG, OOV_TAG, BOW_TAG, EOW_TAG]:
            assert special_word in char_lexicon, f"{special_word} not found in char.dic."
        
        # 从vocab中构建char_vocab
        char_vocab = Vocabulary(unknown=OOV_TAG, padding=PAD_TAG)
        # 需要保证<bow>与<eow>在里面
        char_vocab.add_word_lst([BOW_TAG, EOW_TAG, BOS_TAG, EOS_TAG])
        
        for word, index in vocab:
            char_vocab.add_word_lst(list(word))
        
        self.bos_index, self.eos_index, self._pad_index = len(vocab), len(vocab) + 1, vocab.padding_idx
        # 根据char_lexicon调整, 多设置一位，是预留给word padding的(该位置的char表示为全0表示)
        char_emb_layer = nn.Embedding(len(char_vocab) + 1, int(config['char_cnn']['embedding']['dim']),
                                      padding_idx=len(char_vocab))
        
        # 读入预训练权重 这里的elmo_model 包含char_cnn和 lstm 的 state_dict
        elmo_model = torch.load(os.path.join(self.model_dir, weight_file), map_location='cpu')
        
        char_embed_weights = elmo_model["char_cnn"]['char_emb_layer.weight']
        
        found_char_count = 0
        for char, index in char_vocab:  # 调整character embedding
            if char in char_lexicon:
                index_in_pre = char_lexicon.get(char)
                found_char_count += 1
            else:
                index_in_pre = char_lexicon[OOV_TAG]
            char_emb_layer.weight.data[index] = char_embed_weights[index_in_pre]
        
        logger.info(f"{found_char_count} out of {len(char_vocab)} characters were found in pretrained elmo embedding.")
        # 生成words到chars的映射
        max_chars = config['char_cnn']['max_characters_per_token']
        self.register_buffer('words_to_chars_embedding', torch.full((len(vocab) + 2, max_chars),
                                                                fill_value=len(char_vocab),
                                                                dtype=torch.long))
        for word, index in list(iter(vocab)) + [(BOS_TAG, len(vocab)), (EOS_TAG, len(vocab) + 1)]:
            if len(word) + 2 > max_chars:
                word = word[:max_chars - 2]
            if index == self._pad_index:
                continue
            elif word == BOS_TAG or word == EOS_TAG:
                char_ids = [char_vocab.to_index(BOW_TAG)] + [char_vocab.to_index(word)] + [
                    char_vocab.to_index(EOW_TAG)]
                char_ids += [char_vocab.to_index(PAD_TAG)] * (max_chars - len(char_ids))
            else:
                char_ids = [char_vocab.to_index(BOW_TAG)] + [char_vocab.to_index(c) for c in word] + [
                    char_vocab.to_index(EOW_TAG)]
                char_ids += [char_vocab.to_index(PAD_TAG)] * (max_chars - len(char_ids))
            self.words_to_chars_embedding[index] = torch.LongTensor(char_ids)
        
        self.char_vocab = char_vocab
        
        self.token_embedder = ConvTokenEmbedder(
            config, self.weight_file, None, char_emb_layer)
        elmo_model["char_cnn"]['char_emb_layer.weight'] = char_emb_layer.weight
        self.token_embedder.load_state_dict(elmo_model["char_cnn"])
        
        self.output_dim = config['lstm']['projection_dim']
        
        # lstm encoder
        self.encoder = ElmobiLm(config)
        self.encoder.load_state_dict(elmo_model["lstm"])
        
        if cache_word_reprs:
            if config['char_cnn']['embedding']['dim'] > 0:  # 只有在使用了chars的情况下有用
                logger.info("Start to generate cache word representations.")
                batch_size = 320
                # bos eos
                word_size = self.words_to_chars_embedding.size(0)
                num_batches = word_size // batch_size + \
                              int(word_size % batch_size != 0)
                
                self.cached_word_embedding = nn.Embedding(word_size,
                                                          config['lstm']['projection_dim'])
                with torch.no_grad():
                    for i in range(num_batches):
                        words = torch.arange(i * batch_size,
                                             min((i + 1) * batch_size, word_size)).long()
                        chars = self.words_to_chars_embedding[words].unsqueeze(1)  # batch_size x 1 x max_chars
                        word_reprs = self.token_embedder(words.unsqueeze(1),
                                                         chars).detach()  # batch_size x 1 x config['encoder']['projection_dim']
                        self.cached_word_embedding.weight.data[words] = word_reprs.squeeze(1)
                    
                    logger.info("Finish generating cached word representations. Going to delete the character encoder.")
                del self.token_embedder, self.words_to_chars_embedding
            else:
                logger.info("There is no need to cache word representations, since no character information is used.")
    
    def forward(self, words):
        r"""

        :param words: batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size
        """
        # 扩展<bos>, <eos>
        batch_size, max_len = words.size()
        expanded_words = words.new_zeros(batch_size, max_len + 2)  # 因为pad一定为0，
        seq_len = words.ne(self._pad_index).sum(dim=-1)
        expanded_words[:, 1:-1] = words
        expanded_words[:, 0].fill_(self.bos_index)
        expanded_words[torch.arange(batch_size).to(words), seq_len + 1] = self.eos_index
        seq_len = seq_len + 2
        zero_tensor = expanded_words.new_zeros(expanded_words.shape)
        mask = (expanded_words == zero_tensor).unsqueeze(-1)
        if hasattr(self, 'cached_word_embedding'):
            token_embedding = self.cached_word_embedding(expanded_words)
        else:
            if hasattr(self, 'words_to_chars_embedding'):
                chars = self.words_to_chars_embedding[expanded_words]
            else:
                chars = None
            token_embedding = self.token_embedder(expanded_words, chars)  # batch_size x max_len x embed_dim
        
        encoder_output = self.encoder(token_embedding, seq_len)
        if encoder_output.size(2) < max_len + 2:
            num_layers, _, output_len, hidden_size = encoder_output.size()
            dummy_tensor = encoder_output.new_zeros(num_layers, batch_size,
                                                    max_len + 2 - output_len, hidden_size)
            encoder_output = torch.cat((encoder_output, dummy_tensor), 2)
        sz = encoder_output.size()  # 2, batch_size, max_len, hidden_size
        token_embedding = token_embedding.masked_fill(mask, 0)
        token_embedding = torch.cat((token_embedding, token_embedding), dim=2).view(1, sz[1], sz[2], sz[3])
        encoder_output = torch.cat((token_embedding, encoder_output), dim=0)
        
        # 删除<eos>, <bos>. 这里没有精确地删除，但应该也不会影响最后的结果了。
        encoder_output = encoder_output[:, :, 1:-1]
        return encoder_output
