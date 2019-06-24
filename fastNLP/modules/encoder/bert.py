
import os
from torch import nn
import torch
from ...io.file_utils import _get_base_url, cached_path
from ._bert import _WordPieceBertModel, BertModel

class BertWordPieceEncoder(nn.Module):
    """
    读取bert模型，读取之后调用index_dataset方法在dataset中生成word_pieces这一列。

    :param fastNLP.Vocabulary vocab: 词表
    :param str model_dir_or_name: 模型所在目录或者模型的名称。默认值为``en-base-uncased``
    :param str layers:最终结果中的表示。以','隔开层数，可以以负数去索引倒数几层
    :param bool requires_grad: 是否需要gradient。
    """
    def __init__(self, model_dir_or_name:str='en-base-uncased', layers:str='-1',
                 requires_grad:bool=False):
        super().__init__()
        PRETRAIN_URL = _get_base_url('bert')
        PRETRAINED_BERT_MODEL_DIR = {'en': 'bert-base-cased-f89bfe08.zip',
                                     'en-base-uncased': 'bert-base-uncased-3413b23c.zip',
                                     'en-base-cased': 'bert-base-cased-f89bfe08.zip',
                                     'en-large-uncased': 'bert-large-uncased-20939f45.zip',
                                     'en-large-cased': 'bert-large-cased-e0cf90fc.zip',

                                     'cn': 'bert-base-chinese-29d0a84a.zip',
                                     'cn-base': 'bert-base-chinese-29d0a84a.zip',

                                     'multilingual': 'bert-base-multilingual-cased-1bd364ee.zip',
                                     'multilingual-base-uncased': 'bert-base-multilingual-uncased-f8730fe4.zip',
                                     'multilingual-base-cased': 'bert-base-multilingual-cased-1bd364ee.zip',
                                     }

        if model_dir_or_name in PRETRAINED_BERT_MODEL_DIR:
            model_name = PRETRAINED_BERT_MODEL_DIR[model_dir_or_name]
            model_url = PRETRAIN_URL + model_name
            model_dir = cached_path(model_url)
            # 检查是否存在
        elif os.path.isdir(model_dir_or_name):
            model_dir = model_dir_or_name
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        self.model = _WordPieceBertModel(model_dir=model_dir, layers=layers)
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        """
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        requires_grads = set([param.requires_grad for name, param in self.named_parameters()])
        if len(requires_grads)==1:
            return requires_grads.pop()
        else:
            return None

    @requires_grad.setter
    def requires_grad(self, value):
        for name, param in self.named_parameters():
            param.requires_grad = value

    @property
    def embed_size(self):
        return self._embed_size

    def index_datasets(self, *datasets, field_name):
        """
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input。如果首尾不是
            [CLS]与[SEP]会在首尾额外加入[CLS]与[SEP], 且将word_pieces这一列的pad value设置为了bert的pad value。

        :param datasets: DataSet对象
        :param field_name: 基于哪一列的内容生成word_pieces列。这一列中每个数据应该是List[str]的形式。
        :return:
        """
        self.model.index_dataset(*datasets, field_name=field_name)

    def forward(self, word_pieces, token_type_ids=None):
        """
        计算words的bert embedding表示。传入的words中应该自行包含[CLS]与[SEP]的tag。

        :param words: batch_size x max_len
        :param token_type_ids: batch_size x max_len, 用于区分前一句和后一句话
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        outputs = self.model(word_pieces, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)

        return outputs