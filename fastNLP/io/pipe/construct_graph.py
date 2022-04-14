__all__ = [
    'MRPmiGraphPipe',
    'R8PmiGraphPipe',
    'R52PmiGraphPipe',
    'OhsumedPmiGraphPipe',
    'NG20PmiGraphPipe'
]
try:
    import networkx as nx
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import Pipeline
except:
    pass
from collections import defaultdict
import itertools
import math
import numpy as np

from ..data_bundle import DataBundle
# from ...core.const import Const
from ..loader.classification import MRLoader, OhsumedLoader, R52Loader, R8Loader, NG20Loader
from fastNLP.core.utils import f_rich_progress


def _get_windows(content_lst: list, window_size: int):
    r"""
        滑动窗口处理文本，获取词频和共现词语的词频
        :param content_lst:
        :param window_size:
        :return: 词频，共现词频，窗口化后文本段的数量
    """
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    task_id = f_rich_progress.add_task(description="Split by window", total=len(content_lst))
    for words in content_lst:
        windows = list()

        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)

        f_rich_progress.update(task_id, advance=1)
    f_rich_progress.destroy_task(task_id)
    return word_window_freq, word_pair_count, windows_len


def _cal_pmi(W_ij, W, word_freq_i, word_freq_j):
    r"""
        params: w_ij:为词语i,j的共现词频
                w:文本数量
                word_freq_i: 词语i的词频
                word_freq_j: 词语j的词频
        return: 词语i，j的tfidf值
    """
    p_i = word_freq_i / W
    p_j = word_freq_j / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def _count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    r"""
        params: windows_len: 文本段数量
                word_pair_count: 词共现频率字典
                word_window_freq: 词频率字典
                threshold: 阈值
        return 词语pmi的list列表,其中元素为[word1, word2, pmi]
    """
    word_pmi_lst = list()
    task_id = f_rich_progress.add_task(description="Calculate pmi between words", total=len(word_pair_count))
    for word_pair, W_i_j in word_pair_count.items():
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = _cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])

        f_rich_progress.update(task_id, advance=1)
    f_rich_progress.destory_task(task_id)
    return word_pmi_lst


class GraphBuilderBase:
    def __init__(self, graph_type='pmi', widow_size=10, threshold=0.):
        self.graph = nx.Graph()
        self.word2id = dict()
        self.graph_type = graph_type
        self.window_size = widow_size
        self.doc_node_num = 0
        self.tr_doc_index = None
        self.te_doc_index = None
        self.dev_doc_index = None
        self.doc = None
        self.threshold = threshold

    def _get_doc_edge(self, data_bundle: DataBundle):
        r"""
            对输入的DataBundle进行处理，然后生成文档-单词的tfidf值
            ：param: data_bundle中的文本若为英文，形式为[ 'This is the first document.'],若为中文则为['他 喜欢 吃 苹果']
            : return 返回带有具有tfidf边文档-单词稀疏矩阵
        """
        tr_doc = list(data_bundle.get_dataset("train").get_field('raw_words'))
        val_doc = list(data_bundle.get_dataset("dev").get_field('raw_words'))
        te_doc = list(data_bundle.get_dataset("test").get_field('raw_words'))
        doc = tr_doc + val_doc + te_doc
        self.doc = doc
        self.tr_doc_index = [ind for ind in range(len(tr_doc))]
        self.dev_doc_index = [ind + len(tr_doc) for ind in range(len(val_doc))]
        self.te_doc_index = [ind + len(tr_doc) + len(val_doc) for ind in range(len(te_doc))]
        text_tfidf = Pipeline([('count', CountVectorizer(token_pattern=r'\S+', min_df=1, max_df=1.0)),
                               ('tfidf',
                                TfidfTransformer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False))])

        tfidf_vec = text_tfidf.fit_transform(doc)
        self.doc_node_num = tfidf_vec.shape[0]
        vocab_lst = text_tfidf['count'].get_feature_names()
        for ind, word in enumerate(vocab_lst):
            self.word2id[word] = ind
        for ind, row in enumerate(tfidf_vec):
            for col_index, value in zip(row.indices, row.data):
                self.graph.add_edge(ind, self.doc_node_num + col_index, weight=value)
        return nx.to_scipy_sparse_matrix(self.graph)

    def _get_word_edge(self):
        word_window_freq, word_pair_count, windows_len = _get_windows(self.doc, self.window_size)
        pmi_edge_lst = _count_pmi(windows_len, word_pair_count, word_window_freq, self.threshold)
        for edge_item in pmi_edge_lst:
            word_indx1 = self.doc_node_num + self.word2id[edge_item[0]]
            word_indx2 = self.doc_node_num + self.word2id[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.graph.add_edge(word_indx1, word_indx2, weight=edge_item[2])

    def build_graph(self, data_bundle: DataBundle):
        r"""
            对输入的DataBundle进行处理，然后返回该scipy_sparse_matrix类型的邻接矩阵。

            :param ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象
            :return:
        """
        raise NotImplementedError

    def build_graph_from_file(self, path: str):
        r"""
            传入文件路径，生成处理好的scipy_sparse_matrix对象。paths支持的路径形式可以参考 ：:method:`fastNLP.io.Loader.load()`

            :param path:
            :return: scipy_sparse_matrix
        """
        raise NotImplementedError


class MRPmiGraphPipe(GraphBuilderBase):

    def __init__(self, graph_type='pmi', widow_size=10, threshold=0.):
        super().__init__(graph_type=graph_type, widow_size=widow_size, threshold=threshold)

    def build_graph(self, data_bundle: DataBundle):
        r"""
            params: ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象.
            return 返回csr类型的稀疏矩阵图;训练集，验证集，测试集，在图中的index.
        """
        self._get_doc_edge(data_bundle)
        self._get_word_edge()
        return nx.to_scipy_sparse_matrix(self.graph,
                                         nodelist=list(range(self.graph.number_of_nodes())),
                                         weight='weight', dtype=np.float32, format='csr'), (
               self.tr_doc_index, self.dev_doc_index, self.te_doc_index)

    def build_graph_from_file(self, path: str):
        data_bundle = MRLoader().load(path)
        return self.build_graph(data_bundle)


class R8PmiGraphPipe(GraphBuilderBase):

    def __init__(self, graph_type='pmi', widow_size=10, threshold=0.):
        super().__init__(graph_type=graph_type, widow_size=widow_size, threshold=threshold)

    def build_graph(self, data_bundle: DataBundle):
        r"""
            params: ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象.
            return 返回csr类型的稀疏矩阵图;训练集，验证集，测试集，在图中的index.
        """
        self._get_doc_edge(data_bundle)
        self._get_word_edge()
        return nx.to_scipy_sparse_matrix(self.graph,
                                         nodelist=list(range(self.graph.number_of_nodes())),
                                         weight='weight', dtype=np.float32, format='csr'), (
               self.tr_doc_index, self.dev_doc_index, self.te_doc_index)

    def build_graph_from_file(self, path: str):
        data_bundle = R8Loader().load(path)
        return self.build_graph(data_bundle)


class R52PmiGraphPipe(GraphBuilderBase):

    def __init__(self, graph_type='pmi', widow_size=10, threshold=0.):
        super().__init__(graph_type=graph_type, widow_size=widow_size, threshold=threshold)

    def build_graph(self, data_bundle: DataBundle):
        r"""
            params: ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象.
            return 返回csr类型的稀疏矩阵;训练集，验证集，测试集，在图中的index.
        """
        self._get_doc_edge(data_bundle)
        self._get_word_edge()
        return nx.to_scipy_sparse_matrix(self.graph,
                                         nodelist=list(range(self.graph.number_of_nodes())),
                                         weight='weight', dtype=np.float32, format='csr'), (
               self.tr_doc_index, self.dev_doc_index, self.te_doc_index)

    def build_graph_from_file(self, path: str):
        data_bundle = R52Loader().load(path)
        return self.build_graph(data_bundle)


class OhsumedPmiGraphPipe(GraphBuilderBase):

    def __init__(self, graph_type='pmi', widow_size=10, threshold=0.):
        super().__init__(graph_type=graph_type, widow_size=widow_size, threshold=threshold)

    def build_graph(self, data_bundle: DataBundle):
        r"""
            params: ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象.
            return 返回csr类型的稀疏矩阵图;训练集，验证集，测试集，在图中的index.
        """
        self._get_doc_edge(data_bundle)
        self._get_word_edge()
        return nx.to_scipy_sparse_matrix(self.graph,
                                         nodelist=list(range(self.graph.number_of_nodes())),
                                         weight='weight', dtype=np.float32, format='csr'), (
               self.tr_doc_index, self.dev_doc_index, self.te_doc_index)

    def build_graph_from_file(self, path: str):
        data_bundle = OhsumedLoader().load(path)
        return self.build_graph(data_bundle)


class NG20PmiGraphPipe(GraphBuilderBase):

    def __init__(self, graph_type='pmi', widow_size=10, threshold=0.):
        super().__init__(graph_type=graph_type, widow_size=widow_size, threshold=threshold)

    def build_graph(self, data_bundle: DataBundle):
        r"""
            params: ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象.
            return 返回csr类型的稀疏矩阵图;训练集，验证集，测试集，在图中的index.
        """
        self._get_doc_edge(data_bundle)
        self._get_word_edge()
        return nx.to_scipy_sparse_matrix(self.graph,
                                         nodelist=list(range(self.graph.number_of_nodes())),
                                         weight='weight', dtype=np.float32, format='csr'), (
                   self.tr_doc_index, self.dev_doc_index, self.te_doc_index)

    def build_graph_from_file(self, path: str):
        r"""
            param: path->数据集的路径.
            return: 返回csr类型的稀疏矩阵图;训练集，验证集，测试集，在图中的index.
        """
        data_bundle = NG20Loader().load(path)
        return self.build_graph(data_bundle)
