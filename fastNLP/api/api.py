import warnings

import torch

warnings.filterwarnings('ignore')
import os

from fastNLP.core.dataset import DataSet

from fastNLP.api.utils import load_url
from fastNLP.api.processor import ModelProcessor
from fastNLP.io.dataset_loader import ConllCWSReader, ConllxDataLoader
from fastNLP.core.instance import Instance
from fastNLP.api.pipeline import Pipeline
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.api.processor import IndexerProcessor

# TODO add pretrain urls
model_urls = {
    "cws": "http://123.206.98.91:8888/download/cws_lstm_ctb9_1_20-09908656.pkl",
    "pos": "http://123.206.98.91:8888/download/pos_tag_model_20190119-43f8b435.pkl",
    "parser": "http://123.206.98.91:8888/download/parser_20190204-c72ca5c0.pkl"
}


class API:
    def __init__(self):
        self.pipeline = None
        self._dict = None

    def predict(self, *args, **kwargs):
        """Do prediction for the given input.
        """
        raise NotImplementedError

    def test(self, file_path):
        """Test performance over the given data set.

        :param str file_path:
        :return: a dictionary of metric values
        """
        raise NotImplementedError

    def load(self, path, device):
        if os.path.exists(os.path.expanduser(path)):
            _dict = torch.load(path, map_location='cpu')
        else:
            _dict = load_url(path, map_location='cpu')
        self._dict = _dict
        self.pipeline = _dict['pipeline']
        for processor in self.pipeline.pipeline:
            if isinstance(processor, ModelProcessor):
                processor.set_model_device(device)


class POS(API):
    """FastNLP API for Part-Of-Speech tagging.

    :param str model_path: the path to the model.
    :param str device: device name such as "cpu" or "cuda:0". Use the same notation as PyTorch.

    """

    def __init__(self, model_path=None, device='cpu'):
        super(POS, self).__init__()
        if model_path is None:
            model_path = model_urls['pos']

        self.load(model_path, device)

    def predict(self, content):
        """

        :param content: list of list of str. Each string is a token(word).
        :return answer: list of list of str. Each string is a tag.
        """
        if not hasattr(self, "pipeline"):
            raise ValueError("You have to load model first.")

        sentence_list = content
        # 1. 检查sentence的类型
        for sentence in sentence_list:
            if not all((type(obj) == str for obj in sentence)):
                raise ValueError("Input must be list of list of string.")

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field("words", sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        def merge_tag(words_list, tags_list):
            rtn = []
            for words, tags in zip(words_list, tags_list):
                rtn.append([w + "/" + t for w, t in zip(words, tags)])
            return rtn

        output = dataset.field_arrays["tag"].content
        if isinstance(content, str):
            return output[0]
        elif isinstance(content, list):
            return merge_tag(content, output)

    def test(self, file_path):
        test_data = ConllxDataLoader().load(file_path)

        save_dict = self._dict
        tag_vocab = save_dict["tag_vocab"]
        pipeline = save_dict["pipeline"]
        index_tag = IndexerProcessor(vocab=tag_vocab, field_name="tag", new_added_field_name="truth", is_input=False)
        pipeline.pipeline = [index_tag] + pipeline.pipeline

        test_data.rename_field("pos_tags", "tag")
        pipeline(test_data)
        test_data.set_target("truth")
        prediction = test_data.field_arrays["predict"].content
        truth = test_data.field_arrays["truth"].content
        seq_len = test_data.field_arrays["word_seq_origin_len"].content

        # padding by hand
        max_length = max([len(seq) for seq in prediction])
        for idx in range(len(prediction)):
            prediction[idx] = list(prediction[idx]) + ([0] * (max_length - len(prediction[idx])))
            truth[idx] = list(truth[idx]) + ([0] * (max_length - len(truth[idx])))
        evaluator = SpanFPreRecMetric(tag_vocab=tag_vocab, pred="predict", target="truth",
                                      seq_lens="word_seq_origin_len")
        evaluator({"predict": torch.Tensor(prediction), "word_seq_origin_len": torch.Tensor(seq_len)},
                  {"truth": torch.Tensor(truth)})
        test_result = evaluator.get_metric()
        f1 = round(test_result['f'] * 100, 2)
        pre = round(test_result['pre'] * 100, 2)
        rec = round(test_result['rec'] * 100, 2)

        return {"F1": f1, "precision": pre, "recall": rec}


class CWS(API):
    def __init__(self, model_path=None, device='cpu'):
        """
        中文分词高级接口。

        :param model_path: 当model_path为None，使用默认位置的model。如果默认位置不存在，则自动下载模型
        :param device: str，可以为'cpu', 'cuda'或'cuda:0'等。会将模型load到相应device进行推断。
        """
        super(CWS, self).__init__()
        if model_path is None:
            model_path = model_urls['cws']

        self.load(model_path, device)

    def predict(self, content):
        """
        分词接口。

        :param content: str或List[str], 例如: "中文分词很重要！"， 返回的结果是"中文 分词 很 重要 !"。 如果传入的为List[str]，比如
            [ "中文分词很重要！", ...], 返回的结果["中文 分词 很 重要 !", ...]。
        :return: str或List[str], 根据输入的的类型决定。
        """
        if not hasattr(self, 'pipeline'):
            raise ValueError("You have to load model first.")

        sentence_list = []
        # 1. 检查sentence的类型
        if isinstance(content, str):
            sentence_list.append(content)
        elif isinstance(content, list):
            sentence_list = content

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field('raw_sentence', sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        output = dataset.get_field('output').content
        if isinstance(content, str):
            return output[0]
        elif isinstance(content, list):
            return output

    def test(self, filepath):
        """
        传入一个分词文件路径，返回该数据集上分词f1, precision, recall。
        分词文件应该为:
            1	编者按	编者按	NN	O	11	nmod:topic
            2	：	：	PU	O	11	punct
            3	7月	7月	NT	DATE	4	compound:nn
            4	12日	12日	NT	DATE	11	nmod:tmod
            5	，	，	PU	O	11	punct

            1	这	这	DT	O	3	det
            2	款	款	M	O	1	mark:clf
            3	飞行	飞行	NN	O	8	nsubj
            4	从	从	P	O	5	case
            5	外型	外型	NN	O	8	nmod:prep
        以空行分割两个句子，有内容的每行有7列。

        :param filepath: str, 文件路径路径。
        :return: float, float, float. 分别f1, precision, recall.
        """
        tag_proc = self._dict['tag_proc']
        cws_model = self.pipeline.pipeline[-2].model
        pipeline = self.pipeline.pipeline[:-2]

        pipeline.insert(1, tag_proc)
        pp = Pipeline(pipeline)

        reader = ConllCWSReader()

        # te_filename = '/home/hyan/ctb3/test.conllx'
        te_dataset = reader.load(filepath)
        pp(te_dataset)

        from fastNLP.core.tester import Tester
        from fastNLP.core.metrics import BMESF1PreRecMetric

        tester = Tester(data=te_dataset, model=cws_model, metrics=BMESF1PreRecMetric(target='target'), batch_size=64,
                        verbose=0)
        eval_res = tester.test()

        f1 = eval_res['BMESF1PreRecMetric']['f']
        pre = eval_res['BMESF1PreRecMetric']['pre']
        rec = eval_res['BMESF1PreRecMetric']['rec']
        # print("f1:{:.2f}, pre:{:.2f}, rec:{:.2f}".format(f1, pre, rec))

        return {"F1": f1, "precision": pre, "recall": rec}


class Parser(API):
    def __init__(self, model_path=None, device='cpu'):
        super(Parser, self).__init__()
        if model_path is None:
            model_path = model_urls['parser']

        self.pos_tagger = POS(device=device)
        self.load(model_path, device)

    def predict(self, content):
        if not hasattr(self, 'pipeline'):
            raise ValueError("You have to load model first.")

        # 1. 利用POS得到分词和pos tagging结果
        pos_out = self.pos_tagger.predict(content)
        # pos_out = ['这里/NN 是/VB 分词/NN 结果/NN'.split()]

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field('wp', pos_out)
        dataset.apply(lambda x: ['<BOS>'] + [w.split('/')[0] for w in x['wp']], new_field_name='words')
        dataset.apply(lambda x: ['<BOS>'] + [w.split('/')[1] for w in x['wp']], new_field_name='pos')
        dataset.rename_field("words", "raw_words")

        # 3. 使用pipeline
        self.pipeline(dataset)
        dataset.apply(lambda x: [str(arc) for arc in x['arc_pred']], new_field_name='arc_pred')
        dataset.apply(lambda x: [arc + '/' + label for arc, label in
                                 zip(x['arc_pred'], x['label_pred_seq'])][1:], new_field_name='output')
        # output like: [['2/top', '0/root', '4/nn', '2/dep']]
        return dataset.field_arrays['output'].content

    def load_test_file(self, path):
        def get_one(sample):
            sample = list(map(list, zip(*sample)))
            if len(sample) == 0:
                return None
            for w in sample[7]:
                if w == '_':
                    print('Error Sample {}'.format(sample))
                    return None
            # return word_seq, pos_seq, head_seq, head_tag_seq
            return sample[1], sample[3], list(map(int, sample[6])), sample[7]

        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        data = [get_one(sample) for sample in datalist]
        data_list = list(filter(lambda x: x is not None, data))
        return data_list

    def test(self, filepath):
        data = self.load_test_file(filepath)

        def convert(data):
            BOS = '<BOS>'
            dataset = DataSet()
            for sample in data:
                word_seq = [BOS] + sample[0]
                pos_seq = [BOS] + sample[1]
                heads = [0] + sample[2]
                head_tags = [BOS] + sample[3]
                dataset.append(Instance(raw_words=word_seq,
                                        pos=pos_seq,
                                        gold_heads=heads,
                                        arc_true=heads,
                                        tags=head_tags))
            return dataset

        ds = convert(data)
        pp = self.pipeline
        for p in pp:
            if p.field_name == 'word_list':
                p.field_name = 'gold_words'
            elif p.field_name == 'pos_list':
                p.field_name = 'gold_pos'
        # ds.rename_field("words", "raw_words")
        # ds.rename_field("tag", "pos")
        pp(ds)
        head_cor, label_cor, total = 0, 0, 0
        for ins in ds:
            head_gold = ins['gold_heads']
            head_pred = ins['arc_pred']
            length = len(head_gold)
            total += length
            for i in range(length):
                head_cor += 1 if head_pred[i] == head_gold[i] else 0
        uas = head_cor / total
        # print('uas:{:.2f}'.format(uas))

        for p in pp:
            if p.field_name == 'gold_words':
                p.field_name = 'word_list'
            elif p.field_name == 'gold_pos':
                p.field_name = 'pos_list'

        return {"USA": round(uas, 5)}


class Analyzer:
    def __init__(self, device='cpu'):

        self.cws = CWS(device=device)
        self.pos = POS(device=device)
        self.parser = Parser(device=device)

    def predict(self, content, seg=False, pos=False, parser=False):
        if seg is False and pos is False and parser is False:
            seg = True
        output_dict = {}
        if seg:
            seg_output = self.cws.predict(content)
            output_dict['seg'] = seg_output
        if pos:
            pos_output = self.pos.predict(content)
            output_dict['pos'] = pos_output
        if parser:
            parser_output = self.parser.predict(content)
            output_dict['parser'] = parser_output

        return output_dict

    def test(self, filepath):
        output_dict = {}
        if self.cws:
            seg_output = self.cws.test(filepath)
            output_dict['seg'] = seg_output
        if self.pos:
            pos_output = self.pos.test(filepath)
            output_dict['pos'] = pos_output
        if self.parser:
            parser_output = self.parser.test(filepath)
            output_dict['parser'] = parser_output

        return output_dict
