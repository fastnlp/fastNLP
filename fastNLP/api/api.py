import warnings

import torch

warnings.filterwarnings('ignore')
import os

from fastNLP.core.dataset import DataSet

from fastNLP.api.model_zoo import load_url
from fastNLP.api.processor import ModelProcessor
from reproduction.chinese_word_segment.cws_io.cws_reader import ConllCWSReader
from reproduction.pos_tag_model.pos_reader import ZhConllPOSReader
from reproduction.Biaffine_parser.util import ConllxDataLoader, add_seg_tag
from fastNLP.core.instance import Instance
from fastNLP.api.pipeline import Pipeline
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.api.processor import IndexerProcessor


# TODO add pretrain urls
model_urls = {
    'cws': "http://123.206.98.91:8888/download/cws_crf_1_11-457fc899.pkl"
}

class API:
    def __init__(self):
        self.pipeline = None
        self._dict = None

    def predict(self, *args, **kwargs):
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

        sentence_list = []
        # 1. 检查sentence的类型
        if isinstance(content, str):
            sentence_list.append(content)
        elif isinstance(content, list):
            sentence_list = content

        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field("words", sentence_list)

        # 3. 使用pipeline
        self.pipeline(dataset)

        def decode_tags(ins):
            pred_tags = ins["tag"]
            chars = ins["words"]
            words = []
            start_idx = 0
            for idx, tag in enumerate(pred_tags):
                if tag[0] == "S":
                    words.append(chars[start_idx:idx + 1] + "/" + tag[2:])
                    start_idx = idx + 1
                elif tag[0] == "E":
                    words.append("".join(chars[start_idx:idx + 1]) + "/" + tag[2:])
                    start_idx = idx + 1
            return words

        dataset.apply(decode_tags, new_field_name="tag_output")

        output = dataset.field_arrays["tag_output"].content
        if isinstance(content, str):
            return output[0]
        elif isinstance(content, list):
            return output

    def test(self, file_path):
        test_data = ZhConllPOSReader().load(file_path)

        tag_vocab = self._dict["tag_vocab"]
        pipeline = self._dict["pipeline"]
        index_tag = IndexerProcessor(vocab=tag_vocab, field_name="tag", new_added_field_name="truth", is_input=False)
        pipeline.pipeline = [index_tag] + pipeline.pipeline

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

        return f1, pre, rec


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
        dataset.apply(lambda x: ['<BOS>']+[w.split('/')[0] for w in x['wp']], new_field_name='words')
        dataset.apply(lambda x: ['<BOS>']+[w.split('/')[1] for w in x['wp']], new_field_name='pos')

        # 3. 使用pipeline
        self.pipeline(dataset)
        dataset.apply(lambda x: [str(arc) for arc in x['arc_pred']], new_field_name='arc_pred')
        dataset.apply(lambda x: [arc + '/' + label for arc, label in
                                 zip(x['arc_pred'], x['label_pred_seq'])][1:], new_field_name='output')
        # output like: [['2/top', '0/root', '4/nn', '2/dep']]
        return dataset.field_arrays['output'].content

    def test(self, filepath):
        data = ConllxDataLoader().load(filepath)
        ds = DataSet()
        for ins1, ins2 in zip(add_seg_tag(data), data):
            ds.append(Instance(words=ins1[0], tag=ins1[1],
                               gold_words=ins2[0], gold_pos=ins2[1],
                               gold_heads=ins2[2], gold_head_tags=ins2[3]))

        pp = self.pipeline
        for p in pp:
            if p.field_name == 'word_list':
                p.field_name = 'gold_words'
            elif p.field_name == 'pos_list':
                p.field_name = 'gold_pos'
        pp(ds)
        head_cor, label_cor, total = 0, 0, 0
        for ins in ds:
            head_gold = ins['gold_heads']
            head_pred = ins['heads']
            length = len(head_gold)
            total += length
            for i in range(length):
                head_cor += 1 if head_pred[i] == head_gold[i] else 0
        uas = head_cor / total
        print('uas:{:.2f}'.format(uas))

        for p in pp:
            if p.field_name == 'gold_words':
                p.field_name = 'word_list'
            elif p.field_name == 'gold_pos':
                p.field_name = 'pos_list'

        return uas


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


if __name__ == "__main__":
    # pos_model_path = '/home/zyfeng/fastnlp/reproduction/pos_tag_model/model_pp.pkl'
    # pos = POS(pos_model_path, device='cpu')
    # s = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
    #      '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
    #      '那么这款无人机到底有多厉害？']
    # print(pos.test("/home/zyfeng/data/sample.conllx"))
    # print(pos.predict(s))

    # cws_model_path = '../../reproduction/chinese_word_segment/models/cws_crf_1_11.pkl'
    cws = CWS(device='cpu')
    s = ['本品是一个抗酸抗胆汁的胃黏膜保护剂' ,
        '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
    parser_path = '/home/yfshao/workdir/fastnlp/reproduction/Biaffine_parser/pipe.pkl'
    parser = Parser(parser_path, device='cpu')
    # print(parser.test('/Users/yh/Desktop/test_data/parser_test2.conll'))
    s = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
         '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
         '那么这款无人机到底有多厉害？']
    print(cws.test('/home/hyan/ctb3/test.conllx'))
    print(cws.predict(s))
    print(cws.predict('本品是一个抗酸抗胆汁的胃黏膜保护剂'))

    # parser = Parser(device='cpu')
    # print(parser.test('/Users/yh/Desktop/test_data/parser_test2.conll'))
    # s = ['编者按：7月12日，英国航空航天系统公司公布了该公司研制的第一款高科技隐形无人机雷电之神。',
    #      '这款飞行从外型上来看酷似电影中的太空飞行器，据英国方面介绍，可以实现洲际远程打击。',
    #      '那么这款无人机到底有多厉害？']
    # print(parser.predict(s))
    print(parser.predict(s))
