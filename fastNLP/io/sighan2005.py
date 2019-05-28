from fastNLP.io import DataSetLoader
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance

class sighan2005Loader(DataSetLoader):
    """
    读取sighan2005数据集（PKU, MSR, CityU and AS），读取的DataSet包含fields::
    
        words: list(str)
        labels: list(str)
    数据来源：http://sighan.cs.uchicago.edu/bakeoff2005/
    """

    def __init__(self):
        super(sighan2005Loader, self).__init__()

    def _load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            sents = f.readlines()
        examples = []
        for sent in sents:
            sent = sent.strip().split()
            if len(sent) > 0:
                words = []
                labels = []
                for word in sent:
                    if len(word) == 1:
                        words.append(word)
                        labels.append('S')
                    else:
                        words.append(word[0])
                        labels.append('B')
                        for w in word[1:len(word)-1]:
                            words.append(w)
                            labels.append('M')
                        words.append(word[-1])
                        labels.append('E')
                examples.append((words, labels))
        
        ds = DataSet()
        for words, labels in examples:
            ds.append(Instance(words=words, labels=labels))
        
        return ds

        
if __name__ == "__main__":
    loader = sighan2005Loader()
    dataset = loader._load("pku_test_gold.utf8")
    print(dataset[0])


            



    