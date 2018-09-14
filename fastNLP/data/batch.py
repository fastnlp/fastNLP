from collections import defaultdict
import torch

class Batch(object):
    def __init__(self, dataset, sampler, batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

        self.idx_list = None
        self.curidx = 0

    def __iter__(self):
        self.idx_list = self.sampler(self.dataset)
        self.curidx = 0
        self.lengths = self.dataset.get_length()
        return self
    
    def __next__(self):
        if self.curidx >= len(self.idx_list):
            raise StopIteration
        else:
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            padding_length = {field_name : max(field_length[self.curidx: endidx])
                             for field_name, field_length in self.lengths.items()}
            
            batch_x, batch_y = defaultdict(list), defaultdict(list)
            for idx in range(self.curidx, endidx):
                x, y = self.dataset.to_tensor(idx, padding_length)
                for name, tensor in x.items():
                    batch_x[name].append(tensor)
                for name, tensor in y.items():
                    batch_y[name].append(tensor)

            for batch in (batch_x, batch_y):
                for name, tensor_list in batch.items():
                    print(name, "   ", tensor_list)
                    batch[name] = torch.stack(tensor_list, dim=0)
            self.curidx += endidx
            return batch_x, batch_y
            

if __name__ == "__main__":
    """simple running example
    """
    from field import TextField, LabelField
    from instance import Instance
    from dataset import DataSet

    texts = ["i am a cat",
             "this is a test of new batch",
             "haha"
            ]
    labels = [0, 1, 0]
    
    # prepare vocabulary
    vocab = {}
    for text in texts:
        for tokens in text.split():
            if tokens not in vocab:
                vocab[tokens] = len(vocab)

    # prepare input dataset    
    data = DataSet()
    for text, label in zip(texts, labels):
        x = TextField(text.split(), False)
        y = LabelField(label, is_target=True)
        ins = Instance(text=x, label=y)
        data.append(ins)
    
    # use vocabulary to index data
    data.index_field("text", vocab)

    # define naive sampler for batch class
    class SeqSampler:
        def __call__(self, dataset):
            return list(range(len(dataset)))
            
    # use bacth to iterate dataset
    batcher = Batch(data, SeqSampler(), 2)
    for epoch in range(3):
        for batch_x, batch_y in batcher:
            print(batch_x)
            print(batch_y)
            # do stuff

