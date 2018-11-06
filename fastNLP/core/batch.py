from collections import defaultdict

import torch


class Batch(object):
    """Batch is an iterable object which iterates over mini-batches.

    ::
        for batch_x, batch_y in Batch(data_set):

    """

    def __init__(self, dataset, batch_size, sampler, use_cuda, sort_in_batch=False, sort_key=None):
        """

        :param dataset: a DataSet object
        :param batch_size: int, the size of the batch
        :param sampler: a Sampler object
        :param use_cuda: bool, whether to use GPU

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.use_cuda = use_cuda
        self.sort_in_batch = sort_in_batch
        self.sort_key = sort_key if sort_key is not None else 'word_seq'
        self.idx_list = None
        self.curidx = 0

    def __iter__(self):
        self.idx_list = self.sampler(self.dataset)
        self.curidx = 0
        self.lengths = self.dataset.get_length()
        return self

    def __next__(self):
        """

        :return batch_x: dict of (str: torch.LongTensor), which means (field name: tensor of shape [batch_size, padding_length])
                         E.g.
                         ::
                         {'text': tensor([[ 0,  1,  2,  3,  0,  0,  0], 4,  5,  2,  6,  7,  8,  9]]), 'text_origin_len': [4, 7]})

                batch_y: dict of (str: torch.LongTensor), which means (field name: tensor of shape [batch_size, padding_length])
                All tensors in both batch_x and batch_y will be cuda tensors if use_cuda is True.

        """
        if self.curidx >= len(self.idx_list):
            raise StopIteration
        else:
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            batch_idxes = self.idx_list[self.curidx: endidx]
            padding_length = {field_name: max([field_length[idx] for idx in batch_idxes])
                              for field_name, field_length in self.lengths.items()}
            batch_x, batch_y = defaultdict(list), defaultdict(list)

            # transform index to tensor and do padding for sequences
            batch = []
            for idx in batch_idxes:
                x, y = self.dataset.to_tensor(idx, padding_length)
                batch.append((self.lengths[self.sort_key][idx] if self.sort_in_batch else None, x, y))

            if self.sort_in_batch:
                batch = sorted(batch, key=lambda x: x[0], reverse=True)

            for _, x, y in batch:
                for name, tensor in x.items():
                    batch_x[name].append(tensor)
                for name, tensor in y.items():
                    batch_y[name].append(tensor)

            # combine instances to form a batch
            for batch in (batch_x, batch_y):
                for name, tensor_list in batch.items():
                    if self.use_cuda:
                        batch[name] = torch.stack(tensor_list, dim=0).cuda()
                    else:
                        batch[name] = torch.stack(tensor_list, dim=0)

            self.curidx = endidx
            return batch_x, batch_y

