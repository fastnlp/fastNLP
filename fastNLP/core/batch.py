from collections import defaultdict

import torch


class Batch(object):
    """Batch is an iterable object which iterates over mini-batches.

    ::
        for batch_x, batch_y in Batch(data_set):

    """

    def __init__(self, dataset, batch_size, sampler, use_cuda):
        """

        :param dataset: a DataSet object
        :param batch_size: int, the size of the batch
        :param sampler: a Sampler object
        :param use_cuda: bool, whetjher to use GPU

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.use_cuda = use_cuda
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
                         batch_x also contains an item (str: list of int) about origin lengths,
                         which means ("field_name_origin_len": origin lengths).
                         E.g.
                         ::
                         {'text': tensor([[ 0,  1,  2,  3,  0,  0,  0], 4,  5,  2,  6,  7,  8,  9]]), 'text_origin_len': [4, 7]})

                batch_y: dict of (str: torch.LongTensor), which means (field name: tensor of shape [batch_size, padding_length])
                All tensors in both batch_x and batch_y will be cuda tensors if use_cuda is True.
                The names of fields are defined in preprocessor's convert_to_dataset method.

        """
        if self.curidx >= len(self.idx_list):
            raise StopIteration
        else:
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            padding_length = {field_name: max(field_length[self.curidx: endidx])
                              for field_name, field_length in self.lengths.items()}
            origin_lengths = {field_name: field_length[self.curidx: endidx]
                              for field_name, field_length in self.lengths.items()}

            batch_x, batch_y = defaultdict(list), defaultdict(list)
            for idx in range(self.curidx, endidx):
                x, y = self.dataset.to_tensor(idx, padding_length)
                for name, tensor in x.items():
                    batch_x[name].append(tensor)
                for name, tensor in y.items():
                    batch_y[name].append(tensor)

            batch_origin_length = {}
            # combine instances into a batch
            for batch in (batch_x, batch_y):
                for name, tensor_list in batch.items():
                    if self.use_cuda:
                        batch[name] = torch.stack(tensor_list, dim=0).cuda()
                    else:
                        batch[name] = torch.stack(tensor_list, dim=0)

            # add origin lengths in batch_x
            for name, tensor in batch_x.items():
                if self.use_cuda:
                    batch_origin_length[name + "_origin_len"] = torch.LongTensor(origin_lengths[name]).cuda()
                else:
                    batch_origin_length[name + "_origin_len"] = torch.LongTensor(origin_lengths[name])
            batch_x.update(batch_origin_length)

            self.curidx += endidx
            return batch_x, batch_y

