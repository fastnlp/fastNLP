import torch


class Batch(object):
    """Batch is an iterable object which iterates over mini-batches.

    ::
        for batch_x, batch_y in Batch(data_set):

    """

    def __init__(self, dataset, batch_size, sampler, use_cuda=False):
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
            batch_x, batch_y = {}, {}

            indices = self.idx_list[self.curidx:endidx]

            for field_name, field in self.dataset.get_fields().items():
                if field.need_tensor:
                    batch = torch.from_numpy(field.get(indices))
                    if self.use_cuda:
                        batch = batch.cuda()
                    if field.is_target:
                        batch_y[field_name] = batch
                    else:
                        batch_x[field_name] = batch

            self.curidx = endidx

            return batch_x, batch_y
