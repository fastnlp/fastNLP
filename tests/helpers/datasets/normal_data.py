class NormalIterator:
    def __init__(self, num_of_data=1000):
        self._num_of_data = num_of_data
        self._data = list(range(num_of_data))
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._num_of_data:
            raise StopIteration
        _data = self._data[self._index]
        self._index += 1
        return self._data

    def __len__(self):
        return self._num_of_data