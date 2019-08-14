from .. import DataBundle


class Pipe:
    def process(self, data_bundle: DataBundle) -> DataBundle:
        raise NotImplementedError

    def process_from_file(self, paths) -> DataBundle:
        raise NotImplementedError
