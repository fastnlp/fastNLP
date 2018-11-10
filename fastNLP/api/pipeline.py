from fastNLP.api.processor import Processor


class Pipeline:
    """
        Pipeline takes a DataSet object as input, runs multiple processors sequentially, and
        outputs a DataSet object.
    """

    def __init__(self, processors=None):
        self.pipeline = []
        if isinstance(processors, list):
            for proc in processors:
                assert isinstance(proc, Processor), "Must be a Processor, not {}.".format(type(processor))
            self.pipeline = processors

    def add_processor(self, processor):
        assert isinstance(processor, Processor), "Must be a Processor, not {}.".format(type(processor))
        self.pipeline.append(processor)

    def process(self, dataset):
        assert len(self.pipeline) != 0, "You need to add some processor first."

        for proc_name, proc in self.pipeline:
            dataset = proc(dataset)

        return dataset

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
