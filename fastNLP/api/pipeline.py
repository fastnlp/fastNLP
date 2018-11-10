from fastNLP.api.processor import Processor



class Pipeline:
    def __init__(self):
        self.pipeline = []

    def add_processor(self, processor):
        assert isinstance(processor, Processor), "Must be a Processor, not {}.".format(type(processor))
        self.pipeline.append(processor)

    def process(self, dataset):
        assert len(self.pipeline)!=0, "You need to add some processor first."

        for proc in self.pipeline:
            dataset = proc(dataset)

        return dataset

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def __getitem__(self, item):
        return self.pipeline[item]
