from fastNLP.io import MRLoader, MRPipe, R8Pipe, R8Loader, R52Pipe, R52Loader, OhsumedPipe, OhsumedLoader, _20ngPipe, _20ngLoader
mrpipe = MRPipe(True, 'raw')
data_bundle = MRLoader().load(MRLoader().download())
data_bundle = mrpipe.process(data_bundle)
print(data_bundle.get_dataset("dev"))

r8pipe = R8Pipe(True, 'raw')
data_bundle = R8Loader().load(R8Loader().download())
data_bundle = r8pipe.process(data_bundle)
print(data_bundle.get_dataset("dev"))

r52pipe = R52Pipe(True, 'raw')
data_bundle = R52Loader().load(R52Loader().download())
data_bundle = r52pipe.process(data_bundle)
print(data_bundle.get_dataset("dev"))

ohpipe = OhsumedPipe(True, 'raw')
data_bundle = OhsumedLoader().load(OhsumedLoader().download())
data_bundle = ohpipe.process(data_bundle)
print(data_bundle.get_dataset("dev"))

ngpipe = _20ngPipe(True, 'raw')
data_bundle = _20ngLoader().load(_20ngLoader().download())
data_bundle = ngpipe.process(data_bundle)
print(data_bundle.get_dataset("dev").get_field('target'))