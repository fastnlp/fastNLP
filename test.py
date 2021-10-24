from fastNLP.io.pipe.construct_graph import R52PmiGraphPipe, R8PmiGraphPipe, MRPmiGraphPipe, NG20PmiGraphPipe, OhsumedPmiGraphPipe
from fastNLP.io.loader.classification import R52Loader, R8Loader, MRLoader, NG20Loader, OhsumedLoader

g, ind = R52PmiGraphPipe().build_graph_from_file(R52Loader().download())
print(g, ind)

g, ind = R8PmiGraphPipe().build_graph_from_file(R8Loader().download())
print(g, ind)

g, ind = MRPmiGraphPipe().build_graph_from_file(MRLoader().download())
print(g, ind)

g, ind = OhsumedPmiGraphPipe().build_graph_from_file(OhsumedLoader().download())
print(g, ind)

g, ind = NG20PmiGraphPipe().build_graph_from_file(NG20Loader().download())
print(g, ind)


# data_bundle = R52Loader().load(R52Loader().download())
# print(data_bundle.get_dataset("train"))
# g, ind = R52PmiGraphPipe().build_graph(data_bundle)
# print(ind)
#
# data_bundle = R8Loader().load(R8Loader().download())
# g, ind = R8PmiGraphPipe().build_graph(data_bundle)
# print(ind)
#
# data_bundle = MRLoader().load(MRLoader().download())
# g, ind = MRPmiGraphPipe().build_graph(data_bundle)
# print(ind)
#
# data_bundle = NG20Loader().load(NG20Loader().download())
# g, ind = NG20PmiGraphPipe().build_graph(data_bundle)
# print(ind)
#
# data_bundle = OhsumedLoader().load(OhsumedLoader().download())
# g, ind = OhsumedPmiGraphPipe().build_graph(data_bundle)
# print(ind)