import _pickle as pickle
import argparse
import collections
import logging
import math
import os
import pickle
import random
import sys
import time
from sys import maxsize

import fastNLP
import fastNLP.embeddings
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from fastNLP import BucketSampler, DataSetIter, SequentialSampler, logger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import models
import optm
import utils

NONE_TAG = "<NONE>"
START_TAG = "<sos>"
END_TAG = "<eos>"

DEFAULT_WORD_EMBEDDING_SIZE = 100
DEBUG_SCALE = 200

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help="processed data dir")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--bigram-embeddings", dest="bigram_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--crf", dest="crf", action="store_true", help="crf")
# parser.add_argument("--devi", default="0", dest="devi", help="gpu")
parser.add_argument("--step", default=0, dest="step", type=int,help="step")
parser.add_argument("--num-epochs", default=100, dest="num_epochs", type=int,
                    help="Number of full passes through training set")
parser.add_argument("--batch-size", default=128, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--d_model", default=256, dest="d_model", type=int, help="d_model")
parser.add_argument("--d_ff", default=1024, dest="d_ff", type=int, help="d_ff")
parser.add_argument("--N", default=6, dest="N", type=int, help="N")
parser.add_argument("--h", default=4, dest="h", type=int, help="h")
parser.add_argument("--factor", default=2, dest="factor", type=float, help="Initial learning rate")
parser.add_argument("--dropout", default=0.2, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / serialized models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't serialize model")
parser.add_argument("--always-model", dest="always_model", action="store_true",
                    help="Always serialize model after every epoch")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set, would save some time")
parser.add_argument("--freeze", dest="freeze", action="store_true", help="freeze pretrained embedding")
parser.add_argument("--only-task", dest="only_task", action="store_true", help="only train task embedding")
parser.add_argument("--subset", dest="subset", help="Only train and test on a subset of the whole dataset")
parser.add_argument("--seclude", dest="seclude", help="train and test except a subset")
parser.add_argument("--instances", default=None, dest="instances", type=int,help="num of instances of subset")

parser.add_argument("--seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--test", dest="test", action="store_true", help="Test mode")
parser.add_argument('--local_rank', type=int, default=None)
parser.add_argument('--init_method', type=str, default='env://')
# fmt: on

options, _ = parser.parse_known_args()
print("unknown args", _)
task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
utils.make_sure_path_exists(root_dir)

if options.local_rank is not None:
    torch.cuda.set_device(options.local_rank)
    dist.init_process_group("nccl", init_method=options.init_method)


def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode="w")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    if options.local_rank is None or options.local_rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    return logger


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
# logger = init_logger()
logger.add_file("{}/info.log".format(root_dir), "INFO")
logger.setLevel(logging.INFO if dist.get_rank() == 0 else logging.WARNING)

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(" ".join(sys.argv))
logger.info("")
logger.info(options)

if options.debug:
    logger.info("DEBUG MODE")
    options.num_epochs = 2
    options.batch_size = 20

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
torch.cuda.manual_seed_all(options.python_seed)
logger.info("Python random seed: {}".format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset + "/total_dataset.pkl", "rb"))
train_set = dataset["train_set"]
test_set = dataset["test_set"]
uni_vocab = dataset["uni_vocab"]
bi_vocab = dataset["bi_vocab"]
task_vocab = dataset["task_vocab"]
tag_vocab = dataset["tag_vocab"]
for v in (bi_vocab, uni_vocab, tag_vocab, task_vocab):
    if hasattr(v, "_word2idx"):
        v.word2idx = v._word2idx
for ds in (train_set, test_set):
    ds.rename_field("ori_words", "words")

logger.info("{} {}".format(bi_vocab.to_word(0), tag_vocab.word2idx))
logger.info(task_vocab.word2idx)
if options.skip_dev:
    dev_set = test_set
else:
    train_set, dev_set = train_set.split(0.1)

logger.info("{} {} {}".format(len(train_set), len(dev_set), len(test_set)))

if options.debug:
    train_set = train_set[0:DEBUG_SCALE]
    dev_set = dev_set[0:DEBUG_SCALE]
    test_set = test_set[0:DEBUG_SCALE]

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

# ===============================
if dist.get_rank() != 0:
    dist.barrier()

if options.word_embeddings is None:
    init_embedding = None
else:
    # logger.info("Load: {}".format(options.word_embeddings))
    # init_embedding = utils.embedding_load_with_cache(options.word_embeddings, options.cache_dir, uni_vocab, normalize=False)
    init_embedding = fastNLP.embeddings.StaticEmbedding(
        uni_vocab, options.word_embeddings, word_drop=0.01
    )

bigram_embedding = None
if options.bigram_embeddings:
    # logger.info("Load: {}".format(options.bigram_embeddings))
    # bigram_embedding = utils.embedding_load_with_cache(options.bigram_embeddings, options.cache_dir, bi_vocab, normalize=False)
    bigram_embedding = fastNLP.embeddings.StaticEmbedding(
        bi_vocab, options.bigram_embeddings
    )

if dist.get_rank() == 0:
    dist.barrier()
# ===============================

# select subset training
if options.seclude is not None:
    setname = "<{}>".format(options.seclude)
    logger.info("seclude {}".format(setname))
    train_set.drop(lambda x: x["words"][0] == setname, inplace=True)
    test_set.drop(lambda x: x["words"][0] == setname, inplace=True)
    dev_set.drop(lambda x: x["words"][0] == setname, inplace=True)

if options.subset is not None:
    setname = "<{}>".format(options.subset)
    logger.info("select {}".format(setname))
    train_set.drop(lambda x: x["words"][0] != setname, inplace=True)
    test_set.drop(lambda x: x["words"][0] != setname, inplace=True)
    dev_set.drop(lambda x: x["words"][0] != setname, inplace=True)

# build model and optimizer
i2t = None
if options.crf:
    # i2t=utils.to_id_list(tag_vocab.word2idx)
    i2t = {}
    for x, y in tag_vocab.word2idx.items():
        i2t[y] = x
    logger.info(i2t)

freeze = True if options.freeze else False
model = models.make_CWS(
    d_model=options.d_model,
    N=options.N,
    h=options.h,
    d_ff=options.d_ff,
    dropout=options.dropout,
    word_embedding=init_embedding,
    bigram_embedding=bigram_embedding,
    tag_size=len(tag_vocab),
    task_size=len(task_vocab),
    crf=i2t,
    freeze=freeze,
)

device = "cpu"

if torch.cuda.device_count() > 0:
    if options.local_rank is not None:
        device = "cuda:{}".format(options.local_rank)
        # model=nn.DataParallel(model)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[options.local_rank], output_device=options.local_rank
        )
    else:
        device = "cuda:0"
        model.to(device)


if options.only_task and options.old_model is not None:
    logger.info("fix para except task embedding")
    for name, para in model.named_parameters():
        if name.find("task_embed") == -1:
            para.requires_grad = False
        else:
            para.requires_grad = True
            logger.info(name)

optimizer = optm.NoamOpt(
    options.d_model,
    options.factor,
    4000,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
)

optimizer._step = options.step

best_model_file_name = "{}/model.bin".format(root_dir)

if options.local_rank is None:
    train_sampler = BucketSampler(
        batch_size=options.batch_size, seq_len_field_name="seq_len"
    )
else:
    train_sampler = DistributedSampler(
        train_set, dist.get_world_size(), dist.get_rank()
    )
dev_sampler = SequentialSampler()

i2t = utils.to_id_list(tag_vocab.word2idx)
i2task = utils.to_id_list(task_vocab.word2idx)
dev_set.set_input("words")
test_set.set_input("words")
test_batch = DataSetIter(test_set, options.batch_size, num_workers=2)

word_dic = pickle.load(open(options.dataset + "/oovdict.pkl", "rb"))


def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def tester(model, test_batch, write_out=False):
    res = []
    prf = utils.CWSEvaluator(i2t)
    prf_dataset = {}
    oov_dataset = {}

    logger.info("start evaluation")
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for batch_x, batch_y in test_batch:
            batch_to_device(batch_x, device)
            # batch_to_device(batch_y, device)
            if bigram_embedding is not None:
                out = model(
                    batch_x["task"],
                    batch_x["uni"],
                    batch_x["seq_len"],
                    batch_x["bi1"],
                    batch_x["bi2"],
                )
            else:
                out = model(batch_x["task"], batch_x["uni"], batch_x["seq_len"])
            out = out["pred"]
            # print(out)
            num = out.size(0)
            out = out.detach().cpu().numpy()
            for i in range(num):
                length = int(batch_x["seq_len"][i])

                out_tags = out[i, 1:length].tolist()
                sentence = batch_x["words"][i]
                gold_tags = batch_y["tags"][i][1:length].numpy().tolist()
                dataset_name = sentence[0]
                sentence = sentence[1:]
                # print(out_tags,gold_tags)
                assert utils.is_dataset_tag(dataset_name), dataset_name
                assert len(gold_tags) == len(out_tags) and len(gold_tags) == len(
                    sentence
                )

                if dataset_name not in prf_dataset:
                    prf_dataset[dataset_name] = utils.CWSEvaluator(i2t)
                    oov_dataset[dataset_name] = utils.CWS_OOV(
                        word_dic[dataset_name[1:-1]]
                    )

                prf_dataset[dataset_name].add_instance(gold_tags, out_tags)
                prf.add_instance(gold_tags, out_tags)

                if write_out:
                    gold_strings = utils.to_tag_strings(i2t, gold_tags)
                    obs_strings = utils.to_tag_strings(i2t, out_tags)

                    word_list = utils.bmes_to_words(sentence, obs_strings)
                    oov_dataset[dataset_name].update(
                        utils.bmes_to_words(sentence, gold_strings), word_list
                    )

                    raw_string = " ".join(word_list)
                    res.append(dataset_name + " " + raw_string + " " + dataset_name)

        Ap = 0.0
        Ar = 0.0
        Af = 0.0
        Aoov = 0.0
        tot = 0
        nw = 0.0
        for dataset_name, performance in sorted(prf_dataset.items()):
            p = performance.result()
            if write_out:
                nw = oov_dataset[dataset_name].oov()
                # nw = 0
                logger.info(
                    "{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\t{:04.2f}".format(
                        dataset_name, p[0], p[1], p[2], nw
                    )
                )
            else:
                logger.info(
                    "{}\t{:04.2f}\t{:04.2f}\t{:04.2f}".format(
                        dataset_name, p[0], p[1], p[2]
                    )
                )
            Ap += p[0]
            Ar += p[1]
            Af += p[2]
            Aoov += nw
            tot += 1

        prf = prf.result()
        logger.info(
            "{}\t{:04.2f}\t{:04.2f}\t{:04.2f}".format("TOT", prf[0], prf[1], prf[2])
        )
        if not write_out:
            logger.info(
                "{}\t{:04.2f}\t{:04.2f}\t{:04.2f}".format(
                    "AVG", Ap / tot, Ar / tot, Af / tot
                )
            )
        else:
            logger.info(
                "{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\t{:04.2f}".format(
                    "AVG", Ap / tot, Ar / tot, Af / tot, Aoov / tot
                )
            )
    return prf[-1], res


# start training
if not options.test:
    if options.old_model:
        # incremental training
        logger.info("Incremental training from old model: {}".format(options.old_model))
        model.load_state_dict(torch.load(options.old_model, map_location="cuda:0"))

    logger.info("Number training instances: {}".format(len(train_set)))
    logger.info("Number dev instances: {}".format(len(dev_set)))

    train_batch = DataSetIter(
        batch_size=options.batch_size,
        dataset=train_set,
        sampler=train_sampler,
        num_workers=4,
    )
    dev_batch = DataSetIter(
        batch_size=options.batch_size,
        dataset=dev_set,
        sampler=dev_sampler,
        num_workers=4,
    )

    best_f1 = 0.0
    for epoch in range(int(options.num_epochs)):
        logger.info("Epoch {} out of {}".format(epoch + 1, options.num_epochs))
        train_loss = 0.0
        model.train()
        tot = 0
        t1 = time.time()
        for batch_x, batch_y in train_batch:
            model.zero_grad()
            if bigram_embedding is not None:
                out = model(
                    batch_x["task"],
                    batch_x["uni"],
                    batch_x["seq_len"],
                    batch_x["bi1"],
                    batch_x["bi2"],
                    batch_y["tags"],
                )
            else:
                out = model(
                    batch_x["task"], batch_x["uni"], batch_x["seq_len"], batch_y["tags"]
                )
            loss = out["loss"]
            train_loss += loss.item()
            tot += 1
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

        t2 = time.time()
        train_loss = train_loss / tot
        logger.info(
            "time: {} loss: {} step: {}".format(t2 - t1, train_loss, optimizer._step)
        )
        # Evaluate dev data
        if options.skip_dev and dist.get_rank() == 0:
            logger.info("Saving model to {}".format(best_model_file_name))
            torch.save(model.module.state_dict(), best_model_file_name)
            continue

        model.eval()
        if dist.get_rank() == 0:
            f1, _ = tester(model.module, dev_batch)
            if f1 > best_f1:
                best_f1 = f1
                logger.info("- new best score!")
                if not options.no_model:
                    logger.info("Saving model to {}".format(best_model_file_name))
                    torch.save(model.module.state_dict(), best_model_file_name)

            elif options.always_model:
                logger.info("Saving model to {}".format(best_model_file_name))
                torch.save(model.module.state_dict(), best_model_file_name)
        dist.barrier()

# Evaluate test data (once)
logger.info("\nNumber test instances: {}".format(len(test_set)))


if not options.skip_dev:
    if options.test:
        model.module.load_state_dict(
            torch.load(options.old_model, map_location="cuda:0")
        )
    else:
        model.module.load_state_dict(
            torch.load(best_model_file_name, map_location="cuda:0")
        )

if dist.get_rank() == 0:
    for name, para in model.named_parameters():
        if name.find("task_embed") != -1:
            tm = para.detach().cpu().numpy()
            logger.info(tm.shape)
            np.save("{}/task.npy".format(root_dir), tm)
            break

_, res = tester(model.module, test_batch, True)

if dist.get_rank() == 0:
    with open("{}/testout.txt".format(root_dir), "w", encoding="utf-8") as raw_writer:
        for sent in res:
            raw_writer.write(sent)
            raw_writer.write("\n")

