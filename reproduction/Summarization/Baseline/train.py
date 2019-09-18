#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train Model1: baseline model"""
import os
import sys
import json
import shutil
import argparse
import datetime

import torch
import torch.nn

os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'
sys.path.append('/remote-home/dqwang/FastNLP/fastNLP_brxx/')


from fastNLP.core._logger import logger
# from fastNLP.core._logger import _init_logger
from fastNLP.core.const import Const
from fastNLP.core.trainer import Trainer, Tester
from fastNLP.io.pipe.summarization import ExtCNNDMPipe
from fastNLP.io.model_io import ModelLoader, ModelSaver
from fastNLP.io.embed_loader import EmbedLoader

# from tools.logger import *
# from model.TransformerModel import TransformerModel
from model.TForiginal import TransformerModel
from model.LSTMModel import SummarizationModel
from model.Metric import LossMetric, LabelFMetric, FastRougeMetric, PyRougeMetric
from model.Loss import MyCrossEntropyLoss
from tools.Callback import TrainCallback




def setup_training(model, train_loader, valid_loader, hps):
    """Does setup before starting training (run_training)"""

    train_dir = os.path.join(hps.save_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    if hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        loader = ModelLoader()
        loader.load_pytorch(model, bestmodel_file)
    else:
        logger.info("[INFO] Create new model for training...")

    run_training(model, train_loader, valid_loader, hps) # this is an infinite loop until interrupted

def run_training(model, train_loader, valid_loader, hps):
    logger.info("[INFO] Starting run_training")

    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    eval_dir = os.path.join(hps.save_root, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    criterion = MyCrossEntropyLoss(pred = "p_sent", target=Const.TARGET, mask=Const.INPUT_LEN, reduce='none')

    trainer = Trainer(model=model, train_data=train_loader, optimizer=optimizer, loss=criterion,
                      n_epochs=hps.n_epochs, print_every=100, dev_data=valid_loader, metrics=[LossMetric(pred = "p_sent", target=Const.TARGET, mask=Const.INPUT_LEN, reduce='none'), LabelFMetric(pred="prediction"), FastRougeMetric(hps, pred="prediction")],
                      metric_key="loss", validate_every=-1, save_path=eval_dir,
                      callbacks=[TrainCallback(hps, patience=5)], use_tqdm=False)

    train_info = trainer.train(load_best_model=True)
    logger.info('   | end of Train | time: {:5.2f}s | '.format(train_info["seconds"]))
    logger.info('[INFO] best eval model in epoch %d and iter %d', train_info["best_epoch"], train_info["best_step"])
    logger.info(train_info["best_eval"])

    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel.pkl')  # this is where checkpoints of best models are saved
    saver = ModelSaver(bestmodel_save_path)
    saver.save_pytorch(model)
    logger.info('[INFO] Saving eval best model to %s', bestmodel_save_path)


def run_test(model, loader, hps):
    test_dir = os.path.join(hps.save_root, "test") # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(eval_dir) :
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    if hps.test_model == "evalbestmodel":
        bestmodel_load_path = os.path.join(eval_dir, 'bestmodel.pkl') # this is where checkpoints of best models are saved
    elif hps.test_model == "earlystop":
        train_dir = os.path.join(hps.save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop.pkl')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/earlystop")
    logger.info("[INFO] Restoring %s for testing...The path is %s", hps.test_model, bestmodel_load_path)

    modelloader = ModelLoader()
    modelloader.load_pytorch(model, bestmodel_load_path)

    if hps.use_pyrouge:
        logger.info("[INFO] Use PyRougeMetric for testing")
        tester = Tester(data=loader, model=model,
                        metrics=[LabelFMetric(pred="prediction"), PyRougeMetric(hps, pred="prediction")],
                        batch_size=hps.batch_size)
    else:
        logger.info("[INFO] Use FastRougeMetric for testing")
        tester = Tester(data=loader, model=model,
                        metrics=[LabelFMetric(pred="prediction"), FastRougeMetric(hps, pred="prediction")],
                        batch_size=hps.batch_size)
    test_info = tester.test()
    logger.info(test_info)

def main():
    parser = argparse.ArgumentParser(description='Summarization Model')

    # Where to find data
    parser.add_argument('--data_path', type=str, default='/remote-home/dqwang/Datasets/CNNDM/train.label.jsonl', help='Path expression to pickle datafiles.')
    parser.add_argument('--valid_path', type=str, default='/remote-home/dqwang/Datasets/CNNDM/val.label.jsonl', help='Path expression to pickle valid datafiles.')
    parser.add_argument('--vocab_path', type=str, default='/remote-home/dqwang/Datasets/CNNDM/vocab', help='Path expression to text vocabulary file.')

    # Important settings
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='must be one of train/test')
    parser.add_argument('--embedding', type=str, default='glove', choices=['word2vec', 'glove', 'elmo', 'bert'], help='must be one of word2vec/glove/elmo/bert')
    parser.add_argument('--sentence_encoder', type=str, default='transformer', choices=['bilstm', 'deeplstm', 'transformer'], help='must be one of LSTM/Transformer')
    parser.add_argument('--sentence_decoder', type=str, default='SeqLab', choices=['PN', 'SeqLab'], help='must be one of PN/SeqLab')
    parser.add_argument('--restore_model', type=str , default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--vocab_size', type=int, default=100000, help='Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 128]')

    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 200]')
    parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
    parser.add_argument('--min_kernel_size', type=int, default=1, help='kernel min length for CNN [default:1]')
    parser.add_argument('--max_kernel_size', type=int, default=7, help='kernel max length for CNN [default:7]')
    parser.add_argument('--output_channel', type=int, default=50, help='output channel: repeated times for one kernel')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
    parser.add_argument('--sent_max_len', type=int, default=100, help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50, help='max length of documents (max timesteps of documents)')
    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')

    # Training
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=10, help='for gradient clipping max gradient normalization')

    # test
    parser.add_argument('-m', type=int, default=3, help='decode summary length')
    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [evalbestmodel/evalbestFmodel/trainbestmodel/trainbestFmodel/earlystop]')
    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = args.data_path
    VALID_FILE = args.valid_path
    VOCAL_FILE = args.vocab_path
    LOG_PATH = args.log_root

    # # train_log setting
    if not os.path.exists(LOG_PATH):
        if args.mode == "train":
            os.makedirs(LOG_PATH)
        else:
            raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, args.mode + "_" + nowTime)
    # logger = _init_logger(path=log_path)
    # file_handler = logging.FileHandler(log_path)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)

    # dataset
    hps = args
    dbPipe = ExtCNNDMPipe(vocab_size=hps.vocab_size,
                          vocab_path=VOCAL_FILE,
                          sent_max_len=hps.sent_max_len,
                          doc_max_timesteps=hps.doc_max_timesteps)
    if hps.mode == 'test':
        hps.recurrent_dropout_prob = 0.0
        hps.atten_dropout_prob = 0.0
        hps.ffn_dropout_prob = 0.0
        logger.info(hps)
        paths = {"test": DATA_FILE}
        db = dbPipe.process_from_file(paths)
    else:
        paths = {"train": DATA_FILE, "valid": VALID_FILE}
        db = dbPipe.process_from_file(paths)


    # embedding
    if args.embedding == "glove":
        vocab = db.get_vocab("vocab")
        embed = torch.nn.Embedding(len(vocab), hps.word_emb_dim)
        if hps.word_embedding:
            embed_loader = EmbedLoader()
            pretrained_weight = embed_loader.load_with_vocab(hps.embedding_path, vocab)  # unfound with random init
            embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            embed.weight.requires_grad = hps.embed_train
    else:
        logger.error("[ERROR] embedding To Be Continued!")
        sys.exit(1)

    # model
    if args.sentence_encoder == "transformer" and args.sentence_decoder == "SeqLab":
        model_param = json.load(open("config/transformer.config", "rb"))
        hps.__dict__.update(model_param)
        model = TransformerModel(hps, embed)
    elif args.sentence_encoder == "deeplstm" and args.sentence_decoder == "SeqLab":
        model_param = json.load(open("config/deeplstm.config", "rb"))
        hps.__dict__.update(model_param)
        model = SummarizationModel(hps, embed)
    else:
        logger.error("[ERROR] Model To Be Continued!")
        sys.exit(1)
    if hps.cuda:
        model = model.cuda()
        logger.info("[INFO] Use cuda")

    logger.info(hps)

    if hps.mode == 'train':
        db.get_dataset("valid").set_target("text", "summary")
        setup_training(model, db.get_dataset("train"), db.get_dataset("valid"), hps)
    elif hps.mode == 'test':
        logger.info("[INFO] Decoding...")
        db.get_dataset("test").set_target("text", "summary")
        run_test(model, db.get_dataset("test"), hps, limited=hps.limited)
    else:
        logger.error("The 'mode' flag must be one of train/eval/test")
        raise ValueError("The 'mode' flag must be one of train/eval/test")

if __name__ == '__main__':
    main()
