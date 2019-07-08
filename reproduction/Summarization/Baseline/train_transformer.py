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
import time
import copy
import pickle
import datetime
import argparse
import logging

import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable

from rouge import Rouge

sys.path.append('/remote-home/dqwang/FastNLP/fastNLP/')

from fastNLP.core.batch import Batch
from fastNLP.core.const import Const
from fastNLP.io.model_io import ModelLoader, ModelSaver
from fastNLP.core.sampler import BucketSampler

from tools import utils
from tools.logger import *
from data.dataloader import SummarizationLoader
from model.TransformerModel import TransformerModel

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

    try:
        run_training(model, train_loader, valid_loader, hps) # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_file = os.path.join(train_dir, "earlystop.pkl")
        saver = ModelSaver(save_file)
        saver.save_pytorch(model)
        logger.info('[INFO] Saving early stop model to %s', save_file)

def run_training(model, train_loader, valid_loader, hps):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    logger.info("[INFO] Starting run_training")

    train_dir = os.path.join(hps.save_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    lr = hps.lr
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.98),
                                 # eps=1e-09)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_train_loss = None
    best_train_F= None
    best_loss = None
    best_F = None
    step_num = 0
    non_descent_cnt = 0
    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        total_example_num = 0
        match, pred, true, match_true = 0.0, 0.0, 0.0, 0.0
        epoch_start_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # if i > 10:
            #     break
            model.train()

            iter_start_time=time.time()

            input, input_len = batch_x[Const.INPUT], batch_x[Const.INPUT_LEN]
            label = batch_y[Const.TARGET]

            # logger.info(batch_x["text"][0])
            # logger.info(input[0,:,:])
            # logger.info(input_len[0:5,:])
            # logger.info(batch_y["summary"][0:5])
            # logger.info(label[0:5,:])

            # logger.info((len(batch_x["text"][0]), sum(input[0].sum(-1) != 0)))

            batch_size, N, seq_len = input.size()

            if hps.cuda:
                input = input.cuda()    # [batch, N, seq_len]
                label = label.cuda()
                input_len = input_len.cuda()

            input = Variable(input)
            label = Variable(label)
            input_len = Variable(input_len)

            model_outputs = model.forward(input, input_len)  # [batch, N, 2]

            outputs = model_outputs[Const.OUTPUT].view(-1, 2)

            label = label.view(-1)

            loss = criterion(outputs, label)    # [batch_size, doc_max_timesteps]
            input_len = input_len.float().view(-1)
            loss = loss * input_len
            loss = loss.view(batch_size, -1)
            loss = loss.sum(1).mean()

            if not (np.isfinite(loss.data)).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        logger.info(param.grad.data.sum())
                raise Exception("train Loss is not finite. Stopping.")

            optimizer.zero_grad()
            loss.backward()
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            optimizer.step()
            step_num += 1

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 100 == 0:
                # start debugger
                # import pdb; pdb.set_trace()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.debug(name)
                        logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                                .format(i, (time.time() - iter_start_time),
                                               float(train_loss / 100)))
                train_loss = 0.0

            # calculate the precision, recall and F
            prediction = outputs.max(1)[1]
            prediction = prediction.data
            label = label.data
            pred += prediction.sum()
            true += label.sum()
            match_true += ((prediction == label) & (prediction == 1)).sum()
            match += (prediction == label).sum()
            total_example_num += int(batch_size * N)

        if hps.lr_descent:
            # new_lr = pow(hps.hidden_size, -0.5) * min(pow(step_num, -0.5),
            #                                           step_num * pow(hps.warmup_steps, -1.5))
            new_lr = max(5e-6, lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                                .format(epoch, (time.time() - epoch_start_time),
                                           float(epoch_avg_loss)))

        logger.info("[INFO] Trainset match_true %d, pred %d, true %d, total %d, match %d", match_true, pred, true, total_example_num, match)
        accu, precision, recall, F = utils.eval_label(match_true, pred, true, total_example_num, match)
        logger.info("[INFO] The size of totalset is %d, accu is %f, precision is %f, recall is %f, F is %f", total_example_num / hps.doc_max_timesteps, accu, precision, recall, F)

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel.pkl")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss), save_file)
            saver = ModelSaver(save_file)
            saver.save_pytorch(model)
            best_train_loss = epoch_avg_loss
        elif epoch_avg_loss > best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_file = os.path.join(train_dir, "earlystop.pkl")
            saver = ModelSaver(save_file)
            saver.save_pytorch(model)
            logger.info('[INFO] Saving early stop model to %s', save_file)
            return

        if not best_train_F or F > best_train_F:
            save_file = os.path.join(train_dir, "bestFmodel.pkl")
            logger.info('[INFO] Found new best model with %.3f F score. Saving to %s', float(F), save_file)
            saver = ModelSaver(save_file)
            saver.save_pytorch(model)
            best_train_F = F

        best_loss, best_F, non_descent_cnt = run_eval(model, valid_loader, hps, best_loss, best_F, non_descent_cnt)

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_file = os.path.join(train_dir, "earlystop")
            saver = ModelSaver(save_file)
            saver.save_pytorch(model)
            logger.info('[INFO] Saving early stop model to %s', save_file)
            return

def run_eval(model, loader, hps, best_loss, best_F, non_descent_cnt):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    logger.info("[INFO] Starting eval for this model ...")
    eval_dir = os.path.join(hps.save_root, "eval") # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    model.eval()

    running_loss = 0.0
    match, pred, true, match_true = 0.0, 0.0, 0.0, 0.0
    pairs = {}
    pairs["hyps"] = []
    pairs["refer"] = []
    total_example_num = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    iter_start_time = time.time()

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            # if i > 10:
            #     break

            input, input_len = batch_x[Const.INPUT], batch_x[Const.INPUT_LEN]
            label = batch_y[Const.TARGET]

            if hps.cuda:
                input = input.cuda()    # [batch, N, seq_len]
                label = label.cuda()
                input_len = input_len.cuda()

            batch_size, N, _ = input.size()

            input = Variable(input, requires_grad=False)
            label = Variable(label)
            input_len = Variable(input_len, requires_grad=False)

            model_outputs = model.forward(input,input_len) # [batch, N, 2]
            outputs = model_outputs[Const.OUTPUTS]
            prediction = model_outputs["prediction"]

            outputs = outputs.view(-1, 2)  # [batch * N, 2]
            label = label.view(-1)  # [batch * N]
            loss = criterion(outputs, label)
            input_len = input_len.float().view(-1)
            loss = loss * input_len
            loss = loss.view(batch_size, -1)
            loss = loss.sum(1).mean()
            running_loss += float(loss.data)

            label = label.data
            pred += prediction.sum()
            true += label.sum()
            match_true += ((prediction == label) & (prediction == 1)).sum()
            match += (prediction == label).sum()
            total_example_num += batch_size * N

            # rouge
            prediction = prediction.view(batch_size, -1)
            for j in range(batch_size):
                original_article_sents = batch_x["text"][j]
                sent_max_number = len(original_article_sents)
                refer = "\n".join(batch_x["summary"][j])
                hyps = "\n".join(original_article_sents[id] for id in range(len(prediction[j])) if prediction[j][id]==1 and id < sent_max_number)
                if sent_max_number < hps.m and len(hyps) <= 1:
                    logger.error("sent_max_number is too short %d, Skip!" , sent_max_number)
                    continue

                if len(hyps) >= 1 and hyps != '.':
                    # logger.debug(prediction[j])
                    pairs["hyps"].append(hyps)
                    pairs["refer"].append(refer)
                elif refer == "." or refer == "":
                    logger.error("Refer is None!")
                    logger.debug("label:")
                    logger.debug(label[j])
                    logger.debug(refer)
                elif hyps == "." or hyps == "":
                    logger.error("hyps is None!")
                    logger.debug("sent_max_number:%d", sent_max_number)
                    logger.debug("prediction:")
                    logger.debug(prediction[j])
                    logger.debug(hyps)
                else:
                    logger.error("Do not select any sentences!")
                    logger.debug("sent_max_number:%d", sent_max_number)
                    logger.debug(original_article_sents)
                    logger.debug("label:")
                    logger.debug(label[j])
                    continue

    running_avg_loss = running_loss / len(loader)

    if hps.use_pyrouge:
        logger.info("The number of pairs is %d", len(pairs["hyps"]))
        logging.getLogger('global').setLevel(logging.WARNING)
        if not len(pairs["hyps"]):
            logger.error("During testing, no hyps is selected!")
            return
        if isinstance(pairs["refer"][0], list):
            logger.info("Multi Reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(pairs["hyps"], pairs["refer"])
        else:
            scores_all = utils.pyrouge_score_all(pairs["hyps"], pairs["refer"])
    else:
        if len(pairs["hyps"]) == 0 or len(pairs["refer"]) == 0 :
            logger.error("During testing, no hyps is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(pairs["hyps"], pairs["refer"], avg=True)
        # try:
        #     scores_all = rouge.get_scores(pairs["hyps"], pairs["refer"], avg=True)
        # except ValueError as e:
        #     logger.error(repr(e))
        #     scores_all = []
        #     for idx in range(len(pairs["hyps"])):
        #         try:
        #             scores = rouge.get_scores(pairs["hyps"][idx], pairs["refer"][idx])[0]
        #             scores_all.append(scores)
        #         except ValueError as e:
        #             logger.error(repr(e))
        #             logger.debug("HYPS:\t%s", pairs["hyps"][idx])
        #             logger.debug("REFER:\t%s", pairs["refer"][idx])
        # finally:
        #     logger.error("During testing, some errors happen!")
        #     logger.error(len(scores_all))
        #     exit(1)

    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '
                        .format((time.time() - iter_start_time),
                                       float(running_avg_loss)))

    logger.info("[INFO] Validset match_true %d, pred %d, true %d, total %d, match %d", match_true, pred, true, total_example_num, match)
    accu, precision, recall, F = utils.eval_label(match_true, pred, true, total_example_num, match)
    logger.info("[INFO] The size of totalset is %d, accu is %f, precision is %f, recall is %f, F is %f",
                total_example_num / hps.doc_max_timesteps, accu, precision, recall, F)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel.pkl') # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info('[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s', float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s', float(running_avg_loss), bestmodel_save_path)
        saver = ModelSaver(bestmodel_save_path)
        saver.save_pytorch(model)
        best_loss = running_avg_loss
        non_descent_cnt = 0
    else:
        non_descent_cnt += 1

    if best_F is None or best_F < F:
        bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel.pkl') # this is where checkpoints of best models are saved
        if best_F is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F), float(best_F), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original loss is None, Saving to %s', float(F), bestmodel_save_path)
        saver = ModelSaver(bestmodel_save_path)
        saver.save_pytorch(model)
        best_F = F

    return best_loss, best_F, non_descent_cnt

def run_test(model, loader, hps, limited=False):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    test_dir = os.path.join(hps.save_root, "test") # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(eval_dir) :
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    if hps.test_model == "evalbestmodel":
        bestmodel_load_path = os.path.join(eval_dir, 'bestmodel.pkl') # this is where checkpoints of best models are saved
    elif hps.test_model == "evalbestFmodel":
        bestmodel_load_path = os.path.join(eval_dir, 'bestFmodel.pkl')
    elif hps.test_model == "trainbestmodel":
        train_dir = os.path.join(hps.save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'bestmodel.pkl')
    elif hps.test_model == "trainbestFmodel":
        train_dir = os.path.join(hps.save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'bestFmodel.pkl')
    elif hps.test_model == "earlystop":
        train_dir = os.path.join(hps.save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop,pkl')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    logger.info("[INFO] Restoring %s for testing...The path is %s", hps.test_model, bestmodel_load_path)

    modelloader = ModelLoader()
    modelloader.load_pytorch(model, bestmodel_load_path)

    import datetime
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')#现在
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.data_path.split("/")[-1])
        resfile = open(log_dir, "w")
    else:
        log_dir = os.path.join(test_dir, nowTime)
        resfile = open(log_dir, "wb")
    logger.info("[INFO] Write the Evaluation into %s", log_dir)

    model.eval()

    match, pred, true, match_true = 0.0, 0.0, 0.0, 0.0
    total_example_num = 0.0
    pairs = {}
    pairs["hyps"] = []
    pairs["refer"] = []
    pred_list = []
    iter_start_time=time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):

            input, input_len = batch_x[Const.INPUT], batch_x[Const.INPUT_LEN]
            label = batch_y[Const.TARGET]

            if hps.cuda:
                input = input.cuda()    # [batch, N, seq_len]
                label = label.cuda()
                input_len = input_len.cuda()

            batch_size, N, _ = input.size()

            input = Variable(input)
            input_len = Variable(input_len, requires_grad=False)

            model_outputs = model.forward(input, input_len)  # [batch, N, 2]
            prediction = model_outputs["pred"]

            if hps.save_label:
                pred_list.extend(model_outputs["pred_idx"].data.cpu().view(-1).tolist())
                continue

            pred += prediction.sum()
            true += label.sum()
            match_true += ((prediction == label) & (prediction == 1)).sum()
            match += (prediction == label).sum()
            total_example_num += batch_size * N

            for j in range(batch_size):
                original_article_sents = batch_x["text"][j]
                sent_max_number = len(original_article_sents)
                refer = "\n".join(batch_x["summary"][j])
                hyps = "\n".join(original_article_sents[id].replace("\n", "") for id in range(len(prediction[j])) if prediction[j][id]==1 and id < sent_max_number)
                if limited:
                    k = len(refer.split())
                    hyps = " ".join(hyps.split()[:k])
                    logger.info((len(refer.split()),len(hyps.split())))
                resfile.write(b"Original_article:")
                resfile.write("\n".join(batch_x["text"][j]).encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"Reference:")
                if isinstance(refer, list):
                    for ref in refer:
                        resfile.write(ref.encode('utf-8'))
                        resfile.write(b"\n")
                        resfile.write(b'*' * 40)
                        resfile.write(b"\n")
                else:
                    resfile.write(refer.encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"hypothesis:")
                resfile.write(hyps.encode('utf-8'))
                resfile.write(b"\n")

                if hps.use_pyrouge:
                    pairs["hyps"].append(hyps)
                    pairs["refer"].append(refer)
                else:
                    try:
                        scores = utils.rouge_all(hyps, refer)
                        pairs["hyps"].append(hyps)
                        pairs["refer"].append(refer)
                    except ValueError:
                        logger.error("Do not select any sentences!")
                        logger.debug("sent_max_number:%d", sent_max_number)
                        logger.debug(original_article_sents)
                        logger.debug("label:")
                        logger.debug(label[j])
                        continue

                    # single example res writer
                    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']) \
                            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']) \
                                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'])

                    resfile.write(res.encode('utf-8'))
                resfile.write(b'-' * 89)
                resfile.write(b"\n")

    if hps.save_label:
        import json
        json.dump(pred_list, resfile)
        logger.info('   | end of test | time: {:5.2f}s | '.format((time.time() - iter_start_time)))
        return

    resfile.write(b"\n")
    resfile.write(b'=' * 89)
    resfile.write(b"\n")

    if hps.use_pyrouge:
        logger.info("The number of pairs is %d", len(pairs["hyps"]))
        if not len(pairs["hyps"]):
            logger.error("During testing, no hyps is selected!")
            return
        if isinstance(pairs["refer"][0], list):
            logger.info("Multi Reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(pairs["hyps"], pairs["refer"])
        else:
            scores_all = utils.pyrouge_score_all(pairs["hyps"], pairs["refer"])
    else:
        logger.info("The number of pairs is %d", len(pairs["hyps"]))
        if not len(pairs["hyps"]):
            logger.error("During testing, no hyps is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(pairs["hyps"], pairs["refer"], avg=True)

    # the whole model res writer
    resfile.write(b"The total testset is:")
    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    resfile.write(res.encode("utf-8"))
    logger.info(res)
    logger.info('   | end of test | time: {:5.2f}s | '
                        .format((time.time() - iter_start_time)))



    # label prediction
    logger.info("match_true %d, pred %d, true %d, total %d, match %d", match, pred, true, total_example_num, match)
    accu, precision, recall, F = utils.eval_label(match_true, pred, true, total_example_num, match)
    res = "The size of totalset is %d, accu is %f, precision is %f, recall is %f, F is %f" % (total_example_num / hps.doc_max_timesteps, accu, precision, recall, F)
    resfile.write(res.encode('utf-8'))
    logger.info("The size of totalset is %d, accu is %f, precision is %f, recall is %f, F is %f", len(loader), accu, precision, recall, F)


def main():
    parser = argparse.ArgumentParser(description='Transformer Model')

    # Where to find data
    parser.add_argument('--data_path', type=str, default='/remote-home/dqwang/Datasets/CNNDM/train.label.jsonl', help='Path expression to pickle datafiles.')
    parser.add_argument('--valid_path', type=str, default='/remote-home/dqwang/Datasets/CNNDM/val.label.jsonl', help='Path expression to pickle valid datafiles.')
    parser.add_argument('--vocab_path', type=str, default='/remote-home/dqwang/Datasets/CNNDM/vocab', help='Path expression to text vocabulary file.')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--mode', type=str, default='train', help='must be one of train/test')
    parser.add_argument('--restore_model', type=str , default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')
    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [evalbestmodel/evalbestFmodel/trainbestmodel/trainbestFmodel/earlystop]')
    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')

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
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 200]')
    parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
    parser.add_argument('--min_kernel_size', type=int, default=1, help='kernel min length for CNN [default:1]')
    parser.add_argument('--max_kernel_size', type=int, default=7, help='kernel max length for CNN [default:7]')
    parser.add_argument('--output_channel', type=int, default=50, help='output channel: repeated times for one kernel')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of deeplstm layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size [default: 512]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=2048, help='PositionwiseFeedForward inner hidden size [default: 2048]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1,help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
    parser.add_argument('--sent_max_len', type=int, default=100, help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50, help='max length of documents (max timesteps of documents)')
    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')

    # Training
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup_steps')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')
    parser.add_argument('--limited', action='store_true', default=False, help='limited decode summary length')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    hps = args

    # File paths
    DATA_FILE = args.data_path
    VALID_FILE = args.valid_path
    VOCAL_FILE = args.vocab_path
    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        if hps.mode == "train":
            os.makedirs(LOG_PATH)
        else:
            logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
            raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, hps.mode + "_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info(args)
    logger.info(args)

    sum_loader = SummarizationLoader()


    if hps.mode == 'test':
        paths = {"test": DATA_FILE}
        hps.recurrent_dropout_prob = 0.0
        hps.atten_dropout_prob = 0.0
        hps.ffn_dropout_prob = 0.0
        logger.info(hps)
    else:
        paths = {"train": DATA_FILE, "valid": VALID_FILE}

    dataInfo = sum_loader.process(paths=paths, vocab_size=hps.vocab_size, vocab_path=VOCAL_FILE, sent_max_len=hps.sent_max_len, doc_max_timesteps=hps.doc_max_timesteps, load_vocab=os.path.exists(VOCAL_FILE))

    vocab = dataInfo.vocabs["vocab"]
    model = TransformerModel(hps, vocab)

    if len(hps.gpu) > 1:
        gpuid = hps.gpu.split(',')
        gpuid = [int(s) for s in gpuid]
        model = nn.DataParallel(model,device_ids=gpuid)
        logger.info("[INFO] Use Multi-gpu: %s", hps.gpu)
    if hps.cuda:
        model = model.cuda()
        logger.info("[INFO] Use cuda")

    if hps.mode == 'train':
        trainset = dataInfo.datasets["train"]
        train_sampler = BucketSampler(batch_size=hps.batch_size, seq_len_field_name=Const.INPUT)
        train_batch = Batch(batch_size=hps.batch_size, dataset=trainset, sampler=train_sampler)
        validset = dataInfo.datasets["valid"]
        validset.set_input("text", "summary")
        valid_batch = Batch(batch_size=hps.batch_size, dataset=validset)
        setup_training(model, train_batch, valid_batch, hps)
    elif hps.mode == 'test':
        logger.info("[INFO] Decoding...")
        testset = dataInfo.datasets["test"]
        testset.set_input("text", "summary")
        test_batch = Batch(batch_size=hps.batch_size, dataset=testset)
        run_test(model, test_batch, hps, limited=hps.limited)
    else:
        logger.error("The 'mode' flag must be one of train/eval/test")
        raise ValueError("The 'mode' flag must be one of train/eval/test")

if __name__ == '__main__':
    main()
