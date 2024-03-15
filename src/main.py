# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from disenqnet import DisenQNet, ConceptModel, DisenQNetPlusD, DisenQNetPlusS
from dataset import QuestionDataset, DiffTrainDataset, DiffRankDataset, SimTrainDataset, SimRankDataset

def parse_args():
    parser = ArgumentParser("DisenQNet")
    # runtime args
    parser.add_argument("--dataset", type=str, dest="dataset", default="data/math23k")
    parser.add_argument("--cuda", type=str, dest="cuda", default=None)
    parser.add_argument("--seed", type=int, dest="seed", default=0)
    parser.add_argument("--log", type=str, dest="log", default=None)

    # model args
    parser.add_argument("--hidden", type=int, dest="hidden", default=128)
    parser.add_argument("--dropout", type=float, dest="dropout", default=0.2)
    parser.add_argument("--cp", type=float, dest="cp", default=1.5)
    parser.add_argument("--mi", type=float, dest="mi", default=1.0)
    parser.add_argument("--dis", type=float, dest="dis", default=2.0)
    parser.add_argument("--sup", type=float, dest="sup", default=1.0)
    parser.add_argument("--un", type=float, dest="un", default=0.1)
    parser.add_argument("--norm", type=float, dest="norm", default=1e-4)
    parser.add_argument("--pos-weight", type=float, dest="pos_weight", default=1)

    # training args
    parser.add_argument("--epoch", type=int, dest="epoch", default=50)
    parser.add_argument("--batch", type=int, dest="batch", default=128)
    parser.add_argument("--lr", type=float, dest="lr", default=1e-3)
    parser.add_argument("--step", type=int, dest="step", default=20)
    parser.add_argument("--gamma", type=float, dest="gamma", default=0.5)
    parser.add_argument("--samples", type=int, dest="samples", default=5)

    # eval args
    parser.add_argument("--vi", action="store_true", dest="vi", default=False)
    parser.add_argument("--topk", type=int, dest="topk", default=2)
    parser.add_argument("--top-rank", type=int, dest="top_rank", default=5)
    parser.add_argument("--reduction", type=str, dest="reduction", default="micro")

    # dataset args
    parser.add_argument("--trim-min", type=int, dest="trim_min", default=50)
    parser.add_argument("--max-len", type=int, dest="max_len", default=250)

    # adversarial training args
    parser.add_argument("--adv", type=int, dest="adv", default=10)
    parser.add_argument("--warm-up", type=int, dest="warm_up", default=5)

    args = parser.parse_args()
    return args

def init():
    args = parse_args()
    # cuda
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # log
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename=args.log)
    return args

def main(args):
    # dataset
    # pretrain and concept
    train_path = os.path.join(args.dataset, "train.json")
    test_path = os.path.join(args.dataset, "test.json")
    wv_path = os.path.join(args.dataset, "wv.th")
    word_path = os.path.join(args.dataset, "vocab.list")
    concept_path = os.path.join(args.dataset, "concept.list")
    train_dataset = QuestionDataset(train_path, wv_path, word_path, concept_path, args.hidden, args.trim_min, args.max_len, "train", silent=False)
    test_dataset = QuestionDataset(test_path, wv_path, word_path, concept_path, args.hidden, args.trim_min, args.max_len, "test", silent=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=train_dataset.collate_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collate_data)

    sup_train_path = os.path.join(args.dataset, "sup-train.json")
    sup_test_path = os.path.join(args.dataset, "sup-test.json")
    sup_rank_path = os.path.join(args.dataset, "sup-rank.json")

    # difficulty
    diff_train_dataset = DiffTrainDataset(sup_train_path, wv_path, word_path, concept_path, args.max_len, args.samples, silent=False)
    diff_test_dataset = DiffRankDataset(sup_test_path, sup_rank_path, wv_path, word_path, concept_path, args.max_len, silent=False)
    diff_train_dataloader = DataLoader(diff_train_dataset, batch_size=args.batch, shuffle=True, collate_fn=diff_train_dataset.collate_data)

    # similarity
    # sim_train_dataset = SimTrainDataset(sup_train_path, wv_path, word_path, concept_path, args.max_len, args.samples, silent=False)
    # sim_test_dataset = SimRankDataset(sup_test_path, sup_rank_path, wv_path, word_path, concept_path, args.max_len, silent=False)
    # sim_train_dataloader = DataLoader(sim_train_dataset, batch_size=args.batch, shuffle=True, collate_fn=sim_train_dataset.collate_data)

    # model train and test
    vocab_size = train_dataset.vocab_size
    concept_size = train_dataset.concept_size
    wv = train_dataset.word2vec
    
    # vocab_size = diff_train_dataset.vocab_size
    # concept_size = diff_train_dataset.concept_size
    # wv = diff_train_dataset.wv

    # vocab_size = sim_train_dataset.vocab_size
    # concept_size = sim_train_dataset.concept_size
    # wv = sim_train_dataset.wv

    # pretrain
    disen_q_net = DisenQNet(vocab_size, concept_size, args.hidden, args.dropout, args.pos_weight, args.cp, args.mi, args.dis, wv)
    disen_q_net.train(train_dataloader, test_dataloader, args.device, args.epoch, args.lr, args.step, args.gamma, args.warm_up, args.adv, silent=False)
    disen_q_net.save("disen_q_net.th")
    disen_q_net.load("disen_q_net.th")

    # concept
    concept_model = ConceptModel(concept_size, disen_q_net.disen_q_net, args.dropout, args.pos_weight)
    concept_model.train(train_dataloader, test_dataloader, args.device, args.epoch, args.lr, args.step, args.gamma, silent=False, use_vi=args.vi, top_k=args.topk, reduction=args.reduction)
    # concept_model.save("concept_model.th")
    # concept_model.load("concept_model.th")

    # difficulty
    diff_model = DisenQNetPlusD(disen_q_net.disen_q_net, args.sup, args.un, args.norm, args.dropout, wv)
    diff_model.train(diff_train_dataloader, diff_test_dataset, args.device, args.epoch, args.lr, args.step, args.gamma, silent=False, use_vi=args.vi, n_targets=args.top_rank, n_predicts=args.top_rank)
    # diff_model.save("diff_model.th")
    # diff_model.load("diff_model.th")

    # similarity
    # sim_model = DisenQNetPlusS(disen_q_net.disen_q_net, args.sup, args.un, args.norm, args.dropout, wv)
    # sim_model.train(sim_train_dataloader, sim_test_dataset, args.device, args.epoch, args.lr, args.step, args.gamma, silent=False, use_vi=args.vi, n_predicts=args.top_rank)
    # # sim_model.save("sim_model.th")
    # # sim_model.load("sim_model.th")
    return

if __name__ == "__main__":
    args = init()
    main(args)
