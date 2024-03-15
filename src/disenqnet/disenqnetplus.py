# -*- coding: utf-8 -*-

import logging
import torch
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from .modules import TextEncoder, VectorMILoss
from .utils import MLP, PairwiseLoss
from .metrics import DOA, mAP_difficulty, nDCG_difficulty, mAP_similarity, nDCG_similarity, f1_similarity
from .disenqnet import get_disen_q_net_hidden

class DisenQNetPlusD(object):
    """
        Difficulty ranking training and evaluation model, with pretrained and fixed DisenQNet
        :param disen_q_net: QuestionEncoder, pretrained DisenQNet model
        :param w_sup: float, weight of supervised task loss
        :param w_un: float, weight of unsupervised MI loss
        :param w_norm: float, weight of L2 normalization loss
        :param dropout: float, dropout rate
        :param wv: Tensor of (vocab_size, hidden_dim), initial word embedding, default = None
    """

    def __init__(self, disenq_q_net, w_sup, w_un, w_norm, dropout, wv=None):
        super(DisenQNetPlusD, self).__init__()
        vocab_size = disenq_q_net.vocab_size
        hidden_dim = disenq_q_net.hidden_dim
        self.disen_q_net = disenq_q_net
        self.task_encoder = TextEncoder(vocab_size, hidden_dim, dropout, wv=wv)
        self.estimator = MLP(hidden_dim, 1, hidden_dim, dropout, n_layers=3)
        self.proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.w_sup = w_sup
        self.w_un = w_un
        self.sup_loss = PairwiseLoss(1, w_norm, reduction="mean")
        self.un_loss = VectorMILoss(hidden_dim, dropout)
        self.modules = (self.disen_q_net, self.task_encoder, self.estimator, self.proj, self.un_loss)
        self.norm_params = list(self.estimator.parameters()) + list(self.task_encoder.parameters())
        return
    
    def compute_loss(self, data, device, use_vi):
        text1, length1, text2, length2 = data
        text1, length1, text2, length2 = text1.to(device), length1.to(device), text2.to(device), length2.to(device)
        _, sup_hidden1 = self.task_encoder(text1)
        _, sup_hidden2 = self.task_encoder(text2)
        diff1 = self.estimator(sup_hidden1).squeeze(-1)
        diff2 = self.estimator(sup_hidden2).squeeze(-1)
        sup_loss, _ = self.sup_loss(diff1, diff2, self.norm_params)

        if self.w_un > 0:
            un_hidden1 = F.leaky_relu(self.proj(get_disen_q_net_hidden(text1, length1, self.disen_q_net, use_vi)))
            un_hidden2 = F.leaky_relu(self.proj(get_disen_q_net_hidden(text2, length2, self.disen_q_net, use_vi)))
            un_loss1 = self.un_loss(sup_hidden1, un_hidden1)
            un_loss2 = self.un_loss(sup_hidden2, un_hidden2)
            un_loss = (un_loss1 + un_loss2) / 2
            loss = sup_loss * self.w_sup + un_loss * self.w_un
        else:
            loss = sup_loss * self.w_sup
        return loss, sup_loss
    
    def train(self, train_data, test_data, device, epoch, lr, step_size, gamma, silent, use_vi, n_targets, n_predicts):
        """
            Difficulty ranking model train
            :param train_data: iterable, paired train dataset, contains (text1, length1, text2, length2), difficulty of text1 is larger than text2
                text1: Tensor of (batch_size, seq_len)
                length1: Tensor of (batch_size)
                text2: Tensor of (batch_size, seq_len)
                length2: Tensor of (batch_size)
            :param test_data: iterable, test dataset for ranking
            :param device: str, cpu or cuda
            :param epoch: int, number of epoch
            :param lr: float, initial learning rate
            :param step_size: int, step_size for StepLR, period of learning rate decay
            :param gamma: float, gamma for StepLR, multiplicative factor of learning rate decay
            :param silent: bool, whether to log loss
            :param use_vi: bool, whether to use vi or vk hidden of DisenQNet for training
            :param n_targets: int, number of top-k targets as positive items in ranking evaluation for testing
            :param n_predicts: int, number of top-k predictions as positive items in ranking evaluation for testing
        """
        # optimizer & scheduler
        model_params = list(self.task_encoder.parameters()) + list(self.estimator.parameters()) + list(self.proj.parameters()) + list(self.un_loss.parameters())
        optimizer = Adam(model_params, lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        # train
        for epoch_idx in range(epoch):
            epoch_idx += 1

            self.to(device)
            self.set_mode(True)
            train_data.dataset.sample()
            for data in train_data:
                loss, sup_loss = self.compute_loss(data, device, use_vi)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # test
            loss, sup_loss = self.eval_loss(train_data, device, use_vi)
            if test_data is not None:
                map, ndcg, doa = self.eval_rank(test_data, device, n_targets, n_predicts)
                if not silent:
                    logging.info(f"[Epoch {epoch_idx:2d}] train loss: {loss:.4f}, sup_loss: {sup_loss:.4f}, eval map: {map:.3f}, ndcg: {ndcg:.3f}, doa: {doa:.3f}")
            elif not silent:
                logging.info(f"[Epoch {epoch_idx:2d}] train loss: {loss:.4f}, sup_loss: {sup_loss:.4f}")
        return
    
    def eval_rank(self, test_data, device, n_targets, n_predicts):
        """
            Difficulty ranking model test
            :param test_data: iterable, test dataset for ranking, contains (texts, lengths, difficulty_list, target) as a question list to rank
                texts: Tensor of (list_size, seq_len)
                lengths: Tensor of (list_size)
                difficulty_list: list of float, true difficulty of questions to rank
                target: list of list of int, indexes of questions in target rank, each rank has several questions in a list with same difficulty
            :param device: str, cpu or cuda
            :param n_targets: int, number of top-k targets as positive items in ranking evaluation
            :param n_predicts: int, number of top-k predictions as positive items in ranking evaluation
            :returns: (map, ndcg, doa)
        """
        self.to(device)
        self.set_mode(False)
        total_size = 0
        total_map = 0
        total_ndcg = 0
        total_doa = 0
        with torch.no_grad():
            for data in test_data:
                texts, lengths, difficulty_list, target = data
                texts = texts.to(device)
                _, hiddens = self.task_encoder(texts)
                diffs = self.estimator(hiddens).squeeze(-1)
                map = mAP_difficulty(target, diffs, n_targets, n_predicts)
                ndcg = nDCG_difficulty(target, diffs, difficulty_list, n_targets, n_predicts)
                doa = DOA(torch.tensor(difficulty_list).to(device), diffs)
                total_size += 1
                total_map += map
                total_ndcg += ndcg
                total_doa += doa
        map = total_map / total_size
        ndcg = total_ndcg / total_size
        doa = total_doa / total_size
        return map, ndcg, doa
    
    def eval_loss(self, test_data, device, use_vi):
        """
            Difficulty ranking model test
            :param test_data: iterable, paired test dataset, contains (text1, length1, text2, length2), difficulty of text1 is larger than text2
                text1: Tensor of (batch_size, seq_len)
                length1: Tensor of (batch_size)
                text2: Tensor of (batch_size, seq_len)
                length2: Tensor of (batch_size)
            :param device: str, cpu or cuda
            :param use_vi: bool, whether to use vi or vk hidden of DisenQNet for training
            :returns: (loss, sup_loss)
        """
        self.to(device)
        self.set_mode(False)
        total_size = 0
        total_loss = 0
        total_sup_loss = 0
        with torch.no_grad():
            for data in test_data:
                loss, sup_loss = self.compute_loss(data, device, use_vi)
                batch_size = data[0].size(0)
                loss, sup_loss = loss.item(), sup_loss.item()
                total_size += batch_size
                total_loss += loss * batch_size
                total_sup_loss += sup_loss * batch_size
        loss = total_loss / total_size
        sup_loss = total_sup_loss / total_size
        return loss, sup_loss
    
    def save(self, filepath):
        state_dicts = [module.state_dict() for module in self.modules]
        torch.save(state_dicts, filepath)
        return
    
    def load(self, filepath):
        state_dicts = torch.load(filepath)
        for module, state_dict in zip(self.modules, state_dicts):
            module.load_state_dict(state_dict)
        return
    
    def to(self, device):
        for module in self.modules:
            module.to(device)
        return
    
    def set_mode(self, train):
        self.disen_q_net.eval()
        for module in self.modules:
            if module is not self.disen_q_net:
                if train:
                    module.train()
                else:
                    module.eval()
        return

class DisenQNetPlusS(object):
    """
        Similarity ranking training and evaluation model, with pretrained and fixed DisenQNet
        :param disen_q_net: QuestionEncoder, pretrained DisenQNet model
        :param w_sup: float, weight of supervised task loss
        :param w_un: float, weight of unsupervised MI loss
        :param w_norm: float, weight of L2 normalization loss
        :param dropout: float, dropout rate
        :param wv: Tensor of (vocab_size, hidden_dim), initial word embedding, default = None
    """

    def __init__(self, disenq_q_net, w_sup, w_un, w_norm, dropout, wv=None):
        super(DisenQNetPlusS, self).__init__()
        vocab_size = disenq_q_net.vocab_size
        hidden_dim = disenq_q_net.hidden_dim
        self.disen_q_net = disenq_q_net
        self.task_encoder = TextEncoder(vocab_size, hidden_dim, dropout, wv=wv)
        self.estimator = MLP(hidden_dim*2, 1, hidden_dim*2, dropout, n_layers=3)
        self.proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.w_sup = w_sup
        self.w_un = w_un
        self.sup_loss = PairwiseLoss(1, w_norm, reduction="mean")
        self.un_loss = VectorMILoss(hidden_dim, dropout)
        self.modules = (self.disen_q_net, self.task_encoder, self.estimator, self.proj, self.un_loss)
        self.norm_params = list(self.estimator.parameters()) + list(self.task_encoder.parameters())
        return
    
    def compute_loss(self, data, device, use_vi):
        text_q, length_q, text_p, length_p, text_n, length_n = data
        text_q, length_q, text_p, length_p, text_n, length_n = text_q.to(device), length_q.to(device), text_p.to(device), length_p.to(device), text_n.to(device), length_n.to(device)
        _, sup_hidden_q = self.task_encoder(text_q)
        _, sup_hidden1 = self.task_encoder(text_p)
        _, sup_hidden2 = self.task_encoder(text_n)
        sim1 = self.estimator(torch.cat((sup_hidden_q, sup_hidden1), dim=-1)).squeeze(-1)
        sim2 = self.estimator(torch.cat((sup_hidden_q, sup_hidden2), dim=-1)).squeeze(-1)
        sup_loss, _ = self.sup_loss(sim1, sim2, self.norm_params)

        if self.w_un > 0:
            un_hidden_q = F.leaky_relu(self.proj(get_disen_q_net_hidden(text_q, length_q, self.disen_q_net, use_vi)))
            un_hidden1 = F.leaky_relu(self.proj(get_disen_q_net_hidden(text_p, length_p, self.disen_q_net, use_vi)))
            un_hidden2 = F.leaky_relu(self.proj(get_disen_q_net_hidden(text_n, length_n, self.disen_q_net, use_vi)))
            un_loss_q = self.un_loss(sup_hidden_q, un_hidden_q)
            un_loss1 = self.un_loss(sup_hidden1, un_hidden1)
            un_loss2 = self.un_loss(sup_hidden2, un_hidden2)
            un_loss = (un_loss_q + un_loss1 + un_loss2) / 3
            loss = sup_loss * self.w_sup + un_loss * self.w_un
        else:
            loss = sup_loss * self.w_sup
        return loss, sup_loss
    
    def train(self, train_data, test_data, device, epoch, lr, step_size, gamma, silent, use_vi, n_predicts):
        """
            Similarity ranking model train
            :param train_data: iterable, paired train dataset, contains (text_q, length_q, text_p, length_p, text_n, length_n), similarity between (text_q, text_p) is larger than (text_q, text_n)
                text_q: Tensor of (batch_size, seq_len)
                length_q: Tensor of (batch_size)
                text_p: Tensor of (batch_size, seq_len)
                length_p: Tensor of (batch_size)
                text_n: Tensor of (batch_size, seq_len)
                length_n: Tensor of (batch_size)
            :param test_data: iterable, test dataset for ranking
            :param device: str, cpu or cuda
            :param epoch: int, number of epoch
            :param lr: float, initial learning rate
            :param step_size: int, step_size for StepLR, period of learning rate decay
            :param gamma: float, gamma for StepLR, multiplicative factor of learning rate decay
            :param silent: bool, whether to log loss
            :param use_vi: bool, whether to use vi or vk hidden of DisenQNet for training
            :param n_predicts: int, number of top-k predictions as positive items in ranking evaluation for testing
        """
        # optimizer & scheduler
        model_params = list(self.task_encoder.parameters()) + list(self.estimator.parameters()) + list(self.proj.parameters()) + list(self.un_loss.parameters())
        optimizer = Adam(model_params, lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        # train
        for epoch_idx in range(epoch):
            epoch_idx += 1

            self.to(device)
            self.set_mode(True)
            train_data.dataset.sample()
            for data in train_data:
                loss, sup_loss = self.compute_loss(data, device, use_vi)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # test
            loss, sup_loss = self.eval_loss(train_data, device, use_vi)
            if test_data is not None:
                map, ndcg, f1 = self.eval_rank(test_data, device, n_predicts)
                if not silent:
                    logging.info(f"[Epoch {epoch_idx:2d}] train loss: {loss:.4f}, sup_loss: {sup_loss:.4f}, eval map: {map:.3f}, ndcg: {ndcg:.3f}, f1: {f1:.3f}")
            elif not silent:
                logging.info(f"[Epoch {epoch_idx:2d}] train loss: {loss:.4f}, sup_loss: {sup_loss:.4f}")
        return
    
    def eval_rank(self, test_data, device, n_predicts):
        """
            Similarity ranking model test
            :param test_data: iterable, test dataset for ranking, contains (text_q, length_q, texts, lengths, target) as a question list to rank similarity between text_q and each text in texts
                text_q: Tensor of (1, seq_len)
                length_q: Tensor of (1)
                texts: Tensor of (list_size, seq_len)
                lengths: Tensor of (list_size)
                target: list of int, indexes of positive questions
            :param device: str, cpu or cuda
            :param n_predicts: int, number of top-k predictions as positive items in ranking evaluation
            :returns: (map, ndcg, f1)
        """
        self.to(device)
        self.set_mode(False)
        total_size = 0
        total_map = 0
        total_ndcg = 0
        total_f1 = 0
        with torch.no_grad():
            for data in test_data:
                text_q, length_q, texts, lengths, target = data
                text_q = text_q.to(device)
                texts = texts.to(device)
                _, hidden_q = self.task_encoder(text_q)
                _, hiddens = self.task_encoder(texts)
                sims = self.estimator(torch.cat((hidden_q.expand(hiddens.size(0), -1), hiddens), dim=-1)).squeeze(-1)
                map = mAP_similarity(target, sims, n_predicts)
                ndcg = nDCG_similarity(target, sims, n_predicts)
                f1 = f1_similarity(target, sims, n_predicts)
                total_size += 1
                total_map += map
                total_ndcg += ndcg
                total_f1 += f1
        map = total_map / total_size
        ndcg = total_ndcg / total_size
        f1 = total_f1 / total_size
        return map, ndcg, f1
    
    def eval_loss(self, test_data, device, use_vi):
        """
            Similarity ranking model test
            :param test_data: iterable, paired test dataset, contains (text_q, length_q, text_p, length_p, text_n, length_n), similarity between (text_q, text_p) is larger than (text_q, text_n)
                text_q: Tensor of (batch_size, seq_len)
                length_q: Tensor of (batch_size)
                text_p: Tensor of (batch_size, seq_len)
                length_p: Tensor of (batch_size)
                text_n: Tensor of (batch_size, seq_len)
                length_n: Tensor of (batch_size)
            :param device: str, cpu or cuda
            :param use_vi: bool, whether to use vi or vk hidden of DisenQNet for training
            :returns: (loss, sup_loss)
        """
        self.to(device)
        self.set_mode(False)
        total_size = 0
        total_loss = 0
        total_sup_loss = 0
        with torch.no_grad():
            for data in test_data:
                loss, sup_loss = self.compute_loss(data, device, use_vi)
                batch_size = data[0].size(0)
                loss, sup_loss = loss.item(), sup_loss.item()
                total_size += batch_size
                total_loss += loss * batch_size
                total_sup_loss += sup_loss * batch_size
        loss = total_loss / total_size
        sup_loss = total_sup_loss / total_size
        return loss, sup_loss
    
    def save(self, filepath):
        state_dicts = [module.state_dict() for module in self.modules]
        torch.save(state_dicts, filepath)
        return
    
    def load(self, filepath):
        state_dicts = torch.load(filepath)
        for module, state_dict in zip(self.modules, state_dicts):
            module.load_state_dict(state_dict)
        return
    
    def to(self, device):
        for module in self.modules:
            module.to(device)
        return
    
    def set_mode(self, train):
        self.disen_q_net.eval()
        for module in self.modules:
            if module is not self.disen_q_net:
                if train:
                    module.train()
                else:
                    module.eval()
        return
