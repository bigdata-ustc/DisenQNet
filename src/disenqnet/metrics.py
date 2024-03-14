# -*- coding: utf-8 -*-

import torch

def get_topk_class(probability, top_k):
    """
        select top k class according probability for multi-label classification
        :param probability: Tensor of (batch_size, class_size), probability
        :param topk: int, number of selected classes
        :returns: Tensor of long (batch_size, class_size), whether class is selected, 1 means class selected, 0 means class unselected
    """
    # batch_size * class_size
    device = probability.device
    batch_size, class_size = probability.size()
    # batch_size * k
    label_index = probability.topk(top_k, dim=-1, sorted=False)[1]
    batch_index = torch.arange(batch_size).unsqueeze(-1).expand(-1, top_k)
    # batch_size * class_size
    label = torch.zeros(batch_size, class_size).long()
    batch_index = batch_index.to(device)
    label = label.to(device)
    label[batch_index, label_index] = 1
    return label

def get_confuse_matrix(label, probability, top_k):
    """
        Confuse matrix for multi-label classification by true label and prediction according top k probability
        :param label: Tensor of (batch_size, class_size), multi-label classification label, 1 means True, 0 means False
        :param probability: Tensor of (batch_size, class_size), predicted probability
        :param top_k: int, number of top k classes as positive label for multi-label classification
        :returns: Tensor of long (3, class_size), (TP, FP, FN) count for each class
    """
    prediction = get_topk_class(probability, top_k).bool()
    label = label.bool()
    # batch_size * class_size -> class_size
    tp = (label & prediction).sum(dim=0)
    fp = ((~label) & prediction).sum(dim=0)
    fn = (label & (~prediction)).sum(dim=0)
    # 3 * class_size
    cm = torch.stack((tp, fp, fn), dim=0)
    return cm

def get_f1_score(confuse_matrix, reduction="micro"):
    """
        F1 score for multi-label classification
        Follow https://www.cnblogs.com/fledlingbird/p/10675922.html
        :param confuse_matrix: Tensor of long (3, class_size), (TP, FP, FN) count for each class
        :param reduction: str, macro or micro, reduction type for F1 score
        :returns: float, f1 score for multi-label classification
    """
    # 3 * class_size -> class_size
    tp, fp, fn = confuse_matrix
    if reduction == "macro":
        precise = tp.float() / (tp + fp).float()
        recall = tp.float() / (tp + fn).float()
        f1 = (2 * precise * recall) / (precise + recall)
        f1[tp == 0] = 0
        f1 = f1.mean().item()
    elif reduction == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        precise = tp.float() / (tp + fp).float()
        recall = tp.float() / (tp + fn).float()
        f1 = ((2 * precise * recall) / (precise + recall)).item()
        if tp.item() == 0:
            f1 = 0
    return f1

def DOA(targets, predicts):
    """
        Degree of agreement for difficulty estimation
        :param targets: Tensor of float (size), target difficulty scores
        :param predicts: Tensor of float (size), predicted difficulty scores
        :returns: float, DOA score for difficulty estimation
    """
    targets, sorted_idx = targets.sort(dim=-1, descending=True)
    predicts = predicts.index_select(dim=-1, index=sorted_idx)
    # discard repeated and same-score pairs from all pairs
    all_idx = targets.unsqueeze(0) < targets.unsqueeze(-1)
    # pairs with consistent order to targets
    positive = predicts.unsqueeze(0) < predicts.unsqueeze(-1)
    doa = positive[all_idx].sum().item() / all_idx.sum().item()
    return doa

def mAP(hits):
    """
        mean Average Precision for ranking
        :param hits: Tensor of bool (size), whether each item in the top-n predicted rank hits in the top-n targets
        :returns: float, mAP score for predicted ranking
    """
    if hits.sum().item() == 0:
        ap = 0
    else:
        hit_pos = torch.arange(hits.size(-1))[hits.bool()] + 1
        hit_num = torch.arange(hit_pos.size(-1)) + 1
        ap = ((hit_num.float() / hit_pos.float()).sum() / hits.size(-1)).item()
    return ap

def nDCG(scores):
    """
        normalized Discounted Cumulative Gain for ranking
        :param scores: Tensor of float (size), ground-truth score of each item in the predicted rank
        :returns: float, nDCG score for predicted ranking
    """
    if scores.sum().item() == 0:
        ndcg = 0
    else:
        idx = torch.arange(scores.size(-1)).float() + 1
        dcg = (scores / torch.log2(idx + 1)).sum()
        iscores = scores.sort(dim=-1, descending=True)[0]
        idcg = (iscores / torch.log2(idx + 1)).sum()
        ndcg = (dcg / idcg).item()
    return ndcg

def f1_score(hits, n_targets, n_predicts):
    """
        f1 score for top-k prediction in top-n targets
        :param hits: Tensor of bool (size), whether each item in the top-k prediction hits in the top-n targets
        :param n_targets: int, number of top-n targets
        :param n_predicts: int, number of top-k prediction
        :returns: float, f1 score
    """
    tp = hits.sum().item()
    precise = tp / n_predicts
    recall = tp / n_targets
    if tp == 0:
        f1 = 0
    else:
        f1 = (2 * precise * recall) / (precise + recall)
    return f1

def get_duplicate_topk(target_list, top_k):
    # [(0, 1, 2), (3, 4), (5, 6, 7)] (top 4)=> (0, 1, 2, 3, 4)
    top_list = []
    if top_k > 0:
        for equal_list in target_list:
            top_list.extend(equal_list)
            if len(top_list) >= top_k:
                break
    return top_list

def get_topk_difficulty(target_list, predict_scores, n_targets, n_predicts, return_score=False, true_scores=None):
    """
        get top-k hits or true scores in predicted ranking, take top-n targets as positive label, for difficulty estimation
        :param target_list: list of list of int, item_indexes in target rank, each rank has several items in a list with same score
        :param predict_scores: Tensor of float (size), predicted score of each item in item_index order
        :param n_targets: int, number of top-n targets
        :param n_predicts: int, number of top-k predicts
        :param return_score: bool, whether return scores (float) or hits (bool)
        :param true_scores: list of float, ground-truth score of each item in item_index order
        :returns: Tensor of bool or float (n_predicts), hit or true score of each item in the top-k predicted rank
    """
    # [a, a, a, b, b, c, c, c] => [(0, 1, 2), (3, 4), (5, 6, 7)] (top 4)=> (0, 1, 2, 3, 4)
    positive_items = set(get_duplicate_topk(target_list, n_targets))
    predicted_rank = predict_scores.sort(dim=-1, descending=True)[1].cpu().tolist()[:n_predicts]
    if return_score:
        metrics = torch.tensor([true_scores[item_idx] if item_idx in positive_items else 0 for item_idx in predicted_rank]).float()
    else:
        metrics = torch.tensor([1 if item_idx in positive_items else 0 for item_idx in predicted_rank])
    return metrics

def get_topk_similarity(target_list, predict_scores, n_predicts):
    """
        get top-k hits in predicted ranking given target positive items, for similarity estimation
        :param target_list: list of int, item_indexes of positive items
        :param predict_scores: Tensor of float (size), predicted score of each item in item_index order
        :param n_predicts: int, number of top-k predicts
        :returns: Tensor of bool (n_predicts), whether each item in the top-k predicted rank hits in the target_list
    """
    predicted_rank = predict_scores.sort(dim=-1, descending=True)[1].cpu().tolist()[:n_predicts]
    hits = torch.tensor([1 if item_idx in target_list else 0 for item_idx in predicted_rank])
    return hits

def mAP_difficulty(target_list, predict_scores, n_targets, n_predicts):
    return mAP(get_topk_difficulty(target_list, predict_scores, n_targets, n_predicts))

def nDCG_difficulty(target_list, predict_scores, true_scores, n_targets, n_predicts):
    return nDCG(get_topk_difficulty(target_list, predict_scores, n_targets, n_predicts, return_score=True, true_scores=true_scores))

def mAP_similarity(target_list, predict_scores, n_predicts):
    return mAP(get_topk_similarity(target_list, predict_scores, n_predicts))

def nDCG_similarity(target_list, predict_scores, n_predicts):
    return nDCG(get_topk_similarity(target_list, predict_scores, n_predicts))

def f1_similarity(target_list, predict_scores, n_predicts):
    return f1_score(get_topk_similarity(target_list, predict_scores, n_predicts), len(target_list), n_predicts)
