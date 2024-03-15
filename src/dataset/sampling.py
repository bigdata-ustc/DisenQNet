# -*- coding: utf-8 -*-

import random

def get_difficulty_candidates(data_dict, min_dist=0, max_dist=1):
    qid2candidates = dict()
    for qid1, data1 in data_dict.items():
        diff1 = data1["difficulty"]
        # 0 < min_dist <= |diff1-diff2| <= max_dist
        qid2dist = dict()
        for qid2, data2 in data_dict.items():
            if qid2 != qid1:
                dist = abs(diff1 - data2["difficulty"])
                if dist > 0 and min_dist <= dist <= max_dist:
                    qid2dist[qid2] = dist
        candidates = sorted(qid2dist.keys(), key=lambda q: qid2dist[q])
        qid2candidates[qid1] = candidates
    return qid2candidates

def sample_difficulty_pairwise(question_candidates, data_dict, num_samples):
    train_samples = []
    for qid1, candidates in question_candidates.items():
        q_num_samples = min(num_samples, len(candidates))
        samples = random.sample(candidates, q_num_samples)
        for qid2 in samples:
            # diff1 > diff2
            if data_dict[qid1]["difficulty"] > data_dict[qid2]["difficulty"]:
                train_samples.append((qid1, qid2))
            else:
                train_samples.append((qid2, qid1))
    random.shuffle(train_samples)
    return train_samples

def get_similarity_candidates(data_dict):
    qid2candidates = dict()
    for qid1, data1 in data_dict.items():
        positives = data1["similar_questions"]
        negatives = []
        for qid2 in data_dict.keys():
            if qid2 != qid1 and qid2 not in positives:
                negatives.append(qid2)
        positives = sorted(positives)
        negatives = sorted(negatives)
        qid2candidates[qid1] = (positives, negatives)
    return qid2candidates

def sample_similarity_pairwise(question_candidates, num_samples):
    train_samples = []
    for qid1, (positives, negatives) in question_candidates.items():
        for pos_sample in positives:
            q_num_samples = min(num_samples, len(negatives))
            for _ in range(q_num_samples):
                neg_sample = random.choice(negatives)
                train_samples.append((qid1, pos_sample, neg_sample))
    random.shuffle(train_samples)
    return train_samples
