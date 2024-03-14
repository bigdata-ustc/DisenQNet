# -*- coding: utf-8 -*-

import json
import logging
import os
import numpy as np
import torch

num_token = "<num>"
unk_token = "<unk>"
pad_token = "<pad>"

def load_list(path):
    with open(path, "rt", encoding="utf-8") as file:
        items = file.read().strip().split('\n')
    item2index = {item:index for index, item in enumerate(items)}
    return item2index

def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return

def check_num(s):
    # (1/2) -> 1/2
    if s.startswith('(') and s.endswith(')'):
        if s == "()":
            return False
        else:
            s = s[1:-1]
    # -1/2 -> 1/2
    if s.startswith('-') or s.startswith('+'):
        if s == '-' or s == '+':
            return False
        else:
            s = s[1:]
    # 1/2
    if '/' in s:
        if s == '/':
            return False
        else:
            sep_index = s.index('/')
            is_num = s[:sep_index].isdigit() and s[sep_index+1:].isdigit()
            return is_num
    else:
        # 1.2% -> 1.2
        if s.endswith('%'):
            if s == '%':
                return False
            else:
                s = s[:-1]
        # 1.2
        if '.' in s:
            if s == '.':
                return False
            else:
                try:
                    float(s)
                    return True
                except:
                    return False
        # 12
        else:
            is_num = s.isdigit()
            return is_num

def list_to_onehot(item_list, item2index):
    onehot = np.zeros(len(item2index)).astype(np.int64)
    for c in item_list:
        onehot[item2index[c]] = 1
    return onehot

def get_data_dict(dataset):
    data_dict = {item["question_id"]:item for item in dataset}
    return data_dict

def pad_texts(texts, lengths, pad_idx):
    max_length = max(lengths)
    texts = [text+[pad_idx]*(max_length-len(text)) for text in texts]
    return texts

def read_questions(data_path, wv_path, word_list_path, concept_list_path, max_length, init_wv=True, embed_dim=None, trim_min_count=None, silent=False):
    # read text
    dataset = list()
    stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
    with open(data_path, "rt", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            text = data["content"].strip().split(' ')
            text = [w for w in text if w != '' and w not in stop_words]
            if len(text) == 0:
                text = [unk_token]
            if len(text) > max_length:
                text = text[:max_length]
            text = [num_token if check_num(w) else w for w in text]
            data["content"] = text
            data["length"] = len(text)
            dataset.append(data)
    
    # load or construct word, concept list
    if os.path.exists(wv_path):
        # load word, concept list
        word2index = load_list(word_list_path)
        concept2index = load_list(concept_list_path)
        word2vec = torch.load(wv_path)
        # if not silent:
        #     logging.info(f"load vocab from {word_list_path}")
        #     logging.info(f"load concept from {concept_list_path}")
        #     logging.info(f"load word2vec from {wv_path}")
    elif init_wv:
        # construct word and concept list
        # word count
        word2cnt = dict()
        concepts = set()
        for data in dataset:
            text = data["content"]
            concept = data["knowledge"]
            for w in text:
                word2cnt[w] = word2cnt.get(w, 0) + 1
            for c in concept:
                if c not in concepts:
                    concepts.add(c)
        
        # word & concept list
        ctrl_tokens = [num_token, unk_token, pad_token]
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in ctrl_tokens]
        keep_word_cnts = sum(word2cnt[w] for w in words)
        all_word_cnts = sum(word2cnt.values())
        if not silent:
            logging.info(f"save words({trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words)/len(word2cnt):.4f} with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts/all_word_cnts:.4f}")
        concepts = sorted(concepts)
        words = ctrl_tokens + sorted(words)

        # word2vec
        from gensim.models import Word2Vec
        corpus = list()
        word_set = set(words)
        for data in dataset:
            text = [w if w in word_set else unk_token for w in data["content"]]
            corpus.append(text)
        wv = Word2Vec(corpus, size=embed_dim, min_count=trim_min_count).wv
        wv_list = [wv[w] if w in wv.vocab.keys() else np.random.rand(embed_dim) for w in words]
        # wv = Word2Vec(corpus, vector_size=embed_dim, min_count=trim_min_count).wv
        # wv_list = [wv[w] if w in wv.key_to_index.keys() else np.random.rand(embed_dim) for w in words]
        word2vec = torch.tensor(wv_list)
        concept2index = {concept:index for index, concept in enumerate(concepts)}
        word2index = {word:index for index, word in enumerate(words)}

        # save word, concept list
        save_list(word2index, word_list_path)
        save_list(concept2index, concept_list_path)
        torch.save(word2vec, wv_path)
        if not silent:
            logging.info(f"save vocab to {word_list_path}")
            logging.info(f"save concept to {concept_list_path}")
            logging.info(f"save word2vec to {wv_path}")
    
    # map word to index
    unk_idx = word2index[unk_token]
    for data in dataset:
        data["content"] = [word2index.get(w, unk_idx) for w in data["content"]]
        data["knowledge"] = list_to_onehot(data["knowledge"], concept2index)
    return dataset, word2index, concept2index, word2vec

def get_equal_label_indexes(sorted_labels):
    # [a, a, a, b, b, c, c, c] => [(0, 1, 2), (3, 4), (5, 6, 7)]
    equal_label_indexes = []
    idx_group = []
    group_label = None
    for idx, label in enumerate(sorted_labels):
        if len(idx_group) == 0:
            idx_group.append(idx)
            group_label = label
        elif label == group_label:
            idx_group.append(idx)
        else:
            equal_label_indexes.append(idx_group)
            idx_group = []
            idx_group.append(idx)
            group_label = label
    equal_label_indexes.append(idx_group)
    return equal_label_indexes

def read_difficulty_rank_list(path):
    rank_list = []
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            # diff1 >= diff2 >= diff3 >= diff4
            qid_list = data["question_list"]
            diff_list = data["difficulty_list"]
            target_idx_list = get_equal_label_indexes(diff_list)
            rank_list.append((qid_list, diff_list, target_idx_list))
    return rank_list

def read_similarity_rank_list(path):
    rank_list = []
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            qid = data["query"]
            cid_list = data["candidates"]
            target = data["target"]
            cid_idx_dict = {cid:idx for idx, cid in enumerate(cid_list)}
            target_idx_list = [cid_idx_dict[cid] for cid in target]
            rank_list.append((qid, cid_list, target_idx_list))
    return rank_list
