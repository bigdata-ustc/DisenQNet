# -*- coding: utf-8 -*-

import logging
import torch
from torch.utils.data import Dataset

from .utils import pad_token, read_questions, read_similarity_rank_list, get_data_dict, pad_texts
from .sampling import get_similarity_candidates, sample_similarity_pairwise

class SimTrainDataset(Dataset):
    """
        Pairwise dataset for similarity estimation training, including Tensor (q_text, q_length, p_text, p_length, n_text, n_length)
    """
    def __init__(self, dataset_path, wv_path, word_list_path, concept_list_path, max_length, num_samples, silent):
        super(SimTrainDataset, self).__init__()
        # load dataset, word and concept list
        dataset, word2index, concept2index, word2vec = read_questions(dataset_path, wv_path, word_list_path, concept_list_path, max_length, False, silent=silent)
        
        self.data_dict = get_data_dict(dataset)
        self.pad_idx = word2index[pad_token]
        self.word2vec = word2vec
        self.vocab_size = len(word2index)
        self.concept_size = len(concept2index)
        if not silent:
            logging.info(f"load dataset from {dataset_path}")
            logging.info(f"vocab size: {self.vocab_size}")
            logging.info(f"concept size: {self.concept_size}")
        
        self.num_samples = num_samples
        self.candidates = get_similarity_candidates(self.data_dict)
        self.train_samples = None
        self.sample()
        return
    
    def __getitem__(self, index):
        q_id,  p_id, n_id = self.train_samples[index]
        q_data, p_data, n_data = self.data_dict[q_id], self.data_dict[p_id], self.data_dict[n_id]
        q_text, q_length = q_data["content"], q_data["length"]
        p_text, p_length = p_data["content"], p_data["length"]
        n_text, n_length = n_data["content"], n_data["length"]
        return q_text, q_length, p_text, p_length, n_text, n_length
    
    def __len__(self):
        return len(self.train_samples)
    
    def collate_data(self, batch_data):
        q_text, q_length, p_text, p_length, n_text, n_length = list(zip(*batch_data))
        q_text = pad_texts(q_text, q_length, self.pad_idx)
        p_text = pad_texts(p_text, p_length, self.pad_idx)
        n_text = pad_texts(n_text, n_length, self.pad_idx)
        q_text, p_text, n_text = torch.tensor(q_text), torch.tensor(p_text), torch.tensor(n_text)
        q_length, p_length, n_length = torch.tensor(q_length), torch.tensor(p_length), torch.tensor(n_length)
        return q_text, q_length, p_text, p_length, n_text, n_length
    
    def sample(self):
        self.train_samples = sample_similarity_pairwise(self.candidates, self.num_samples)
        return

class SimRankDataset(Dataset):
    """
        List-wise dataset for similarity ranking evaluation, including (q_text, q_length, c_texts, c_lengths, target rank)
    """
    def __init__(self, dataset_path, rank_path, wv_path, word_list_path, concept_list_path, max_length, silent):
        super(SimRankDataset, self).__init__()
        # load dataset, word and concept list
        dataset, word2index, concept2index, word2vec = read_questions(dataset_path, wv_path, word_list_path, concept_list_path, max_length, False, silent=silent)
        rank_list = read_similarity_rank_list(rank_path)
        
        self.data_dict = get_data_dict(dataset)
        self.rank_list = rank_list
        self.pad_idx = word2index[pad_token]
        if not silent:
            logging.info(f"load dataset from {dataset_path}")
        return
    
    def __getitem__(self, index):
        qid, cid_list, target = self.rank_list[index]
        q_data = self.data_dict[qid]
        q_text, q_length = q_data["content"], q_data["length"]
        q_text, q_length = torch.tensor([q_text]), torch.tensor([q_length])
        batch_data = []
        for cid in cid_list:
            c_data = self.data_dict[cid]
            c_text, c_length = c_data["content"], c_data["length"]
            batch_data.append((c_text, c_length))
        c_texts, c_lengths = list(zip(*batch_data))
        c_texts = pad_texts(c_texts, c_lengths, self.pad_idx)
        c_texts, c_lengths = torch.tensor(c_texts), torch.tensor(c_lengths)
        return q_text, q_length, c_texts, c_lengths, target
    
    def __len__(self):
        return len(self.rank_list)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
        return
