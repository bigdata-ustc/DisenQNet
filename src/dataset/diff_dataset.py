# -*- coding: utf-8 -*-

import logging
import torch
from torch.utils.data import Dataset

from .utils import pad_token, read_questions, read_difficulty_rank_list, get_data_dict, pad_texts
from .sampling import get_difficulty_candidates, sample_difficulty_pairwise

class DiffTrainDataset(Dataset):
    """
        Pairwise dataset for difficulty estimation training, including Tensor (text1, length1, text2, length2)
    """
    def __init__(self, dataset_path, wv_path, word_list_path, concept_list_path, max_length, num_samples, silent):
        super(DiffTrainDataset, self).__init__()
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
        self.candidates = get_difficulty_candidates(self.data_dict, 0, 1)
        self.train_samples = None
        self.sample()
        return
    
    def __getitem__(self, index):
        qid1, qid2 = self.train_samples[index]
        data1, data2 = self.data_dict[qid1], self.data_dict[qid2]
        text1, length1 = data1["content"], data1["length"]
        text2, length2 = data2["content"], data2["length"]
        return text1, length1, text2, length2
    
    def __len__(self):
        return len(self.train_samples)
    
    def collate_data(self, batch_data):
        text1, length1, text2, length2 = list(zip(*batch_data))
        text1 = pad_texts(text1, length1, self.pad_idx)
        text2 = pad_texts(text2, length2, self.pad_idx)
        text1, text2 = torch.tensor(text1), torch.tensor(text2)
        length1, length2 = torch.tensor(length1), torch.tensor(length2)
        return text1, length2, text2, length2
    
    def sample(self):
        self.train_samples = sample_difficulty_pairwise(self.candidates, self.data_dict, self.num_samples)
        return

class DiffRankDataset(Dataset):
    """
        List-wise dataset for difficulty ranking evaluation, including (texts, lengths, difficulty list, target rank)
    """
    def __init__(self, dataset_path, rank_path, wv_path, word_list_path, concept_list_path, max_length, silent):
        super(DiffRankDataset, self).__init__()
        # load dataset, word and concept list
        dataset, word2index, concept2index, word2vec = read_questions(dataset_path, wv_path, word_list_path, concept_list_path, max_length, False, silent=silent)
        rank_list = read_difficulty_rank_list(rank_path)
        
        self.data_dict = get_data_dict(dataset)
        self.rank_list = rank_list
        self.pad_idx = word2index[pad_token]
        if not silent:
            logging.info(f"load dataset from {dataset_path}")
        return
    
    def __getitem__(self, index):
        qid_list, diff_list, target = self.rank_list[index]
        batch_data = []
        for qid in qid_list:
            data = self.data_dict[qid]
            text, length = data["content"], data["length"]
            batch_data.append((text, length))
        texts, lengths = list(zip(*batch_data))
        texts = pad_texts(texts, lengths, self.pad_idx)
        texts, lengths = torch.tensor(texts), torch.tensor(lengths)
        return texts, lengths, diff_list, target
    
    def __len__(self):
        return len(self.rank_list)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
        return
