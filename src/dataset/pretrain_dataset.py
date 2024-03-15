# -*- coding: utf-8 -*-

import logging
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import pad_token, read_questions, pad_texts

class QuestionDataset(Dataset):
    """
        Question dataset including text, length, concept Tensors
    """
    def __init__(self, dataset_path, wv_path, word_list_path, concept_list_path, embed_dim, trim_min_count, max_length, dataset_type, silent):
        super(QuestionDataset, self).__init__()
        # load dataset, word and concept list
        if dataset_type == "train":
            dataset, word2index, concept2index, word2vec = read_questions(dataset_path, wv_path, word_list_path, concept_list_path, max_length, True, embed_dim, trim_min_count, silent=silent)
        else:
            dataset, word2index, concept2index, word2vec = read_questions(dataset_path, wv_path, word_list_path, concept_list_path, max_length, False, silent=silent)
        
        self.dataset = dataset
        self.pad_idx = word2index[pad_token]
        if not silent:
            logging.info(f"load dataset from {dataset_path}")
        if dataset_type == "train":
            self.word2vec = word2vec
            self.vocab_size = len(word2index)
            self.concept_size = len(concept2index)
            if not silent:
                logging.info(f"vocab size: {self.vocab_size}")
                logging.info(f"concept size: {self.concept_size}")
        return
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        text = data["content"]
        length = data["length"]
        concept = data["knowledge"]
        return text, length, concept
    
    def collate_data(self, batch_data):
        text, length, concept = list(zip(*batch_data))
        text = pad_texts(text, length, self.pad_idx)
        text = torch.tensor(text)
        length = torch.tensor(length)
        concept = torch.tensor(np.array(concept))
        return text, length, concept
