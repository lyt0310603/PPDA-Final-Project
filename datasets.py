import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import itertools


class TextDataset(Dataset):
    def __init__(self, data_path, max_length=512):
        self.data_path = data_path
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.vocab = {}
        self.vocab_size = 0
        
    def create_vocab(self, vocab_size=30000):
        """
        建立詞彙表
        Args:
            vocab_size: 詞彙表大小預設為30000
        """
        # 將所有文字分割成詞並合併
        corpus = [text.split() for text in self.data]
        corpus = list(itertools.chain.from_iterable(corpus))
        
        # 統計詞頻
        count_words = Counter(corpus)
        print('總詞數:', len(count_words))
        
        # 按頻率排序
        sorted_words = count_words.most_common()
        
        # 確定實際詞彙表大小
        if vocab_size > len(sorted_words):
            v = len(sorted_words)
        else:
            v = vocab_size - 2  # 減去 <pad> 和 <unk>
            
        # 建立詞到索引的映射
        self.vocab = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        
        self.vocab_size = len(self.vocab)
        print('詞彙表大小:', self.vocab_size)
        return self.vocab
    
    def convert2id(self, text):
        """
        將文字轉換為 ID 序列
        Args:
            text: 輸入文字
        Returns:
            List[int]: ID 序列
        """
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        sequence = self.convert2id(text)
        return sequence, label


class SST2Dataset(TextDataset):
    def __init__(self, data_path, split='train'):
        super().__init__(data_path)
        dataset = load_dataset('glue', 'sst2', split=split)
        self.data = dataset['sentence']
        self.labels = dataset['label']
        self.create_vocab()

class IMDBDataset(TextDataset):
    def __init__(self, data_path, split='train'):
        super().__init__(data_path)
        dataset = load_dataset('imdb', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class AGNewsDataset(TextDataset):
    def __init__(self, data_path, split='train'):
        super().__init__(data_path)
        dataset = load_dataset('ag_news', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class DBPediaDataset(TextDataset):
    def __init__(self, data_path, split='train'):
        super().__init__(data_path)
        dataset = load_dataset('dbpedia_14', split=split)
        self.data = dataset['content']
        self.labels = dataset['label']
        self.create_vocab()

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
