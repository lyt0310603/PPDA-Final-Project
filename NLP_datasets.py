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
    def __init__(self, max_length=512):
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


class IMDBDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('imdb', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class AGNewsDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('ag_news', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class DBPediaDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('dbpedia_14', split=split)
        self.data = dataset['content']
        self.labels = dataset['label']
        self.create_vocab()

class SST2Dataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('glue', 'sst2', split=split)
        self.data = dataset['sentence']
        self.labels = dataset['label']
        self.create_vocab()

class Newsgroups20Dataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('20newsgroups', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class TRECDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('trec', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class YelpReviewDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('yelp_review', split=split)
        self.data = dataset['text']
        self.labels = dataset['label']
        self.create_vocab()

class MOONTextDataset(TextDataset):
    def __init__(self, dataset_name, split='train', alpha=0.5, n_clients=None):
        """
        初始化 MOON 數據集
        Args:
            dataset_name: 數據集名稱 ('imdb', 'ag_news', 'dbpedia_14', 'sst2', '20newsgroups', 'trec', 'yelp_review')
            split: 數據集分割 ('train' 或 'test')
            alpha: Dirichlet 分布的參數，控制數據分布的不平衡程度
            n_clients: 總客戶端數量
        """
        super().__init__()
        self.alpha = alpha
        self.n_clients = n_clients
        
        # 根據數據集名稱加載對應的數據
        if dataset_name == 'imdb':
            dataset = IMDBDataset(split)
        elif dataset_name == 'ag_news':
            dataset = AGNewsDataset(split)
        elif dataset_name == 'dbpedia_14':
            dataset = DBPediaDataset(split)
        elif dataset_name == 'sst2':
            dataset = SST2Dataset(split)
        elif dataset_name == '20newsgroups':
            dataset = Newsgroups20Dataset(split)
        elif dataset_name == 'trec':
            dataset = TRECDataset(split)
        elif dataset_name == 'yelp_review':
            dataset = YelpReviewDataset(split)
        else:
            raise ValueError(f"不支持的數據集: {dataset_name}")
            
        self.data = dataset.data
        self.labels = dataset.labels
        self.vocab = dataset.vocab
        self.vocab_size = dataset.vocab_size
        
        # 如果是訓練集且指定了客戶端數量，則分配數據
        if split == 'train' and n_clients is not None:
            self.client_indices = self._assign_client_data()
        else:
            self.client_indices = None
        
    def _assign_client_data(self):
        """
        使用 Dirichlet 分布為所有客戶端分配數據
        Returns:
            Dict[int, np.ndarray]: 客戶端 ID 到數據索引的映射
        """
        # 獲取所有唯一的標籤
        unique_labels = np.unique(self.labels)
        n_classes = len(unique_labels)
        
        # 為每個類別生成 Dirichlet 分布
        label_distribution = np.random.dirichlet([self.alpha] * self.n_clients)
        
        # 初始化客戶端數據索引字典
        client_indices = {i: [] for i in range(self.n_clients)}
        
        # 為每個類別分配數據
        for label in unique_labels:
            # 獲取當前類別的所有樣本索引
            label_indices = np.where(self.labels == label)[0]
            n_samples = len(label_indices)
            
            # 為每個客戶端分配樣本
            for client_id in range(self.n_clients):
                n_samples_for_client = int(n_samples * label_distribution[client_id])
                if n_samples_for_client > 0:
                    selected_indices = np.random.choice(
                        label_indices, 
                        size=n_samples_for_client, 
                        replace=False
                    )
                    client_indices[client_id].extend(selected_indices)
        
        # 將列表轉換為 numpy 數組
        return {k: np.array(v) for k, v in client_indices.items()}
    
    def get_client_data(self, client_id):
        """
        獲取指定客戶端的數據索引
        Args:
            client_id: 客戶端 ID
        Returns:
            np.ndarray: 分配給該客戶端的數據索引
        """
        if self.client_indices is None:
            raise ValueError("此數據集未分配客戶端數據")
        if client_id not in self.client_indices:
            raise ValueError(f"客戶端 ID {client_id} 不存在")
        return self.client_indices[client_id]


