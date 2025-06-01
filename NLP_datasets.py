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
    def __init__(self, max_length=512, vocab_size=30000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.data = []
        self.labels = []
        self.vocab = {}
        
    def _clean_and_truncate_data(self):
        """
        清理和截斷數據：
        1. 移除空白文本
        2. 截斷過長的文本
        """
        cleaned_data = []
        cleaned_labels = []
        
        for text, label in zip(self.data, self.labels):
            # 移除空白文本
            if not text or not text.strip():
                continue
                
            # 分割文本並截斷
            words = text.split()
            if len(words) > self.max_length:
                words = words[:self.max_length]
            
            # 重新組合文本
            cleaned_text = ' '.join(words)
            cleaned_data.append(cleaned_text)
            cleaned_labels.append(label)
        
        self.data = cleaned_data
        self.labels = cleaned_labels
        print(f'清理後的數據數量: {len(self.data)}')
        
    def create_vocab(self):
        """
        建立詞彙表
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
        if self.vocab_size > len(sorted_words):
            v = len(sorted_words)
        else:
            v = self.vocab_size - 2  # 減去 <pad> 和 <unk>
            
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
        label = int(self.labels[idx])  # 確保標籤是 Python int
        sequence = self.convert2id(text)
        return sequence, label


class IMDBDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('imdb', split=split)
        self.data = dataset['text']
        self.labels = np.array(dataset['label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class AGNewsDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('ag_news', split=split)
        self.data = dataset['text']
        self.labels = np.array(dataset['label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class DBPediaDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('dbpedia_14', split=split)
        self.data = dataset['content']
        self.labels = np.array(dataset['label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class SST2Dataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        split = 'validation' if split == 'test' else split
        dataset = load_dataset('glue', 'sst2', split=split)
        self.data = dataset['sentence']
        self.labels = np.array(dataset['label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class Newsgroups20Dataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('SetFit/20_newsgroups', split=split)
        self.data = dataset['text']
        self.labels = np.array(dataset['label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class TRECDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('trec', split=split, trust_remote_code=True)
        self.data = dataset['text']
        self.labels = np.array(dataset['coarse_label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class YelpReviewDataset(TextDataset):
    def __init__(self, split='train'):
        super().__init__()
        dataset = load_dataset('yelp_review_full', split=split)
        self.data = dataset['text']
        self.labels = np.array(dataset['label'], dtype=np.int64)
        self._clean_and_truncate_data()
        self.create_vocab()

class MOONTextDataset(TextDataset):
    def __init__(self, dataset_name, split='train', beta=0.5, n_clients=None, max_length=512, vocab_size=30000):
        """
        初始化 MOON 數據集
        Args:
            dataset_name: 數據集名稱 ('imdb', 'ag_news', 'dbpedia_14', 'sst2', '20newsgroups', 'trec', 'yelp_review')
            split: 數據集分割 ('train' 或 'test')
            beta: Dirichlet 分布的參數，控制數據分布的不平衡程度
            n_clients: 總客戶端數量
            max_length: 文本最大長度
            vocab_size: 詞彙表大小
        """
        super().__init__(max_length=max_length, vocab_size=vocab_size)
        self.beta = beta
        self.n_clients = n_clients
        
        # 根據數據集名稱加載對應的數據
        if dataset_name == 'imdb':
            dataset = IMDBDataset(split)
        elif dataset_name == 'ag_news':
            dataset = AGNewsDataset(split)
        elif dataset_name == 'dbpedia_14':
            dataset = DBPediaDataset(split)
        elif dataset_name == 'sst2':
            dataset = SST2Dataset(split)  # SST2Dataset 會自動處理 test -> validation 的映射
        elif dataset_name == '20newsgroups':
            dataset = Newsgroups20Dataset(split)
        elif dataset_name == 'trec':
            dataset = TRECDataset(split)
        elif dataset_name == 'yelp_review':
            dataset = YelpReviewDataset(split)
        else:
            raise ValueError(f"不支持的數據集: {dataset_name}")
            
        self.data = dataset.data
        self.labels = np.array(dataset.labels, dtype=np.int64)
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
        
        # 初始化客戶端數據索引字典
        client_indices = {i: [] for i in range(self.n_clients)}
        
        # 根據數據集特點設置最小樣本數要求
        total_samples = len(self.labels)
        avg_samples_per_client = total_samples / self.n_clients
        
        # 根據數據集大小和類別數動態調整最小樣本數
        if total_samples < 10000:  # 小型數據集（如 SST2）
            min_require_size = max(10, int(avg_samples_per_client * 0.1))
        elif total_samples < 50000:  # 中型數據集（如 IMDB）
            min_require_size = max(20, int(avg_samples_per_client * 0.05))
        else:  # 大型數據集（如 DBPedia）
            min_require_size = max(50, int(avg_samples_per_client * 0.02))
        
        # 根據類別數調整最小樣本數
        min_require_size = max(min_require_size, n_classes * 2)  # 確保每個類別至少有 2 個樣本
        
        min_size = 0
        
        while min_size < min_require_size:
            # 初始化每個客戶端的索引列表
            idx_batch = [[] for _ in range(self.n_clients)]
            
            # 對每個類別進行分配
            for k in unique_labels:
                # 獲取當前類別的所有樣本索引
                idx_k = np.where(self.labels == k)[0]
                np.random.shuffle(idx_k)
                
                # 生成 Dirichlet 分布
                proportions = np.random.dirichlet(np.repeat(self.beta, self.n_clients))
                
                # 確保每個客戶端的樣本數不超過平均值
                proportions = np.array([p * (len(idx_j) < len(self.labels) / self.n_clients) 
                                     for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                
                # 計算每個客戶端應該獲得的樣本數
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                # 分配樣本
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                
                # 更新最小樣本數
                min_size = min([len(idx_j) for idx_j in idx_batch])
        
        # 打亂每個客戶端的數據順序
        for j in range(self.n_clients):
            np.random.shuffle(idx_batch[j])
            client_indices[j] = np.array(idx_batch[j])
        
        return client_indices
    
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


