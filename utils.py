import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
from NLP_datasets import MOONTextDataset
from model import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed):
    """
    設置隨機種子以確保實驗可重現性
    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_moon_datasets(dataset_name, n_clients, alpha=0.5, max_length=512):
    """
    創建 MOON 數據集
    Args:
        dataset_name: 數據集名稱
        n_clients: 客戶端數量
        alpha: Dirichlet 分布參數
        max_length: 文本最大長度
    Returns:
        train_dataset: 訓練數據集
        test_dataset: 測試數據集
    """
    # 創建測試集
    test_dataset = MOONTextDataset(dataset_name, split='test', max_length=max_length)
    
    # 創建訓練集（包含所有客戶端的數據）
    train_dataset = MOONTextDataset(
        dataset_name, 
        split='train', 
        alpha=alpha,
        n_clients=n_clients,
        max_length=max_length
    )
    
    return train_dataset, test_dataset

def collate_fn(batch):
    """
    將批次中的資料填充到相同長度
    Args:
        batch: 批次資料，包含 (sequence, label) 元組
    Returns:
        padded_texts: 填充後的文字張量
        labels: 標籤張量
    """
    x = [torch.tensor(x) for x, y in batch]
    y = torch.tensor([y for x, y in batch])
    x_tensor = pad_sequence(x, batch_first=True)
    return x_tensor, y

def create_moon_dataloaders(train_dataset, test_dataset, batch_size=32):
    """
    為 MOON 數據集創建 DataLoader
    Args:
        train_dataset: 訓練數據集
        test_dataset: 測試數據集
        batch_size: 批次大小
    Returns:
        client_dataloaders: 客戶端 DataLoader 字典 {client_id: dataloader}
        test_dataloader: 測試集 DataLoader
    """
    # 創建測試集 DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 為每個客戶端創建 DataLoader
    client_dataloaders = {}
    for client_id in range(train_dataset.n_clients):
        client_indices = train_dataset.get_client_data(client_id)
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
        
        client_dataloader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(torch.initial_seed())
        )
        client_dataloaders[client_id] = client_dataloader
    
    return client_dataloaders, test_dataloader
    
