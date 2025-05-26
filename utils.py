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


def create_moon_datasets(dataset_name, n_clients, alpha=0.5):
    """
    創建 MOON 數據集
    Args:
        dataset_name: 數據集名稱
        n_clients: 客戶端數量
        alpha: Dirichlet 分布參數
    Returns:
        train_dataset: 訓練數據集
        test_dataset: 測試數據集
    """
    # 創建測試集
    test_dataset = MOONTextDataset(dataset_name, split='test')
    
    # 創建訓練集（包含所有客戶端的數據）
    train_dataset = MOONTextDataset(
        dataset_name, 
        split='train', 
        alpha=alpha,
        n_clients=n_clients
    )
    
    return train_dataset, test_dataset

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
        num_workers=4
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
            num_workers=4
        )
        client_dataloaders[client_id] = client_dataloader
    
    return client_dataloaders, test_dataloader
    
