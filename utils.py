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
from collections import defaultdict
from tqdm import tqdm


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

def create_moon_datasets(dataset_name, n_clients, beta=0.5, max_length=512, vocab_size=30000):
    """
    創建 MOON 數據集
    Args:
        dataset_name: 數據集名稱
        n_clients: 客戶端數量
        beta: Dirichlet 分布參數
        max_length: 文本最大長度
        vocab_size: 詞彙表大小
    Returns:
        train_dataset: 訓練數據集
        test_dataset: 測試數據集
    """
    # 創建測試集
    test_dataset = MOONTextDataset(dataset_name, split='test', max_length=max_length, vocab_size=vocab_size)
    
    # 創建訓練集（包含所有客戶端的數據）
    train_dataset = MOONTextDataset(
        dataset_name, 
        split='train', 
        beta=beta,
        n_clients=n_clients,
        max_length=max_length,
        vocab_size=vocab_size
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
    
def load_glove_embeddings(vocab, embedding_dim=300):
    """
    載入 GloVe 預訓練詞嵌入
    
    參數:
        vocab (dict): 詞彙表，格式為 {word: index}
        embedding_dim (int): 詞嵌入維度，必須與 GloVe 文件維度相匹配 (50, 100, 200, 300)
    
    返回:
        torch.Tensor: 預訓練詞嵌入矩陣，形狀為 [vocab_size, embedding_dim]
    """
    if embedding_dim not in [50, 100, 200, 300]:
        raise ValueError("embedding_dim 必須是 50, 100, 200 或 300")
        
    print(f"正在載入 GloVe {embedding_dim}d 詞嵌入...")
    
    fname = f'glove.6B.{embedding_dim}d.txt'
    
    with open(fname, 'rt', encoding='utf8') as fi:
        full_content = fi.read().strip().split('\n')
    
    data = {}
    for i in tqdm(range(len(full_content)), total=len(full_content), desc='loading glove vocabs...'):
        i_word = full_content[i].split(' ')[0]
        if i_word not in vocab.keys():
            continue
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        data[i_word] = i_embeddings
    
    # 建立詞嵌入矩陣
    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(embedding_dim))
    
    print(f'在 GloVe {embedding_dim}d 中找到 {find} 個詞')
    return torch.stack(w, dim=0)

def compute_accuracy(model, dataloader, device):
    """計算模型在給定數據集上的準確率
    
    參數:
        model: 要評估的模型
        dataloader: 數據加載器
        device: 計算設備（'cpu' 或 'cuda'）
        
    返回:
        accuracy: 模型在數據集上的準確率
    """
    model.eval()  # 設置模型為評估模式
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不計算梯度
        for x, y in dataloader:
            # 將數據移到指定設備
            x = x.to(device)
            y = y.to(device)
            
            # 前向傳播
            if isinstance(model, MOONModel):
                outputs, _ = model(x)  # MOON 模型返回 (logits, projected)
            else:
                outputs = model(x)
            
            # 計算準確率
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

    
