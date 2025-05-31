import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random

from model import *
from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    
    # 模型架構相關參數
    parser.add_argument('--model_name', type=str, default='LSTM', choices=['LSTM', 'Transformer'], help='model used in training')
    parser.add_argument('--n_layers', type=int, default=3, help='the number of layers for the model')
    parser.add_argument('--embedding_dim', type=int, default=300, 
                      help='the dimension of the embedding layer (must match GloVe dimensions when use_pretrained_embeddings=True)')
    parser.add_argument('--hidden_dim', type=int, default=300, help='the dimension of the hidden layer')
    parser.add_argument('--dropout', type=float, default=0.3, help='the dropout rate for the model')
    parser.add_argument('--projection_dim', type=int, default=256, help='the dimension of the projection layer')
    parser.add_argument('--n_heads', type=int, default=4, help='the number of attention heads')
    parser.add_argument('--use_pretrained_embeddings', type=bool, default=True, help='whether to use the pretrained embeddings')
    parser.add_argument('--freeze_embeddings', type=bool, default=False, help='whether to freeze the pretrained embeddings')

    # 數據集相關參數
    parser.add_argument('--dataset', type=str, default='imdb', 
                       choices=['imdb', 'ag_news', 'dbpedia_14', 'sst2', '20newsgroups', 'trec', 'yelp_review'], 
                       help='dataset used in training')
    parser.add_argument('--max_length', type=int, default=350, help="max length of text")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--vocab_size', type=int, default=30000, help='The size of the vocabulary')
    
    # 訓練相關參數
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')

    # 聯邦學習相關參數
    parser.add_argument('--comm_round', type=int, default=30, help='number of maximum communication rounds')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='moon', 
                       choices=['fedavg', 'fedprox', 'moon'], 
                       help='communication strategy: fedavg/fedprox/moon')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    
    # 其他參數
    parser.add_argument('--save_path', type=str, help='the path to save the results')

    args = parser.parse_args()
    
    # 驗證參數
    if args.use_pretrained_embeddings and args.embedding_dim not in [50, 100, 200, 300]:
        raise ValueError("When using pretrained embeddings, embedding_dim must be one of [50, 100, 200, 300]")
    
    return args

def init_nets(n_parties, args, device='cpu'):
    # 設定分類數量
    if args.dataset == 'imdb':
        n_classes = 2
    elif args.dataset == 'ag_news':
        n_classes = 4
    elif args.dataset == 'dbpedia_14':
        n_classes = 14
    elif args.dataset == 'sst2':
        n_classes = 2
    elif args.dataset == '20newsgroups':
        n_classes = 20
    elif args.dataset == 'trec':
        n_classes = 6
    elif args.dataset == 'yelp_review':
        n_classes = 5
    else:
        raise ValueError(f"不支持的數據集: {args.dataset}")
    
    args.n_classes = n_classes
    
    # 初始化網絡
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.alg == 'moon':
            nets[net_i] = MOONModel(args=args)
        elif args.alg == 'fedavg':
            nets[net_i] = FedAvgModel(args=args)
        elif args.alg == 'fedprox':
            nets[net_i] = FedProxModel(args=args)
            
        # 將模型移動到指定設備
        nets[net_i] = nets[net_i].to(device)

    return nets

def local_train_net():
    pass

def train_fedprox():
    pass

def train_moon():
    pass

def train_fedavg():
    pass




if __name__ == '__main__':
    args = get_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    print(f'正在建立資料集: {args.dataset}')

    # 創建數據集
    train_dataset, test_dataset = create_moon_datasets(
        dataset_name=args.dataset.lower(),
        n_clients=args.n_parties,
        beta=args.beta,
        max_length=args.max_length,
        vocab_size=args.vocab_size
    )
    
    # 創建 DataLoader
    client_dataloaders, test_dataloader = create_moon_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size
    )
    
    # 設置設備
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 獲取預訓練詞嵌入
    if args.use_pretrained_embeddings:
        pretrained_embeddings = load_glove_embeddings(
            vocab=train_dataset.vocab,
            embedding_dim=args.embedding_dim
        )
        args.pretrained_embeddings = pretrained_embeddings
    else:
        args.pretrained_embeddings = None
    
    # 初始化客戶端網絡
    clients_nets = init_nets(n_parties=args.n_parties, args=args, device=device)

    # 初始化服務器網絡
    global_nets = init_nets(n_parties=1, args=args, device=device)
    global_model = global_nets[0]

    
    
    
