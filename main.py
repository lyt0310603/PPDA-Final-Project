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
    
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'Transformer'], help='model used in training')
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'ag_news', 'dbpedia_14', 'sst2', '20newsgroups', 'trec', 'yelp_review'], help='dataset used in training')
    parser.add_argument('--max_length', type=int, default=350, help="max length of text")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='moon', choices=['fedavg', 'fedprox', 'moon'], help='communication strategy: fedavg/fedprox/moon')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--save_path', type=str, help='the path to save the results')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建數據集
    train_dataset, test_dataset = create_moon_datasets(
        dataset_name=args.dataset.lower(),
        n_clients=args.n_parties,
        alpha=args.beta,
        max_length=args.max_length
    )
    
    # 創建 DataLoader
    client_dataloaders, test_dataloader = create_moon_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size
    )
    
    # 設置設備
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
