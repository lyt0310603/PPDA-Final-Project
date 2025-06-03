import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import random
import time
import copy
from datetime import datetime

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
                       choices=['imdb', 'ag_news', 'dbpedia_14', 'sst2', 'sst5', '20newsgroups', 'trec', 'yelp_review', 'financial_phrasebank'], 
                       help='dataset used in training (imdb, ag_news, dbpedia_14, sst2, sst5, 20newsgroups, trec, yelp_review, financial_phrasebank)')
    parser.add_argument('--max_length', type=int, default=350, help="max length of text")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--vocab_size', type=int, default=30000, help='The size of the vocabulary')
    
    # 訓練相關參數
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--seed', type=int, default=114514, help='random seed for reproducibility')
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
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the results')

    args = parser.parse_args()
    
    # 驗證參數
    if args.use_pretrained_embeddings and args.embedding_dim not in [50, 100, 200, 300]:
        raise ValueError("When using pretrained embeddings, embedding_dim must be one of [50, 100, 200, 300]")
    
    return args

def local_train_net(this_round_nets, args, client_dataloaders, test_dataloader, global_model=None, prev_models=None, round=None, device='cpu'):
    avg_acc = 0.0
    acc_dict = {}
    n_epoch = args.epochs
    client_times = []
    total_start_time = time.time()

    print(f'\n第 {round+1} 輪全域訓練開始')
    for net_id, net in this_round_nets.items():
        client_start_time = time.time()
        print(f"\n--- 訓練客戶端 {net_id} ---")
        client_dataloader = client_dataloaders[net_id]

        if args.alg == 'fedavg':
            # 進行 FedAvg 學習訓練
            trainacc, testacc = train_fedavg(net_id, net, client_dataloader, test_dataloader, n_epoch, args, round, device=device)
        elif args.alg == 'fedprox':
            # 進行 FedProx 學習訓練
            trainacc, testacc = train_fedprox(net_id, net, global_model, client_dataloader, test_dataloader, n_epoch, args, round, device=device)
        elif args.alg == 'moon':
            # 取出對應客戶端的歷史模型
            prev_models_client = []
            for i in range(len(prev_models)):
                prev_models_client.append(prev_models[i][net_id])

            # 進行 MOON 學習訓練
            trainacc, testacc = train_moon(net_id, net, global_model, prev_models_client, client_dataloader, test_dataloader, n_epoch, args, round, device=device)

        client_time = time.time() - client_start_time
        client_times.append(client_time)
        # 統一處理訓練結果輸出
        print(f"\n客戶端 {net_id} 訓練完成:")
        print(f"- 訓練準確率: {trainacc:.4f}")
        print(f"- 測試準確率: {testacc:.4f}")
        print(f"- 訓練時間: {client_time:.2f} 秒")
        acc_dict[net_id] = testacc
        avg_acc += testacc

    total_time = time.time() - total_start_time
    avg_acc /= len(this_round_nets)
    print(f'第 {round+1} 輪全域訓練完成')
    print(f"平均測試準確率: {avg_acc:.4f}")
    print(f"總訓練時間: {total_time:.2f} 秒")
    return avg_acc, acc_dict, client_times

def train_fedprox(net_id, net, global_model, client_dataloader, test_dataloader, n_epoch, args, round, device='cpu'):
    """
    執行 FedProx 的本地訓練
    
    參數:
        net_id: 客戶端 ID
        net: 客戶端模型
        global_model: 全域模型
        client_dataloader: 客戶端數據加載器
        test_dataloader: 測試數據加載器
        n_epoch: 本地訓練輪數
        args: 訓練參數
        round: 當前通信輪數
        device: 訓練設備
    
    返回:
        train_acc: 訓練準確率
        test_acc: 測試準確率
    """
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(n_epoch):
        print(f"第 {epoch+1} 輪訓練", end="\t")
        net.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(client_dataloader):
            data, target = data.to(device), target.to(device)

            opt.zero_grad()
            outputs = net(data)
            loss = net.loss(outputs, target, global_model)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"平均損失: {avg_loss}")

    train_acc = compute_accuracy(net, client_dataloader, device)
    test_acc = compute_accuracy(net, test_dataloader, device)

    return train_acc, test_acc

def train_moon(net_id, net, global_model, prev_models, client_dataloader, test_dataloader, n_epoch, args, round, device='cpu'):
    """
    執行 MOON 的本地訓練
    
    參數:
        net_id: 客戶端 ID
        net: 客戶端模型
        global_model: 全域模型
        prev_models: 歷史模型列表
        client_dataloader: 客戶端數據加載器
        test_dataloader: 測試數據加載器
        n_epoch: 本地訓練輪數
        args: 訓練參數
        round: 當前通信輪數
        device: 訓練設備
    
    返回:
        train_acc: 訓練準確率
        test_acc: 測試準確率
    """
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(n_epoch):
        print(f"第 {epoch+1} 輪訓練", end="\t")
        net.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(client_dataloader):
            data, target = data.to(device), target.to(device)

            opt.zero_grad()
            outputs = net(data)
            loss = net.loss(outputs, target, data, global_model, prev_models)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"平均損失: {avg_loss}")

    train_acc = compute_accuracy(net, client_dataloader, device)
    test_acc = compute_accuracy(net, test_dataloader, device)

    return train_acc, test_acc

def train_fedavg(net_id, net, client_dataloader, test_dataloader, n_epoch, args, round, device='cpu'):
    """
    執行 FedAvg 的本地訓練
    
    參數:
        net_id: 客戶端 ID
        net: 客戶端模型
        client_dataloader: 客戶端數據加載器
        test_dataloader: 測試數據加載器
        n_epoch: 本地訓練輪數
        args: 訓練參數
        round: 當前通信輪數
        device: 訓練設備
    
    返回:
        train_acc: 訓練準確率
        test_acc: 測試準確率
    """
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(n_epoch):
        print(f"第 {epoch+1} 輪訓練", end="\t")
        net.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(client_dataloader):
            data, target = data.to(device), target.to(device)

            opt.zero_grad()
            outputs = net(data)
            loss = net.loss(outputs, target)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"平均損失: {avg_loss}")

    train_acc = compute_accuracy(net, client_dataloader, device)
    test_acc = compute_accuracy(net, test_dataloader, device)

    return train_acc, test_acc

def global_train_round(args, round, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, prev_models=None, device='cpu'):
    """
    執行一輪訓練
    
    參數:
        args: 訓練參數
        round: 當前輪數
        clients_nets: 客戶端模型字典
        global_model: 全域模型
        client_dataloaders: 客戶端數據加載器字典
        test_dataloader: 測試數據加載器
        party_list_rounds: 每輪參與訓練的客戶端列表
        prev_models: 歷史模型列表（用於 MOON 算法）
        device: 訓練設備
    
    返回:
        avg_acc: 平均準確率
        acc_dict: 每個客戶端的準確率字典
        global_test_acc: 全域模型的測試準確率
    """    
    # 取得全域模型的權重
    global_state = global_model.state_dict()
    
    # 選擇本輪參與訓練的客戶端列表
    party_list_this_round = party_list_rounds[round]

    # 選擇本輪參與訓練的客戶端模型
    nets_this_round = {k: clients_nets[k] for k in party_list_this_round}

    # 將全域模型權重載入到每個客戶端模型
    for net in nets_this_round.values():
        net.load_state_dict(global_state)

    # 進行本地訓練
    avg_acc, acc_dict, client_times = local_train_net(nets_this_round, args, client_dataloaders, test_dataloader, global_model, prev_models, round, device)

    # 更新全域模型權重
    global_state = weights_aggregation(nets_this_round, client_dataloaders)

    # 更新全域模型
    global_model.load_state_dict(global_state)

    # 計算全域模型的測試準確率
    global_test_acc = compute_accuracy(global_model, test_dataloader, device)
    print(f"第 {round+1} 輪全域模型測試準確率: {global_test_acc:.4f}")

    return avg_acc, acc_dict, global_test_acc, client_times

def global_train_moon(args, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, device='cpu'):
    """
    執行 MOON 算法的全域訓練
    
    參數:
        args: 訓練參數
        clients_nets: 客戶端模型字典
        global_model: 全域模型
        client_dataloaders: 客戶端數據加載器字典
        test_dataloader: 測試數據加載器
        party_list_rounds: 每輪參與訓練的客戶端列表
        device: 訓練設備
    
    返回:
        comm_acc: 每輪的平均準確率列表
        comm_acc_dict: 每輪每個客戶端的準確率字典
        global_acc: 每輪全域模型的測試準確率列表
    """
    comm_acc = []
    comm_acc_dict = {}
    global_acc = []
    comm_times = {}

    # 初始化模型緩存
    old_nets_pool = []
    for _ in range(args.model_buffer_size):
        round_models = {}
        for client_id in range(args.n_parties):
            # 創建新的模型實例並複製權重
            new_args = copy.deepcopy(clients_nets[client_id].args)
            new_model = MOONModel(new_args)
            new_model.load_state_dict(clients_nets[client_id].state_dict())
            new_model.to(device)
            round_models[client_id] = new_model
        old_nets_pool.append(round_models)
    
    for round in range(args.comm_round):
        # 執行一輪訓練
        avg_acc, acc_dict, global_test_acc, client_times = global_train_round(
            args, round, clients_nets, global_model, client_dataloaders, 
            test_dataloader, party_list_rounds, old_nets_pool, device
        )
        
        comm_acc.append(avg_acc)
        comm_acc_dict[round] = acc_dict
        global_acc.append(global_test_acc)
        comm_times[round] = client_times

        # 更新模型緩存
        round_models = {}
        for client_id in range(args.n_parties):
            # 創建新的模型實例並複製權重
            new_args = copy.deepcopy(clients_nets[client_id].args)
            new_model = MOONModel(new_args)
            new_model.load_state_dict(clients_nets[client_id].state_dict())
            new_model.to(device)
            round_models[client_id] = new_model
        
        # 如果緩存已滿，移除最舊的模型
        if len(old_nets_pool) >= args.model_buffer_size:
            old_nets_pool.pop(0)
        old_nets_pool.append(round_models)

    return comm_acc, comm_acc_dict, global_acc, comm_times

def global_train_fedavg(args, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, device='cpu'):
    """
    執行 FedAvg 算法的全域訓練
    
    參數:
        args: 訓練參數
        clients_nets: 客戶端模型字典
        global_model: 全域模型
        client_dataloaders: 客戶端數據加載器字典
        test_dataloader: 測試數據加載器
        party_list_rounds: 每輪參與訓練的客戶端列表
        device: 訓練設備
    
    返回:
        comm_acc: 每輪的平均準確率列表
        comm_acc_dict: 每輪每個客戶端的準確率字典
        global_acc: 每輪全域模型的測試準確率列表
    """
    comm_acc = []
    comm_acc_dict = {}
    global_acc = []
    comm_times = {}
    for round in range(args.comm_round):
        # 執行一輪訓練
        avg_acc, acc_dict, global_test_acc, client_times = global_train_round(
            args, round, clients_nets, global_model, client_dataloaders, 
            test_dataloader, party_list_rounds, None, device
        )
        
        comm_acc.append(avg_acc)
        comm_acc_dict[round] = acc_dict
        global_acc.append(global_test_acc)
        comm_times[round] = client_times
    return comm_acc, comm_acc_dict, global_acc, comm_times

def global_train_fedprox(args, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, device='cpu'):
    """
    執行 FedProx 算法的全域訓練
    
    參數:
        args: 訓練參數
        clients_nets: 客戶端模型字典
        global_model: 全域模型
        client_dataloaders: 客戶端數據加載器字典
        test_dataloader: 測試數據加載器
        party_list_rounds: 每輪參與訓練的客戶端列表
        device: 訓練設備
    
    返回:
        comm_acc: 每輪的平均準確率列表
        comm_acc_dict: 每輪每個客戶端的準確率字典
        global_acc: 每輪全域模型的測試準確率列表
    """
    comm_acc = []
    comm_acc_dict = {}
    global_acc = []
    comm_times = {}
    for round in range(args.comm_round):
        # 執行一輪訓練
        avg_acc, acc_dict, global_test_acc, client_times = global_train_round(
            args, round, clients_nets, global_model, client_dataloaders, 
            test_dataloader, party_list_rounds, None, device
        )
        
        comm_acc.append(avg_acc)
        comm_acc_dict[round] = acc_dict
        global_acc.append(global_test_acc)
        comm_times[round] = client_times
    return comm_acc, comm_acc_dict, global_acc, comm_times

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
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('警告：CUDA 不可用，將使用 CPU 進行訓練')
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # 獲取預訓練詞嵌入
    pretrained_embeddings = None
    if args.use_pretrained_embeddings:
        pretrained_embeddings = load_glove_embeddings(
            vocab=train_dataset.vocab,
            embedding_dim=args.embedding_dim
        )
    
    # 初始化客戶端網絡
    clients_nets = init_nets(n_parties=args.n_parties, args=args, device=device, pretrained_embeddings=pretrained_embeddings)

    # 初始化服務器網絡
    global_nets = init_nets(n_parties=1, args=args, device=device, pretrained_embeddings=pretrained_embeddings)
    global_model = global_nets[0]
    
    # 計算每輪參與訓練的客戶端數量
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    # 訓練過程
    comm_acc, comm_acc_dict, global_acc, comm_times = [], {}, [], {}
    if args.alg == 'moon':
        comm_acc, comm_acc_dict, global_acc, comm_times = global_train_moon(args, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, device)
    elif args.alg == 'fedavg':
        comm_acc, comm_acc_dict, global_acc, comm_times = global_train_fedavg(args, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, device)
    elif args.alg == 'fedprox':
        comm_acc, comm_acc_dict, global_acc, comm_times = global_train_fedprox(args, clients_nets, global_model, client_dataloaders, test_dataloader, party_list_rounds, device)
    else:
        raise ValueError(f"不支持的算法: {args.alg}")

    # 保存訓練結果
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_path = f'results/{args.dataset}_{args.alg}_{args.model_name}_{timestamp}_results.json'

    # 將 args 轉換為字典
    args_dict = vars(args)

    results = {
        'args': args_dict,   # 訓練參數
        'comm_acc_avg': comm_acc,  # 每輪的客戶端平均準確率
        'comm_acc_client': comm_acc_dict,  # 每輪每個客戶端的準確率字典
        'global_acc': global_acc,  # 每輪全域模型的測試準確率
        'comm_times': comm_times  # 每輪的訓練時間
    }

    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"訓練結果已保存到 {args.save_path}")
