import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def load_model_weights(model, weights):
    """
    將自訂格式的權重加載到模型中
    
    參數:
        model: 要加載權重的模型
        weights: 自訂格式的權重字典，格式為：
            {
                'embedding': {...},
                'encoder': {...},
                'fc': {...},
                'projection_head': {...} (可選)
            }
    """
    # 創建一個新的 state_dict
    new_state_dict = {}
    
    # 更新 embedding 權重
    for k, v in weights['embedding'].items():
        new_state_dict[f'embedding.{k}'] = v
        
    # 更新 encoder 權重
    for k, v in weights['encoder'].items():
        new_state_dict[f'encoder.{k}'] = v
        
    # 更新 fc 權重
    for k, v in weights['fc'].items():
        new_state_dict[f'fc.{k}'] = v
        
    # 更新 projection_head 權重（如果是 MOON 模型）
    if 'projection_head' in weights:
        for k, v in weights['projection_head'].items():
            new_state_dict[f'projection_head.{k}'] = v
            
    # 使用新的 state_dict 更新模型
    model.load_state_dict(new_state_dict)

class LSTM(nn.Module):
    def __init__(self, args):
        """
        初始化 LSTM 模型
        
        參數:
            args: 包含模型參數的命名空間
                - hidden_dim: LSTM 隱藏層維度
                - n_layers: LSTM 層數
                - dropout: Dropout 比率
        """
        super().__init__()
        
        # 手動建立多個 LSTM 層
        self.lstm_layers = nn.ModuleList()
        for i in range(args.n_layers):
            input_size = args.embedding_dim if i == 0 else args.hidden_dim * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=args.hidden_dim,
                    bidirectional=True,
                    batch_first=True
                )
            )
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, embedded, mask=None):
        """
        前向傳播
        
        參數:
            embedded: 輸入文本張量 [batch_size, seq_len, embedding_dim]
            mask: 注意力掩碼 [batch_size, seq_len] 或 None（LSTM 不使用此參數）
        返回:
            last_hidden: 最後一個時間步的隱藏狀態 [batch_size, hidden_dim * 2]
        """
        # 逐層通過 LSTM
        current_input = embedded
        for lstm_layer in self.lstm_layers:
            current_input, _ = lstm_layer(current_input)
            current_input = self.dropout(current_input)
        
        # 取最後一個時間步的輸出
        # current_input = [batch size, sent len, hid dim * 2]
        last_hidden = current_input[:, -1, :]
        # last_hidden = [batch size, hid dim * 2]
        
        return last_hidden

class Transformer(nn.Module):
    def __init__(self, args):
        """
        初始化 Transformer 模型
        
        參數:
            args: 包含模型參數的命名空間
                - hidden_dim: Transformer 隱藏層維度
                - n_layers: Transformer 層數
                - n_heads: 注意力頭數
                - dropout: Dropout 比率
                - max_length: 最大序列長度
        """
        super().__init__()
        
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.dropout = args.dropout
        self.max_length = args.max_length
        self.embedding_dim = args.embedding_dim
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(args.embedding_dim, args.max_length)
        
        # 手動建立多個 Transformer 編碼器層
        self.encoder_layers = nn.ModuleList()
        for i in range(args.n_layers):
            self.encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=args.embedding_dim,
                    nhead=args.n_heads,
                    dim_feedforward=args.hidden_dim * 4,
                    dropout=args.dropout,
                    batch_first=True
                )
            )
        
        self.dropout = nn.Dropout(args.dropout)
        
    def get_mask(self, x):
        """
        生成 padding mask
        
        參數:
            x: 輸入張量 [batch_size, seq_len]
        返回:
            mask: padding mask [batch_size, seq_len]，True 表示需要被遮擋的位置
        """
        # 創建 mask，其中 0 的位置（padding）會被遮擋
        mask = (x == 0)
        return mask
        
    def forward(self, embedded, mask=None):
        """
        前向傳播
        
        參數:
            embedded: 輸入文本張量 [batch_size, seq_len, embedding_dim]
            mask: 注意力掩碼 [batch_size, seq_len] 或 None
        返回:
            last_hidden: 聚合後的隱藏狀態 [batch_size, hidden_dim * 2]
        """
        # 添加位置編碼
        embedded = self.pos_encoder(embedded)
        
        # 逐層通過 Transformer 編碼器
        current_input = embedded
        for encoder_layer in self.encoder_layers:
            current_input = encoder_layer(current_input, src_key_padding_mask=mask)
            current_input = self.dropout(current_input)
        
        # 使用平均池化聚合所有時間步的信息
        # 如果提供了 mask，則只對非 padding 位置進行平均
        if mask is not None:
            # 將 mask 轉換為浮點數，並擴展維度以進行廣播
            mask = (~mask).float().unsqueeze(-1)
            # 計算每個序列的有效長度
            lengths = mask.sum(dim=1)
            # 對非 padding 位置進行平均
            last_hidden = (current_input * mask).sum(dim=1) / (lengths + 1e-9)
        else:
            # 如果沒有 mask，直接對所有位置進行平均
            last_hidden = current_input.mean(dim=1)
        
        return last_hidden

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """
        初始化位置編碼
        
        參數:
            d_model: 模型維度
            max_seq_len: 最大序列長度
        """
        super().__init__()
        
        # 創建位置編碼矩陣
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 註冊為緩衝區（不參與反向傳播）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置編碼
        
        參數:
            x: 輸入張量 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1)].unsqueeze(0)

class BaseModel(nn.Module):
    def __init__(self, args, pretrained_embeddings=None):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.n_classes
        self.n_layers = args.n_layers
        self.dropout_rate = args.dropout
        self.freeze_embeddings = args.freeze_embeddings

        # 初始化詞嵌入層
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=args.freeze_embeddings)
            self.args.vocab_size = pretrained_embeddings.shape[0]
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.model_name == 'LSTM':
            self.encoder = LSTM(args)
            self.encoder_output_dim = args.hidden_dim * 2
        elif self.model_name == 'Transformer':
            self.encoder = Transformer(args)
            self.encoder_output_dim = args.embedding_dim

        # 使用 Sequential 包裝多層全連接網路
        self.fc = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

        # 初始化
        self.init_weights()
    
    def forward(self, x, mask=None):
        embedded = self.dropout(self.embedding(x))
        
        # 如果沒有提供 mask，則生成 mask
        if mask is None and self.model_name == 'Transformer':
            mask = self.encoder.get_mask(x)
            
        hidden = self.encoder(embedded, mask)
        return self.fc(hidden)
    
    def loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)
        
    def init_weights(self):
        """初始化模型權重，除了預訓練的 embedding"""
        for name, param in self.named_parameters():
            # 跳過預訓練的 embedding
            if name == 'embedding.weight' and self.args.use_pretrained_embeddings:
                continue
            # 只對權重矩陣使用 Xavier 初始化
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            # 對偏置項使用零初始化
            elif 'bias' in name:
                nn.init.zeros_(param)

class MOONModel(BaseModel):
    def __init__(self, args, pretrained_embeddings=None):
        super().__init__(args, pretrained_embeddings)
        self.projection_head = nn.Linear(self.encoder_output_dim, args.projection_dim)
        self.temperature = args.temperature
        self.mu = args.mu
        self.cos = nn.CosineSimilarity(dim=-1)  # 初始化餘弦相似度計算器
        self.init_weights()
    
    def forward(self, x, mask=None):
        """
        前向傳播
        
        參數:
            x: 輸入數據
            mask: 注意力掩碼（可選）
        
        返回:
            (logits, projected): 分類輸出和投影輸出
        """
        # 計算嵌入
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # 如果沒有提供 mask，則生成 mask
        if mask is None and self.model_name == 'Transformer':
            mask = self.encoder.get_mask(x)
        
        # 計算隱藏狀態
        hidden = self.encoder(embedded, mask)
        
        # 計算分類輸出和投影輸出
        logits = self.fc(hidden)
        projected = self.projection_head(hidden)
        
        return logits, projected
    
    def loss(self, outputs, labels, x, global_model=None, prev_models=None):
        """計算 MOON 損失
        
        參數:
            outputs: (logits, projected) 元組
            labels: 真實標籤
            x: 輸入數據
            global_model: 全局模型
            prev_models: 歷史模型列表
            
        返回:
            total_loss: 總損失
        """
        logits, projected = outputs
        # 正規化投影向量
        projected = F.normalize(projected, dim=1)
        cls_loss = F.cross_entropy(logits, labels)
        
        if global_model is None or prev_models is None or len(prev_models) == 0:
            return cls_loss
            
        # 計算對比損失
        with torch.no_grad():
            global_model.eval()
            _, global_projected = global_model(x)
            global_projected = F.normalize(global_projected, dim=1)
        
        # 計算與全局模型的相似度作為正樣本
        pos_sim = self.cos(projected, global_projected)
        logits_contrast = pos_sim.unsqueeze(1)  # [batch_size, 1]
        
        # 計算與所有歷史模型的相似度作為負樣本
        for prev_model in prev_models:
            with torch.no_grad():
                prev_model.eval()
                _, prev_projected = prev_model(x)
                prev_projected = F.normalize(prev_projected, dim=1)
            neg_sim = self.cos(projected, prev_projected)
            logits_contrast = torch.cat([logits_contrast, neg_sim.unsqueeze(1)], dim=1)  # [batch_size, n_models]
        
        # 應用溫度縮放
        logits_contrast = logits_contrast / self.temperature
        
        # 創建標籤（第一個位置為正樣本）
        contrast_labels = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        
        # 計算對比損失
        contrast_loss = F.cross_entropy(logits_contrast, contrast_labels)
        
        return cls_loss + self.mu * contrast_loss

class FedAvgModel(BaseModel):
    def __init__(self, args, pretrained_embeddings=None):
        super().__init__(args, pretrained_embeddings)
        self.init_weights()

class FedProxModel(BaseModel):
    def __init__(self, args, pretrained_embeddings=None):
        super().__init__(args, pretrained_embeddings)
        self.mu = args.mu
        self.init_weights()
    
    def loss(self, outputs, labels, global_model=None):
        """計算 FedProx 損失，包括分類損失和正則化項
        
        參數:
            outputs: 模型輸出
            labels: 真實標籤
            global_model: 全局模型
            
        返回:
            total_loss: 總損失
        """
        cls_loss = F.cross_entropy(outputs, labels)
        
        if global_model is None:
            return cls_loss
            
        # 獲取當前模型和全局模型的權重
        current_state = self.state_dict()
        global_state = global_model.state_dict()
        
        # 計算正則化項
        proximal_loss = 0.0
        for key in current_state:
            if 'weight' in key or 'bias' in key:  # 只計算權重和偏置項
                proximal_loss += torch.mean((current_state[key] - global_state[key]) ** 2)
        
        return cls_loss + (self.mu / 2) * proximal_loss



