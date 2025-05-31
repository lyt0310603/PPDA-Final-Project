import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, d_model, max_seq_len=5000):
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
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model_name
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.n_classes
        self.n_layers = args.n_layers
        self.dropout_rate = args.dropout
        self.pretrained_embeddings = args.pretrained_embeddings
        self.freeze_embeddings = args.freeze_embeddings

        # 初始化詞嵌入層
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        if args.pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(args.pretrained_embeddings, freeze=args.freeze_embeddings)
        
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
        
    def get_weights(self):
        """獲取模型的所有權重"""
        return {
            'encoder': {k: v.detach().clone() for k, v in self.encoder.state_dict().items()}
        }
        
    def init_weights(self):
        """初始化模型權重，除了預訓練的 embedding"""
        for name, param in self.named_parameters():
            # 跳過預訓練的 embedding
            if name == 'embedding.weight' and self.pretrained_embeddings is not None:
                continue
            # 只對權重矩陣使用 Xavier 初始化
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            # 對偏置項使用零初始化
            elif 'bias' in name:
                nn.init.zeros_(param)

class MOONModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.projection_head = nn.Linear(self.encoder_output_dim, args.projection_dim)
        self.temperature = args.temperature
        self.mu = args.mu
        self.cos = nn.CosineSimilarity(dim=-1)  # 初始化餘弦相似度計算器
        self.init_weights()
    
    def forward(self, x, mask=None):
        embedded = self.dropout(self.embedding(x))
        
        # 如果沒有提供 mask，則生成 mask
        if mask is None and self.model_name == 'Transformer':
            mask = self.encoder.get_mask(x)
            
        hidden = self.encoder(embedded, mask)
        projected = self.projection_head(hidden)
        return self.fc(hidden), projected
    
    def get_weights(self):
        """獲取模型的所有權重"""
        weights = super().get_weights()
        weights['projection'] = {k: v.detach().clone() for k, v in self.projection_head.state_dict().items()}
        return weights
    
    def get_projection_weights(self):
        """獲取投影頭的權重"""
        return {
            'encoder': {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
            'projection': {k: v.detach().clone() for k, v in self.projection_head.state_dict().items()}
        }
    
    def _compute_projection(self, x, weights):
        """使用給定的權重計算投影
        
        參數:
            x: 輸入數據
            weights: 包含 encoder 和 projection 權重的字典
        """
        with torch.no_grad():
            # 臨時保存當前權重
            current_encoder = self.encoder.state_dict()
            current_projection = self.projection_head.state_dict()
            
            # 載入給定的權重
            self.encoder.load_state_dict(weights['encoder'])
            self.projection_head.load_state_dict(weights['projection'])
            
            # 計算投影
            embedded = self.embedding(x)
            if self.model_name == 'Transformer':
                mask = self.encoder.get_mask(x)
            else:
                mask = None
            hidden = self.encoder(embedded, mask)
            projected = self.projection_head(hidden)
            
            # 正規化投影向量
            projected = F.normalize(projected, dim=1)
            
            # 恢復原始權重
            self.encoder.load_state_dict(current_encoder)
            self.projection_head.load_state_dict(current_projection)
            
            return projected
    
    def loss(self, outputs, labels, x, global_weight=None, prev_weights=None):
        """計算 MOON 損失
        
        參數:
            outputs: (logits, projected) 元組
            labels: 真實標籤
            x: 輸入數據
            global_weights: 全局模型的權重
            prev_weights: 歷史模型權重的列表
            
        返回:
            total_loss: 總損失
        """
        logits, projected = outputs
        cls_loss = F.cross_entropy(logits, labels)
        
        if global_weight is None or prev_weights is None or len(prev_weights) == 0:
            return cls_loss
            
        # 計算對比損失
        global_projected = self._compute_projection(x, global_weight)
        
        # 計算與全局模型的相似度作為正樣本
        pos_sim = self.cos(projected, global_projected)
        logits = pos_sim.reshape(-1, 1)
        
        # 計算與所有歷史模型的相似度作為負樣本
        for prev_weight in prev_weights:
            prev_projected = self._compute_projection(x, prev_weight)
            neg_sim = self.cos(projected, prev_projected)
            logits = torch.cat((logits, neg_sim.reshape(-1, 1)), dim=1)
        
        # 應用溫度縮放
        logits /= self.temperature
        
        # 創建標籤（第一個位置為正樣本）
        contrast_labels = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        
        # 計算對比損失
        contrast_loss = F.cross_entropy(logits, contrast_labels)
        
        return cls_loss + self.mu * contrast_loss

class FedAvgModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.init_weights()

class FedProxModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.mu = args.mu
        self.init_weights()
        
    def get_weights(self):
        """獲取模型的所有權重"""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
    
    def loss(self, outputs, labels, global_weights=None):
        """計算 FedProx 損失，包括分類損失和正則化項
        
        參數:
            outputs: 模型輸出
            labels: 真實標籤
            global_weights: 全局模型的權重
            
        返回:
            total_loss: 總損失
        """
        cls_loss = F.cross_entropy(outputs, labels)
        
        if global_weights is None:
            return cls_loss
            
        # 計算正則化項
        proximal_loss = 0.0
        for k, v in self.state_dict().items():
            proximal_loss += torch.sum((v - global_weights[k]) ** 2)
        
        return cls_loss + (self.mu / 2) * proximal_loss



