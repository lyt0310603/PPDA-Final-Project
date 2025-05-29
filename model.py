import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, args):
        """
        初始化 LSTM 模型
        
        參數:
            args: 包含模型參數的命名空間
                - vocab_size: 詞彙表大小
                - embedding_dim: 詞嵌入維度
                - hidden_dim: LSTM 隱藏層維度
                - n_layers: LSTM 層數
                - dropout: Dropout 比率
                - pretrained_embeddings: 預訓練的詞嵌入矩陣
                - freeze_embeddings: 是否凍結詞嵌入層
        """
        super().__init__()
        
        # 初始化詞嵌入層
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        if args.pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(args.pretrained_embeddings, freeze=args.freeze_embeddings)
        
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
        
    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: 輸入文本張量 [batch_size, seq_len]
        返回:
            last_hidden: 最後一個時間步的隱藏狀態 [batch_size, hidden_dim * 2]
        """
        # text = [batch size, sent len]
        
        embedded = self.dropout(self.embedding(x))
        # embedded = [batch size, sent len, emb dim]
        
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
    pass

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model_name
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.n_classes
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.pretrained_embeddings = args.pretrained_embeddings
        self.freeze_embeddings = args.freeze_embeddings

        if self.model_name == 'LSTM':
            self.encoder = LSTM(args)
        elif self.model_name == 'Transformer':
            self.encoder = Transformer(args)
        
        # 使用 Sequential 包裝多層全連接網路
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, x):
        hidden = self.encoder(x)
        return self.fc(hidden), None
    
    def loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)

class MOONModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.projection_head = nn.Linear(self.hidden_dim*2, args.projection_dim)
        self.temperature = args.temperature
    
    def forward(self, x):
        hidden = self.encoder(x)
        projected = self.projection_head(hidden)
        return self.fc(hidden), projected
    
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
            hidden = self.encoder(x)
            projected = self.projection_head(hidden)
            
            # 恢復原始權重
            self.encoder.load_state_dict(current_encoder)
            self.projection_head.load_state_dict(current_projection)
            
            return projected
    
    def loss(self, outputs, labels, x, global_weights=None, prev_weights=None, mu=0.1):
        """計算 MOON 損失
        
        參數:
            outputs: (logits, projected) 元組
            labels: 真實標籤
            x: 輸入數據
            global_weights: 全局模型的權重
            prev_weights: 上一個本地模型的權重
            mu: 對比損失權重
            
        返回:
            total_loss: 總損失
        """
        logits, projected = outputs
        cls_loss = F.cross_entropy(logits, labels)
        
        if global_weights is None or prev_weights is None:
            return cls_loss
            
        # 計算對比損失
        global_projected = self._compute_projection(x, global_weights)
        prev_projected = self._compute_projection(x, prev_weights)
        
        # 計算正樣本對的相似度（當前模型和全局模型）
        pos_sim = F.cosine_similarity(projected, global_projected)
        
        # 計算負樣本對的相似度（當前模型和上一個本地模型）
        neg_sim = F.cosine_similarity(projected, prev_projected)
        
        # 計算 InfoNCE loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        contrast_loss = F.cross_entropy(logits / self.temperature, labels)
        
        return cls_loss + mu * contrast_loss

class FedAvgModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

class FedProxModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
    
    def get_weights(self):
        """獲取模型的所有權重"""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
    
    def loss(self, outputs, labels, global_weights=None, mu=0.1):
        """計算 FedProx 損失，包括分類損失和正則化項
        
        參數:
            outputs: 模型輸出
            labels: 真實標籤
            global_weights: 全局模型的權重
            mu: 正則化係數
            
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
        
        return cls_loss + (mu / 2) * proximal_loss



