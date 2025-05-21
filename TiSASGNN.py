import numpy as np
import torch
import torch.nn.functional as F
import sys
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2**32+1)
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class GraphConvLayer(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, gnn_type='gcn'):
        super(GraphConvLayer, self).__init__()
        
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.gnn_type=gnn_type
        if gnn_type == 'gcn':
            self.conv = GCNConv(hidden_units, hidden_units)
        elif gnn_type == 'gat':
            self.conv = GATConv(hidden_units, hidden_units,edge_dim=1)
        elif gnn_type == 'sage':
            self.conv = SAGEConv(hidden_units, hidden_units)
            self.edge_encoder = torch.nn.Linear(1, hidden_units)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
        self.norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        
    def forward(self, x, edge_index,edge_attr=None):
        residual = x
        if self.gnn_type == 'gat' and edge_attr is not None:
            # GAT可以直接使用边特征
            x = self.conv(x, edge_index, edge_attr)
        elif self.gnn_type == 'sage' and edge_attr is not None:
            # 自定义处理SAGE的边特征
            edge_features = self.edge_encoder(edge_attr)
            # 将边特征应用到目标节点（简化版本）
            source, target = edge_index
            source_x = x[source]
            # 加权聚合，使用边特征作为权重
            weighted_source_x = source_x * edge_features
            # 执行常规SAGE操作
            x = self.conv(x, edge_index)
            # 修改结果，考虑边特征
            # 这里需要根据您的具体需求设计聚合策略
        else:
            # 默认情况，不使用边特征
            x = self.conv(x, edge_index)
        
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class TiSASGNN(torch.nn.Module):
    def __init__(self, user_num, item_num, time_num, args):
        super(TiSASGNN, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        # Embedding layers
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.user_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Time-aware components from TiSASRec
        self.abs_pos_K_emb = torch.nn.Embedding(args.max_len, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.max_len, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)

        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # SASRec components
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # GNN components
        self.gnn_layers = torch.nn.ModuleList()
        
        # Fusion layer to combine SASRec and GNN outputs
        self.fusion_gate = torch.nn.Linear(args.hidden_units * 2, args.hidden_units)
        self.fusion_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        # Final layer norm
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Initialize blocks for both SASRec and GNN branches
        for _ in range(args.num_blocks):
            # SASRec branch
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                args.hidden_units,
                args.num_heads,
                args.dropout_rate,
                args.device
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            
            # GNN branch
            new_gnn_layer = GraphConvLayer(
                args.hidden_units, 
                args.dropout_rate,
                gnn_type=args.gnn_type
            )
            self.gnn_layers.append(new_gnn_layer)

    def build_graph(self, user_ids, log_seqs,ratings=None):
        """Build user-item interaction graph for GNN processing"""
        batch_size = log_seqs.shape[0]
        seq_len = log_seqs.shape[1]
        
        # Create edges between users and items they interacted with
        edge_list = []
        edge_attr=[]
        node_features = []
        
        # First, add all users
        for i, user_id in enumerate(user_ids):
            if user_id == 0:  # Skip padding user
                continue
            node_features.append(self.user_emb(torch.tensor([user_id]).to(self.dev)))
        
        user_offset = len(node_features)
        
        # Then add all items
        item_id_to_node_idx = {}
        for i in range(batch_size):
            for j in range(seq_len):
                item_id = log_seqs[i, j]
                if item_id == 0:  # Skip padding item
                    continue
                
                if item_id not in item_id_to_node_idx:
                    item_id_to_node_idx[item_id] = len(node_features) - user_offset
                    node_features.append(self.item_emb(torch.tensor([item_id]).to(self.dev)))
                
                rating=1.0
                if ratings is not None:
                    rating = ratings[i, j]
                # Add edge: user -> item and item -> user (bidirectional)
                if user_ids[i] != 0:  # Skip padding user
                    user_node_idx = i
                    item_node_idx = item_id_to_node_idx[item_id] + user_offset
                    edge_list.append([user_node_idx, item_node_idx])
                    edge_list.append([item_node_idx, user_node_idx])  # Bidirectional
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.dev)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(self.dev)
        x = torch.cat(node_features, dim=0).squeeze(1)
        
        return x, edge_index, edge_attr,user_offset, item_id_to_node_idx

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        """Process sequential data with time-aware self-attention (TiSASRec branch)"""
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # Mask for timeline and attention
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # Mask padded items

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # Apply self-attention blocks
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs

            # Point-wise Feed-forward
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats
    
    def graph2feats(self, user_ids, log_seqs,ratings=None):
        """Process graph data with GNN (Graph branch)"""
        # Build graph from user-item interactions
        x, edge_index, edge_attr,user_offset, item_map = self.build_graph(user_ids, log_seqs,ratings)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index,edge_attr)
        
        # Extract user and item embeddings from the graph
        batch_size = len(user_ids)
        seq_len = log_seqs.shape[1]
        
        # Initialize output tensor for item sequences
        gnn_feats = torch.zeros((batch_size, seq_len, x.shape[1]), device=self.dev)
        
        # Fill in the GNN-processed features
        for i in range(batch_size):
            for j in range(seq_len):
                item_id = log_seqs[i, j]
                if item_id == 0:  # Skip padding
                    continue
                
                if item_id in item_map:
                    item_node_idx = item_map[item_id] + user_offset
                    gnn_feats[i, j] = x[item_node_idx]
        
        return gnn_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs,ratings=None):
        """Forward pass for training"""
        # Get sequence features from TiSASRec branch
        seq_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        
        # Get graph features from GNN branch
        graph_feats = self.graph2feats(user_ids, log_seqs,ratings)
        
        # Fusion gate to combine both features
        # Concatenate features and apply fusion gate
        combined_feats = torch.cat([seq_feats, graph_feats], dim=-1)
        fusion_weights = torch.sigmoid(self.fusion_gate(combined_feats))
        
        # Weighted combination of sequence and graph features
        log_feats = fusion_weights * seq_feats + (1 - fusion_weights) * graph_feats
        log_feats = self.fusion_norm(log_feats)
        
        # Calculate logits for positive and negative samples
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
        # 使用 log_feats[:, :-1, :] 来匹配 pos_seqs 的维度
        # 因为 pos_seqs 是 log_seqs 向右移动一位（即 log_seqs[:, 1:]）
        log_feats = log_feats[:, :-1, :]

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, time_matrices, item_indices,ratings=None):
        """Forward pass for inference"""
        # Get sequence features from TiSASRec branch
        seq_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        
        # Get graph features from GNN branch
        graph_feats = self.graph2feats(user_ids, log_seqs,ratings)
        
        # Fusion gate to combine both features - use only the last position
        # Concatenate features and apply fusion gate
        combined_feats = torch.cat([seq_feats, graph_feats], dim=-1)
        fusion_weights = torch.sigmoid(self.fusion_gate(combined_feats))
        
        # Weighted combination of sequence and graph features
        log_feats = fusion_weights * seq_feats + (1 - fusion_weights) * graph_feats
        log_feats = self.fusion_norm(log_feats)
        
        # Use the final representation for prediction
        final_feat = log_feats[:, -1, :]  # only use last position

        # Get embeddings for all candidate items
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        # Calculate scores
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


# Utility function for training the model
def train_tisasgnn(model, train_data, optimizer, batch_size, args):
    """
    Training utility function
    """
    model.train()
    total_loss = 0
    
    # Get training data
    users, seqs, times, ratings,labels = train_data
    
    num_batches = len(users) // batch_size + (0 if len(users) % batch_size == 0 else 1)
    
    for i in range(num_batches):
        batch_users = users[i*batch_size:(i+1)*batch_size]
        batch_seqs = seqs[i*batch_size:(i+1)*batch_size]
        batch_times = times[i*batch_size:(i+1)*batch_size]
        batch_ratings = ratings[i*batch_size:(i+1)*batch_size]
        # For each sequence, generate positive and negative samples
        pos_seqs = batch_seqs[:, 1:]  # Shifted sequence for next item prediction
        neg_seqs = np.random.randint(1, model.item_num+1, pos_seqs.shape)  # Random negative sampling
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pos_logits, neg_logits = model(batch_users, batch_seqs, batch_times, pos_seqs, neg_seqs,batch_ratings)
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits)))
        
        # L2 regularization if specified
        if args.l2_emb > 0:
            l2_loss = 0
            for name, param in model.named_parameters():
                if 'emb' in name:
                    l2_loss += torch.sum(param ** 2)
            loss += args.l2_emb * l2_loss
            
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


