import numpy as np
import torch
import torch.nn.functional as F
import sys
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import random

FLOAT_MIN = -sys.float_info.max

class RandomWalkEnhancer:
    """
    随机游走模块，用于增强图结构，处理稀疏数据
    """
    def __init__(self, walk_length=3, num_walks=10, p=1.0, q=1.0, use_edge_weights=True):
        """
        初始化随机游走增强器
        
        参数:
            walk_length: 每次随机游走的长度
            num_walks: 从每个节点开始的随机游走次数
            p: 返回参数（控制返回上一节点的可能性）
            q: 出入参数（控制深度优先vs宽度优先搜索）
            use_edge_weights: 是否在游走中考虑边权重
        """
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.use_edge_weights = use_edge_weights
    
    def preprocess_transition_probs(self, edge_index, edge_weight=None, num_nodes=None):
        """
        预处理转移概率，用于偏置随机游走（类似node2vec）
        """
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
            
        # 构建邻接表
        adj_dict = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in adj_dict:
                adj_dict[src] = []
            
            # 如果有边权重，把它和目标节点一起存储
            if edge_weight is not None and self.use_edge_weights:
                weight = edge_weight[i].item()
                adj_dict[src].append((dst, weight))
            else:
                adj_dict[src].append((dst, 1.0))
        
        return adj_dict
    
    def compute_transition_prob(self, prev_node, curr_node, next_node, adj_dict):
        """
        计算从当前节点到下一节点的转移概率
        """
        if next_node not in [dst for dst, _ in adj_dict.get(curr_node, [])]:
            return 0.0
            
        # 为biased walk设置权重
        weight = 1.0
        
        # 如果有前一个节点，应用p和q参数
        if prev_node is not None:
            if next_node == prev_node:  # 返回上一个节点
                weight = 1.0 / self.p
            elif next_node in [dst for dst, _ in adj_dict.get(prev_node, [])]:  # 前一个节点的邻居
                weight = 1.0
            else:  # 其他节点
                weight = 1.0 / self.q
                
        # 如果使用边权重，乘以边权重
        for dst, w in adj_dict.get(curr_node, []):
            if dst == next_node:
                weight *= w
                break
                
        return weight
    
    def get_next_node(self, prev_node, curr_node, adj_dict):
        """
        根据转移概率选择下一个节点
        """
        neighbors = adj_dict.get(curr_node, [])
        if not neighbors:
            return None
            
        # 计算所有可能的下一个节点的转移概率
        probs = []
        nodes = []
        
        for next_node, _ in neighbors:
            prob = self.compute_transition_prob(prev_node, curr_node, next_node, adj_dict)
            if prob > 0:
                probs.append(prob)
                nodes.append(next_node)
        
        if not nodes:
            return None
            
        # 归一化概率
        probs_sum = sum(probs)
        probs = [p / probs_sum for p in probs]
        
        # 根据概率选择下一个节点
        return random.choices(nodes, weights=probs, k=1)[0]
    
    def generate_walk(self, start_node, adj_dict):
        """
        从给定的起始节点生成一次随机游走
        """
        walk = [start_node]
        prev_node = None
        curr_node = start_node
        
        for _ in range(self.walk_length - 1):
            next_node = self.get_next_node(prev_node, curr_node, adj_dict)
            if next_node is None:
                break
                
            walk.append(next_node)
            prev_node = curr_node
            curr_node = next_node
            
        return walk
    
    def generate_walks(self, edge_index, edge_weight=None, num_nodes=None):
        """
        从所有节点生成随机游走
        """
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
            
        adj_dict = self.preprocess_transition_probs(edge_index, edge_weight, num_nodes)
        
        walks = []
        nodes = list(range(num_nodes))
        
        # 从每个节点开始进行随机游走
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if node in adj_dict:  # 只从有出边的节点开始游走
                    walk = self.generate_walk(node, adj_dict)
                    if len(walk) > 1:  # 只保留长度大于1的游走
                        walks.append(walk)
        
        return walks
    
    def enhance_graph(self, edge_index, edge_weight=None, num_nodes=None):
        """
        通过随机游走增强图结构
        
        返回:
            enhanced_edge_index: 增强后的边索引
            enhanced_edge_weight: 增强后的边权重
        """
        device = edge_index.device
        
        # 生成随机游走
        walks = self.generate_walks(edge_index, edge_weight, num_nodes)
        
        # 创建新的边和权重
        new_edges = []
        new_weights = []
        
        # 现有边的字典，用于快速查找
        existing_edges = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges[(src, dst)] = i
        
        # 从随机游走中提取边
        for walk in walks:
            for i in range(len(walk) - 1):
                src, dst = walk[i], walk[i + 1]
                
                # 根据节点在路径中的距离计算权重衰减
                decay = 1.0 / (i + 1)  # 距离越远，权重越小
                
                # 检查边是否已存在
                if (src, dst) in existing_edges:
                    idx = existing_edges[(src, dst)]
                    if edge_weight is not None:
                        # 增加现有边的权重
                        edge_weight[idx] += decay
                else:
                    # 添加新边
                    new_edges.append([src, dst])
                    new_weights.append(decay)
                    # 添加反向边以保持图的无向性
                    new_edges.append([dst, src])
                    new_weights.append(decay)
                    # 更新现有边字典
                    existing_edges[(src, dst)] = len(existing_edges)
                    existing_edges[(dst, src)] = len(existing_edges)
        
        # 合并原始边和新边
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=device).t()
            new_edge_weight = torch.tensor(new_weights, dtype=torch.float, device=device)
            
            # 组合原始边和新边
            enhanced_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            
            if edge_weight is not None:
                enhanced_edge_weight = torch.cat([edge_weight, new_edge_weight])
            else:
                # 如果原始图没有边权重，为所有原始边分配权重1
                orig_weight = torch.ones(edge_index.shape[1], dtype=torch.float, device=device)
                enhanced_edge_weight = torch.cat([orig_weight, new_edge_weight])
        else:
            enhanced_edge_index = edge_index
            enhanced_edge_weight = edge_weight if edge_weight is not None else torch.ones(edge_index.shape[1], dtype=torch.float, device=device)
        
        return enhanced_edge_index, enhanced_edge_weight


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
        
    def forward(self, x, edge_index, edge_attr=None):
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

        # self.global_graph_built = False
        # self.global_x = None
        # self.global_edge_index = None
        # self.global_edge_attr = None
        # self.global_user_offset = None
        self.global_item_map = {}
        # 随机游走增强器
        self.random_walk_enhancer = RandomWalkEnhancer(
            walk_length=args.walk_length if hasattr(args, 'walk_length') else 3,
            num_walks=args.num_walks if hasattr(args, 'num_walks') else 10,
            p=args.rw_p if hasattr(args, 'rw_p') else 1.0,
            q=args.rw_q if hasattr(args, 'rw_q') else 1.0,
            use_edge_weights=args.use_edge_weights if hasattr(args, 'use_edge_weights') else True
        )
        
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
    # def build_global_graph(self, all_user_ids, all_log_seqs, all_ratings=None):
    #     """Build the full user-item interaction graph once"""
    #     print("Building global interaction graph...")
        
    #     # Similar to build_graph but processes all data at once
    #     edge_list = []
    #     edge_attr_list = []
    #     node_features = []
        
    #     # Add all users
    #     user_id_to_node_idx = {}
    #     for i, user_id in enumerate(all_user_ids):
    #         if user_id == 0:  # Skip padding users
    #             continue
    #         if user_id not in user_id_to_node_idx:
    #             user_id_to_node_idx[user_id] = len(node_features)
    #             node_features.append(self.user_emb(torch.tensor([user_id]).to(self.dev)))
        
    #     user_offset = len(node_features)
        
    #     # Add all items
    #     item_id_to_node_idx = {}
    #     for i in range(len(all_user_ids)):
    #         user_id = all_user_ids[i]
    #         if user_id == 0:  # Skip padding users
    #             continue
                
    #         for j in range(all_log_seqs.shape[1]):
    #             item_id = all_log_seqs[i, j]
    #             if item_id == 0:  # Skip padding items
    #                 continue
                
    #             if item_id not in item_id_to_node_idx:
    #                 item_id_to_node_idx[item_id] = len(node_features) - user_offset
    #                 node_features.append(self.item_emb(torch.tensor([item_id]).to(self.dev)))
                
    #             # Set edge weight
    #             rating = 1.0
    #             if all_ratings is not None:
    #                 rating = all_ratings[i, j]
                
    #             # Add edges: user -> item and item -> user (bidirectional)
    #             user_node_idx = user_id_to_node_idx[user_id]
    #             item_node_idx = item_id_to_node_idx[item_id] + user_offset
                
    #             edge_list.append([user_node_idx, item_node_idx])
    #             edge_attr_list.append([rating])
                
    #             edge_list.append([item_node_idx, user_node_idx])  # Bidirectional
    #             edge_attr_list.append([rating])  # Same weight
        
    #     # Convert to tensors
    #     if len(edge_list) == 0:  # Handle empty graph case
    #         total_nodes = len(node_features)
    #         # Create a simple self-loop as fallback
    #         edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.dev)
    #         edge_attr = torch.tensor([[1.0]], dtype=torch.float).to(self.dev)
    #     else:
    #         edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.dev)
    #         edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(self.dev)
        
    #     x = torch.cat(node_features, dim=0).squeeze(1) if node_features else torch.zeros((1, self.item_emb.embedding_dim)).to(self.dev)
        
    #     # Enhance graph using random walk
    #     total_nodes = len(node_features) if node_features else 1
    #     print(f"Applying random walks to enhance global graph with {total_nodes} nodes and {len(edge_list)} edges...")
    #     enhanced_edge_index, enhanced_edge_attr = self.random_walk_enhancer.enhance_graph(
    #         edge_index, edge_attr, total_nodes
    #     )
        
    #     # Store the global graph
    #     self.global_x = x
    #     self.global_edge_index = enhanced_edge_index
    #     self.global_edge_attr = enhanced_edge_attr
    #     self.global_user_offset = user_offset
    #     self.global_item_map = item_id_to_node_idx
    #     self.global_user_map = user_id_to_node_idx
    #     self.global_graph_built = True
        
    #     print(f"Global graph built with {total_nodes} nodes and {enhanced_edge_index.shape[1]} edges (after enhancement)")
        
    #     return x, enhanced_edge_index, enhanced_edge_attr, user_offset, item_id_to_node_idx, user_id_to_node_idx
    
    def build_graph(self, user_ids, log_seqs, ratings=None):
        """构建基础用户-物品交互图，随后使用随机游走增强"""
        batch_size = log_seqs.shape[0]
        seq_len = log_seqs.shape[1]
        
        # 创建用户和他们交互的物品之间的边
        edge_list = []
        edge_attr_list = []
        node_features = []
        
        # 首先，添加所有用户
        for i, user_id in enumerate(user_ids):
            if user_id == 0:  # 跳过填充用户
                continue
            node_features.append(self.user_emb(torch.tensor([user_id]).to(self.dev)))
        
        user_offset = len(node_features)
        
        # 然后添加所有物品
        item_id_to_node_idx = {}
        for i in range(batch_size):
            for j in range(seq_len):
                item_id = log_seqs[i, j]
                if item_id == 0:  # 跳过填充物品
                    continue
                
                if item_id not in item_id_to_node_idx:
                    item_id_to_node_idx[item_id] = len(node_features) - user_offset
                    node_features.append(self.item_emb(torch.tensor([item_id]).to(self.dev)))
                
                # 设置边权重（如果有评分，使用评分，否则使用默认权重1.0）
                rating = 1.0
                if ratings is not None:
                    rating = ratings[i, j]
                
                # 添加边：用户 -> 物品 和 物品 -> 用户（双向）
                if user_ids[i] != 0:  # 跳过填充用户
                    user_node_idx = i
                    item_node_idx = item_id_to_node_idx[item_id] + user_offset
                    
                    edge_list.append([user_node_idx, item_node_idx])
                    edge_attr_list.append([rating])
                    
                    edge_list.append([item_node_idx, user_node_idx])  # 双向
                    edge_attr_list.append([rating])  # 相同权重
        
        # 转换为张量
        if len(edge_list) == 0:  # 处理空图的情况
            total_nodes = len(node_features)
            # 创建一个简单的自环作为后备
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.dev)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float).to(self.dev)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.dev)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(self.dev)
        
        x = torch.cat(node_features, dim=0).squeeze(1) if node_features else torch.zeros((1, self.item_emb.embedding_dim)).to(self.dev)
        
        # 使用随机游走增强图
        total_nodes = len(node_features) if node_features else 1
        enhanced_edge_index, enhanced_edge_attr = self.random_walk_enhancer.enhance_graph(
            edge_index, edge_attr, total_nodes
        )
        
        return x, enhanced_edge_index, enhanced_edge_attr, user_offset, item_id_to_node_idx

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        """使用时间感知自注意力处理序列数据（TiSASRec分支）"""
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

        # 时间线和注意力的掩码
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # 掩码填充物品

        tl = seqs.shape[1]  # 时间维度长度，用于强制因果关系
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # 应用自注意力块
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs

            # 逐点前馈网络
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats
    
    def graph2feats(self, user_ids, log_seqs, ratings=None):
        """使用GNN处理图数据（图分支）"""
        # 从用户-物品交互构建图（使用随机游走增强）
        x, edge_index, edge_attr, user_offset, item_map = self.build_graph(user_ids, log_seqs, ratings)
        
        # 应用GNN层
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
        
        # 从图中提取用户和物品嵌入
        batch_size = len(user_ids)
        seq_len = log_seqs.shape[1]
        
        # 初始化物品序列的输出张量
        gnn_feats = torch.zeros((batch_size, seq_len, x.shape[1]), device=self.dev)
        
        # 填充GNN处理后的特征
        for i in range(batch_size):
            for j in range(seq_len):
                item_id = log_seqs[i, j]
                if item_id == 0:  # 跳过填充
                    continue
                
                if item_id in item_map:
                    item_node_idx = item_map[item_id] + user_offset
                    gnn_feats[i, j] = x[item_node_idx]
        
        return gnn_feats
    # def graph2feats(self, user_ids, log_seqs, ratings=None):
    #     """Using GNN to process graph data (Graph branch) - using global graph"""
    #     # Check if global graph is built
    #     if not self.global_graph_built:
    #         # If not called during training/evaluation setup, build on-the-fly
    #         # But note this is less efficient than building once
    #         print("Warning: Global graph not pre-built. Building on-the-fly...")
    #         self.build_global_graph(user_ids, log_seqs, ratings)
        
    #     # Get the static graph data
    #     x = self.global_x.clone()  # Clone to avoid modifying the original data
    #     edge_index = self.global_edge_index
    #     edge_attr = self.global_edge_attr
        
    #     # Apply GNN layers to the global graph
    #     for gnn_layer in self.gnn_layers:
    #         x = gnn_layer(x, edge_index, edge_attr)
        
    #     # Extract representations for the current batch
    #     batch_size = len(user_ids)
    #     seq_len = log_seqs.shape[1]
        
    #     # Initialize output tensor for item sequence features
    #     gnn_feats = torch.zeros((batch_size, seq_len, x.shape[1]), device=self.dev)
        
    #     # Fill with the GNN-processed features
    #     for i in range(batch_size):
    #         # Get user representation if available
    #         user_id = user_ids[i]
            
    #         for j in range(seq_len):
    #             item_id = log_seqs[i, j]
    #             if item_id == 0:  # Skip padding
    #                 continue
                
    #             if item_id in self.global_item_map:
    #                 item_node_idx = self.global_item_map[item_id] + self.global_user_offset
    #                 gnn_feats[i, j] = x[item_node_idx]
        
    #     return gnn_feats
    
    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs, ratings=None):
        """训练的前向传播"""
        # 从TiSASRec分支获取序列特征
        seq_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        
        # 从GNN分支获取图特征
        graph_feats = self.graph2feats(user_ids, log_seqs, ratings)
        
        # 融合门组合两种特征
        # 连接特征并应用融合门
        combined_feats = torch.cat([seq_feats, graph_feats], dim=-1)
        fusion_weights = torch.sigmoid(self.fusion_gate(combined_feats))
        
        # 序列和图特征的加权组合
        log_feats = fusion_weights * seq_feats + (1 - fusion_weights) * graph_feats
        log_feats = self.fusion_norm(log_feats)
        
        # 计算正样本和负样本的logits
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
        # 使用log_feats[:, :-1, :] 来匹配 pos_seqs 的维度
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
    
#Utility function for training the model
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
# def train_tisasgnn(model, train_data, optimizer, batch_size, args):
#     """
#     Training utility function with proper gradient handling
#     """
#     model.train()
#     total_loss = 0
    
#     # Get training data
#     users, seqs, times, ratings, labels = train_data
    
#     # Build global graph once before training
#     if not model.global_graph_built:
#         print("Building global graph for training...")
#         model.build_global_graph(users, seqs, ratings)
    
#     num_batches = len(users) // batch_size + (0 if len(users) % batch_size == 0 else 1)
    
#     for i in range(num_batches):
#         # Clear any previous gradients
#         optimizer.zero_grad()
        
#         # Get batch data
#         batch_users = users[i*batch_size:(i+1)*batch_size]
#         batch_seqs = seqs[i*batch_size:(i+1)*batch_size]
#         batch_times = times[i*batch_size:(i+1)*batch_size]
#         batch_ratings = ratings[i*batch_size:(i+1)*batch_size]
        
#         # For each sequence, generate positive and negative samples
#         pos_seqs = batch_seqs[:, 1:]  # Shifted sequence for next item prediction
#         neg_seqs = np.random.randint(1, model.item_num+1, pos_seqs.shape)  # Random negative sampling
        
#         # Forward pass
#         pos_logits, neg_logits = model(batch_users, batch_seqs, batch_times, pos_seqs, neg_seqs, batch_ratings)
        
#         # BPR loss
#         loss = -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits)))
        
#         # L2 regularization if specified
#         if args.l2_emb > 0:
#             l2_loss = 0
#             for name, param in model.named_parameters():
#                 if 'emb' in name:
#                     l2_loss += torch.sum(param ** 2)
#             loss += args.l2_emb * l2_loss
            
#         # Backward pass and optimize
#         loss.backward(retain_graph=True)
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     return total_loss / num_batches



# def refresh_global_graph_embeddings(model):
#     """
#     Call this periodically to update node features in the global graph after embedding updates
#     """
#     if not model.global_graph_built:
#         print("Global graph not built yet, cannot refresh embeddings")
#         return
        
#     # Rebuild node features with updated embeddings
#     node_features = []
    
#     # Add user embeddings with updated parameters
#     for user_id, idx in model.global_user_map.items():
#         node_features.append(model.user_emb(torch.tensor([user_id]).to(model.dev)))
    
#     # Add item embeddings with updated parameters
#     for item_id, idx in model.global_item_map.items():
#         node_features.append(model.item_emb(torch.tensor([item_id]).to(model.dev)))
    
#     # Update global_x with fresh embeddings
#     model.global_x = torch.cat(node_features, dim=0).squeeze(1)
    
#     print("Global graph embeddings refreshed")