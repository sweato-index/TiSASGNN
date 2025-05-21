import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class TimeAwareGNNLayer(MessagePassing):
    """时间感知的图神经网络层"""
    def __init__(self, hidden_size, time_span):
        super().__init__(aggr='mean')
        self.hidden_size = hidden_size
        self.time_span = time_span
        
        # 时间间隔嵌入层
        self.time_emb = nn.Embedding(time_span+1, hidden_size)
        
        # 消息传递网络
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # 更新网络
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_time):
        # 输入维度检查
        assert x.dim() == 3, f"Input x should be 3D [batch, seq, hidden], got {x.shape}"
        assert edge_index.dim() == 2, f"edge_index should be 2D, got {edge_index.shape}"
        assert edge_time.dim() == 1, f"edge_time should be 1D, got {edge_time.shape}"

        # 标准化时间间隔
        edge_time = torch.clamp(edge_time, 0, self.time_span)
        
        # 调整输入维度 [batch, seq, hidden] -> [batch * seq, hidden]
        batch_size, seq_len, hidden_size = x.shape
        x = x.reshape(-1, hidden_size)
        
        # 传播前维度检查
        assert x.dim() == 2, f"Reshaped x should be 2D, got {x.shape}"
        
        # 执行消息传递
        out = self.propagate(edge_index, x=x, edge_time=edge_time)
        
        # 恢复原始维度 [batch * seq, hidden] -> [batch, seq, hidden]
        out = out.reshape(batch_size, seq_len, hidden_size)
        
        return out

    def message(self, x_j, x_i, edge_time):
        # x_j: 源节点特征 [batch * edges, hidden]
        # edge_time: 时间间隔 [edges]
        
        # 获取时间特征并确保正确维度 [edges, hidden]
        time_feat = self.time_emb(edge_time).squeeze()
        if len(time_feat.shape) == 1:
            time_feat = time_feat.unsqueeze(-1)
        
        # 确保x_j是2D [batch * edges, hidden]
        x_j = x_j.view(-1, x_j.size(-1))
        
        # 维度检查
        assert x_j.dim() == 2, f"x_j should be 2D, got {x_j.dim()}"
        assert time_feat.dim() == 2, f"time_feat should be 2D, got {time_feat.dim()}"
        
        # 拼接源节点特征和时间特征
        msg_input = torch.cat([x_j, time_feat], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        # 拼接聚合结果和原始特征
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)

class TiSASGNN(nn.Module):
    """时间感知的序列推荐GNN模型"""
    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device
        self.time_span = args.time_span
        
        # 基础嵌入层
        self.user_emb = nn.Embedding(user_num+1, args.hidden_units, padding_idx=0)
        self.item_emb = nn.Embedding(item_num+1, args.hidden_units, padding_idx=0)
        
        # 时间相关嵌入
        self.abs_pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_emb = nn.Embedding(args.time_span+1, args.hidden_units)
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            TimeAwareGNNLayer(args.hidden_units, args.time_span)
            for _ in range(args.num_blocks)
        ])
        
        # 预测层
        self.predictor = nn.Linear(args.hidden_units, 1)
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass

    def build_temporal_graph(self, seq, time_matrix):
        """构建时序图结构"""
        # 生成节点索引
        seq_len = seq.size(1)
        nodes = torch.arange(seq_len, device=self.device)
        
        # 创建边(考虑时序因果关系)
        src = nodes[:-1].repeat_interleave(seq_len - 1 - nodes[:-1])
        dst = torch.cat([nodes[i+1:] for i in range(seq_len-1)])
        
        # 从时间矩阵提取边对应的时间差
        batch_size = time_matrix.size(0)
        edge_time = time_matrix[torch.arange(batch_size).unsqueeze(1), src, dst]
        edge_time = edge_time.flatten()
        edge_time = torch.clamp(edge_time, 0, self.time_span)
        
        return torch.stack([src, dst]), edge_time

    def forward(self, user_ids, seq, time_matrix, pos=None, neg=None):
        """模型前向传播"""
        # 转换输入数据类型
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.LongTensor(user_ids).to(self.device)
        if not isinstance(seq, torch.Tensor):
            seq = torch.LongTensor(seq).to(self.device)
        if not isinstance(time_matrix, torch.Tensor):
            time_matrix = torch.LongTensor(time_matrix).to(self.device)
        
        # 获取嵌入
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(seq)
        
        # 添加绝对位置信息
        positions = torch.arange(seq.size(1), device=self.device).expand_as(seq)
        pos_emb = self.abs_pos_emb(positions)
        item_emb = item_emb + pos_emb
        
        # 构建时序图
        edge_index, edge_time = self.build_temporal_graph(seq, time_matrix)
        
        # GNN消息传递
        for layer in self.gnn_layers:
            item_emb = layer(item_emb, edge_index, edge_time)
        
        # 取序列最后一个节点作为用户当前兴趣
        user_curr_emb = item_emb[:, -1, :]
        
        # 计算预测得分
        if pos is not None and neg is not None:
            if not isinstance(pos, torch.Tensor):
                pos = torch.LongTensor(pos).to(self.device)
            if not isinstance(neg, torch.Tensor):
                neg = torch.LongTensor(neg).to(self.device)
            pos_emb = self.item_emb(pos)
            neg_emb = self.item_emb(neg)
            pos_logits = (user_curr_emb * pos_emb).sum(-1)
            neg_logits = (user_curr_emb * neg_emb).sum(-1)
            return pos_logits, neg_logits
        else:
            return user_curr_emb

    def predict(self, user_ids, seq, time_matrix, item_indices):
        """预测接口"""
        user_emb = self.forward(user_ids, seq, time_matrix)
        item_emb = self.item_emb(item_indices)
        return (user_emb * item_emb).sum(-1)
