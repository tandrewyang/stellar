import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = th.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = th.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerQMixer(nn.Module):
    """
    基于Transformer架构的QMIX Mixer网络
    
    核心改进：
    1. 使用多头自注意力机制捕捉状态中的关键信息
    2. 使用Transformer编码器层增强状态表示
    3. 在混合Q值时考虑智能体间的相互影响
    """
    def __init__(self, args):
        super(TransformerQMixer, self).__init__()
        
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        
        # Transformer参数
        self.embed_dim = getattr(args, 'mixing_embed_dim', 64)
        self.n_heads = getattr(args, 'transformer_heads', 4)
        self.n_layers = getattr(args, 'transformer_layers', 2)
        self.d_ff = getattr(args, 'transformer_ff_dim', 256)
        self.dropout = getattr(args, 'transformer_dropout', 0.1)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 智能体Q值编码
        self.agent_q_encoder = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 混合网络
        self.mixing_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
        
        # V(s) 用于状态相关的bias
        self.V = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
        
    def forward(self, agent_qs, states, dropout=False):
        """
        Args:
            agent_qs: [batch_size, seq_len, n_agents] 智能体Q值
            states: [batch_size, seq_len, state_dim] 或 [batch_size, seq_len, n_agents, state_dim] (dropout=True)
        Returns:
            q_tot: [batch_size, seq_len, 1] 总Q值
        """
        # 处理dropout情况下的states形状（与原始QMIX保持一致）
        if dropout:
            # 当dropout=True时，states需要扩展为 [bs, seq_len, n_agents, state_dim]
            if len(states.shape) == 3:
                states = states.reshape(states.shape[0], states.shape[1], 1, states.shape[2]).repeat(1, 1, self.n_agents, 1)
        
        # 获取实际batch size和seq_len
        if len(states.shape) == 4:
            # dropout模式: [bs, seq_len, n_agents, state_dim]
            bs, seq_len, n_agents_state, state_dim = states.shape
            states = states.permute(0, 2, 1, 3).reshape(bs * n_agents_state, seq_len, state_dim)
            bs = bs * n_agents_state
            states_for_encoding = states
        else:
            # 正常模式: [bs, seq_len, state_dim]
            bs = states.size(0)
            seq_len = states.size(1)
            states_for_encoding = states
        
        # 编码状态
        states_flat = states_for_encoding.reshape(-1, self.state_dim)
        state_emb = self.state_encoder(states_flat)  # [bs*seq_len, embed_dim]
        state_emb = state_emb.view(bs, seq_len, self.embed_dim)
        
        # 编码智能体Q值
        agent_qs_flat = agent_qs.reshape(-1, self.n_agents, 1)
        agent_q_emb = self.agent_q_encoder(agent_qs_flat)  # [bs*seq_len, n_agents, embed_dim]
        agent_q_emb = agent_q_emb.view(bs, seq_len, self.n_agents, self.embed_dim)
        
        # 将智能体Q值嵌入与状态嵌入结合
        # 为每个智能体添加状态信息
        state_emb_expanded = state_emb.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        combined_emb = agent_q_emb + state_emb_expanded  # [bs, seq_len, n_agents, embed_dim]
        
        # 重塑为序列格式用于Transformer
        combined_emb_flat = combined_emb.reshape(bs * seq_len, self.n_agents, self.embed_dim)
        
        # 通过Transformer编码器
        transformer_out = combined_emb_flat
        for layer in self.transformer_layers:
            transformer_out = layer(transformer_out)
        
        # 聚合智能体信息（使用平均池化或注意力加权）
        aggregated = th.mean(transformer_out, dim=1)  # [bs*seq_len, embed_dim]
        
        # 通过混合网络
        mixed = self.mixing_net(aggregated)  # [bs*seq_len, 1]
        
        # 添加状态相关的bias
        state_emb_flat = state_emb.reshape(bs * seq_len, self.embed_dim)
        v = self.V(state_emb_flat)  # [bs*seq_len, 1]
        
        q_tot = (mixed + v).view(bs, seq_len, 1)
        
        return q_tot
    
    def k(self, states):
        """计算智能体权重（用于分析）"""
        bs = states.size(0)
        states_flat = states.reshape(-1, self.state_dim)
        state_emb = self.state_encoder(states_flat)
        state_emb = state_emb.view(bs, -1, self.embed_dim)
        
        # 简化版本：基于状态嵌入计算权重
        weights = th.softmax(th.sum(state_emb, dim=-1), dim=-1)
        return weights.unsqueeze(-1).expand(-1, -1, self.n_agents)
    
    def b(self, states):
        """计算状态相关的bias"""
        bs = states.size(0)
        states_flat = states.reshape(-1, self.state_dim)
        state_emb = self.state_encoder(states_flat)
        state_emb = state_emb.view(bs, -1, self.embed_dim)
        
        state_emb_flat = state_emb.reshape(bs * state_emb.size(1), self.embed_dim)
        b = self.V(state_emb_flat)
        return b.view(bs, -1, 1)

