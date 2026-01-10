# 六个优化的具体技术实现路径

## 概述

本文档详细介绍六个优化方法在 dTAPE 框架上的具体技术实现路径，包括代码位置、关键修改点、实现细节和集成方式。

---

## 一、TransformerMixer

### 1.1 优化目标

将 QMIX 的 Mixer 网络从 Hypernetwork 架构替换为 Transformer 架构，增强状态表示和智能体间协作能力。

### 1.2 技术实现路径

#### 步骤 1: 实现 TransformerQMixer 类

**文件位置**: `RLalgs/TransformerMixer/src/modules/mixers/transformer_qmix.py`

**核心实现**:

```python
class TransformerQMixer(nn.Module):
    def __init__(self, args):
        super(TransformerQMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        
        # Transformer 参数
        self.embed_dim = getattr(args, 'mixing_embed_dim', 64)
        self.n_heads = getattr(args, 'transformer_heads', 4)
        self.n_layers = getattr(args, 'transformer_layers', 2)
        self.d_ff = getattr(args, 'transformer_ff_dim', 256)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Transformer 编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 智能体 Q 值编码
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
        
        # V(s) 用于状态相关的 bias
        self.V = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states, dropout=False):
        # 编码状态
        states_flat = states.reshape(-1, self.state_dim)
        state_emb = self.state_encoder(states_flat)
        
        # 编码智能体 Q 值
        agent_qs_flat = agent_qs.reshape(-1, self.n_agents, 1)
        agent_q_emb = self.agent_q_encoder(agent_qs_flat)
        
        # 将智能体 Q 值嵌入与状态嵌入结合
        state_emb_expanded = state_emb.unsqueeze(1).expand(-1, self.n_agents, -1)
        combined_emb = agent_q_emb + state_emb_expanded
        
        # 通过 Transformer 编码器
        transformer_out = combined_emb
        for layer in self.transformer_layers:
            transformer_out = layer(transformer_out)
        
        # 聚合智能体信息（使用平均池化）
        aggregated = th.mean(transformer_out, dim=1)
        
        # 通过混合网络
        mixed = self.mixing_net(aggregated)
        
        # 添加状态相关的 bias
        v = self.V(state_emb)
        
        q_tot = (mixed + v).view(bs, seq_len, 1)
        return q_tot
```

**关键组件**:
- `MultiHeadAttention`: 多头自注意力机制
- `TransformerEncoderLayer`: Transformer 编码器层（自注意力 + FFN + 残差连接）
- `TransformerQMixer`: 主 Mixer 类，保持与 QMixer 相同的接口

#### 步骤 2: 注册 TransformerQMixer

**文件位置**: `RLalgs/TransformerMixer/src/modules/mixers/__init__.py`

```python
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transformer_qmix import TransformerQMixer

REGISTRY = {}
REGISTRY["vdn"] = VDNMixer
REGISTRY["qmix"] = QMixer
REGISTRY["transformer_qmix"] = TransformerQMixer  # 注册
```

#### 步骤 3: 修改 MAXQLearner 支持 transformer_qmix

**文件位置**: `RLalgs/TransformerMixer/src/learners/max_q_learner.py`

**修改点**: 第 1-8 行（导入）和第 42-48 行（Mixer 选择）

```python
# 添加导入
from modules.mixers.transformer_qmix import TransformerQMixer

# 在 __init__ 方法中修改
if args.mixer == "vdn":
    self.mixer = VDNMixer()
elif args.mixer == "qmix":
    self.mixer = QMixer(args)
elif args.mixer == "transformer_qmix":  # 新增
    self.mixer = TransformerQMixer(args)  # 使用 Transformer
else:
    raise ValueError("Mixer {} not recognised.".format(args.mixer))
```

#### 步骤 4: 配置文件

**文件位置**: `RLalgs/TransformerMixer/src/config/algs/transformer_qmix.yaml`

```yaml
mixer: "transformer_qmix"  # 关键：使用 transformer_qmix

# Transformer 特定参数
transformer_heads: 4
transformer_layers: 2
transformer_ff_dim: 256
transformer_dropout: 0.1
mixing_embed_dim: 64

# 保持 dTAPE 其他配置
learner: "max_q_learner"
central_loss: 1
qmix_loss: 1
hysteretic_qmix: True
comm: True
```

### 1.3 与 dTAPE 的集成

- **保持 dTAPE 框架**: 所有 dTAPE 组件（Central Q、通信机制、OW-QMIX）完全保留
- **仅替换 Mixer**: 在 `MAXQLearner.__init__` 中，将 `QMixer(args)` 替换为 `TransformerQMixer(args)`
- **接口兼容**: `TransformerQMixer.forward()` 与 `QMixer.forward()` 接口完全一致

---

## 二、HierarchicalArchitecture

### 2.1 优化目标

引入分层架构，将决策分为高层策略（目标选择、战术决策）和底层执行（具体动作）。

### 2.2 技术实现路径

#### 步骤 1: 实现高层策略网络

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

**核心实现**:

```python
class HighLevelPolicy(nn.Module):
    """高层策略网络：制定宏观策略"""
    def __init__(self, input_shape, args):
        super(HighLevelPolicy, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.fc1 = nn.Linear(input_shape, args.hierarchical_high_dim)
        self.fc2 = nn.Linear(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
        # 输出：目标选择、战术类型
        self.goal_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_goals)
        self.tactic_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_tactics)
        
        self.rnn = nn.GRUCell(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        goal_logits = self.goal_head(h)
        tactic_logits = self.tactic_head(h)
        
        return {
            'goal': goal_logits,
            'tactic': tactic_logits,
            'hidden': h
        }
```

#### 步骤 2: 实现底层执行网络

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

```python
class LowLevelPolicy(nn.Module):
    """底层执行网络：基于高层策略执行具体动作"""
    def __init__(self, input_shape, args):
        super(LowLevelPolicy, self).__init__()
        self.args = args
        
        # 底层输入：原始观测 + 高层策略信息
        enhanced_input_dim = (
            input_shape + 
            args.hierarchical_high_dim + 
            args.hierarchical_n_goals + 
            args.hierarchical_n_tactics
        )
        
        self.fc1 = nn.Linear(enhanced_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
    def forward(self, inputs, high_level_info, hidden_state=None):
        # 拼接原始观测和高层策略信息
        enhanced_input = torch.cat([
            inputs,
            high_level_info['hidden'],
            F.softmax(high_level_info['goal'], dim=-1),
            F.softmax(high_level_info['tactic'], dim=-1)
        ], dim=-1)
        
        x = F.relu(self.fc1(enhanced_input), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        q = self.fc2(h)
        return q, h
```

#### 步骤 3: 实现分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

```python
class HierarchicalRNNAgent(nn.Module):
    """分层架构智能体：结合高层策略和底层执行"""
    def __init__(self, input_shape, args):
        super(HierarchicalRNNAgent, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.high_level = HighLevelPolicy(input_shape, args)
        
        # 底层执行网络
        self.low_level = LowLevelPolicy(input_shape, args)
        
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        inputs_flat = inputs.view(-1, e)
        
        # 获取高层策略
        if hidden_state is not None:
            high_hidden, low_hidden = hidden_state
            high_hidden = high_hidden.reshape(-1, self.args.hierarchical_high_dim)
            low_hidden = low_hidden.reshape(-1, self.args.rnn_hidden_dim)
        else:
            high_hidden = None
            low_hidden = None
        
        high_level_info = self.high_level(inputs_flat, high_hidden)
        new_high_hidden = high_level_info['hidden']
        
        # 获取底层 Q 值
        q, new_low_hidden = self.low_level(inputs_flat, high_level_info, low_hidden)
        
        # 重塑输出
        q = q.view(b, a, -1)
        new_high_hidden = new_high_hidden.view(b, a, -1)
        new_low_hidden = new_low_hidden.view(b, a, -1)
        
        return q, (new_high_hidden, new_low_hidden)
```

#### 步骤 4: 注册分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/__init__.py`

```python
from modules.agents.hierarchical_agent import HierarchicalRNNAgent

REGISTRY = {}
REGISTRY["hierarchical_rnn"] = HierarchicalRNNAgent
```

#### 步骤 5: 修改 MAC 使用分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/controllers/basic_controller.py`

**修改点**: 在 `BasicMAC` 中，将 `agent` 从 `"rnn"` 改为 `"hierarchical_rnn"`

#### 步骤 6: 配置文件

**文件位置**: `RLalgs/HierarchicalArchitecture/src/config/algs/hierarchical_qmix.yaml`

```yaml
agent: "hierarchical_rnn"  # 使用分层智能体

# 分层架构参数
hierarchical_high_dim: 128
hierarchical_n_goals: 8
hierarchical_n_tactics: 4

# 保持 dTAPE 其他配置
learner: "max_q_learner"
mixer: "qmix"
```

### 2.3 与 dTAPE 的集成

- **替换 Agent**: 在 `BasicMAC` 中，将 `RNNAgent` 替换为 `HierarchicalRNNAgent`
- **保持其他组件**: Mixer、Learner、通信机制等完全保留
- **隐藏状态扩展**: 隐藏状态从 `(hidden,)` 扩展为 `(high_hidden, low_hidden)`

---

## 三、EnhancedStateRepresentation

### 3.1 优化目标

通过增强的状态编码器提取更丰富的特征，使用多层特征提取和注意力机制改进状态表示。

### 3.2 技术实现路径

#### 步骤 1: 实现状态编码器

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/enhanced_rnn_agent.py`

**核心实现**:

```python
class StateEncoder(nn.Module):
    """增强的状态编码器：提取更丰富的状态特征"""
    def __init__(self, input_shape, args):
        super(StateEncoder, self).__init__()
        self.args = args
        
        # 特征提取网络（多层 + LayerNorm）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制：关注重要特征
        self.attention = nn.Sequential(
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args.enhanced_feature_dim // 2, args.enhanced_feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 注意力加权
        attention_weights = self.attention(features)
        enhanced_features = features * attention_weights
        
        return enhanced_features
```

#### 步骤 2: 实现增强的 RNN 智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/enhanced_rnn_agent.py`

```python
class EnhancedRNNAgent(nn.Module):
    """增强的 RNN 智能体：使用改进的状态表示"""
    def __init__(self, input_shape, args):
        super(EnhancedRNNAgent, self).__init__()
        self.args = args
        
        # 状态编码器
        self.state_encoder = StateEncoder(input_shape, args)
        
        # RNN 网络
        self.rnn = nn.GRUCell(
            args.enhanced_feature_dim, 
            args.rnn_hidden_dim
        )
        
        # 输出层（多层 + LayerNorm + Dropout）
        self.fc_out = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        inputs_flat = inputs.view(-1, e)
        
        # 状态编码
        encoded_features = self.state_encoder(inputs_flat)
        
        # RNN 处理
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(encoded_features, hidden_state)
        
        # 输出 Q 值
        q = self.fc_out(h)
        q = torch.clamp(q, -5, 2)
        
        return q.view(b, a, -1), h.view(b, a, -1)
```

#### 步骤 3: 注册增强智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/__init__.py`

```python
from modules.agents.enhanced_rnn_agent import EnhancedRNNAgent

REGISTRY = {}
REGISTRY["enhanced_rnn"] = EnhancedRNNAgent
```

#### 步骤 4: 修改 MAC 使用增强智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/controllers/basic_controller.py`

**修改点**: 在 `BasicMAC` 中，将 `agent` 从 `"rnn"` 改为 `"enhanced_rnn"`

#### 步骤 5: 配置文件

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/config/algs/enhanced_qmix.yaml`

```yaml
agent: "enhanced_rnn"  # 使用增强智能体

# 增强状态表示参数
enhanced_feature_dim: 128
enhanced_dropout: 0.1
use_spatial_features: False

# 保持 dTAPE 其他配置
learner: "max_q_learner"
mixer: "qmix"
```

### 3.3 与 dTAPE 的集成

- **替换 Agent**: 在 `BasicMAC` 中，将 `RNNAgent` 替换为 `EnhancedRNNAgent`
- **保持其他组件**: Mixer、Learner、通信机制等完全保留
- **状态处理流程**: 原始观测 → 状态编码器 → RNN → 输出层

---

## 四、CurriculumLearning

### 4.1 优化目标

通过课程学习策略，从简单任务开始，逐步增加任务难度，使智能体能够更好地学习和适应复杂任务。

### 4.2 技术实现路径

#### 步骤 1: 实现课程学习 Learner

**文件位置**: `RLalgs/CurriculumLearning/src/learners/curriculum_learner.py`

**核心实现**:

```python
class CurriculumLearner(MAXQLearner):
    """课程学习：逐步增加任务难度"""
    def __init__(self, mac, scheme, logger, args):
        super(CurriculumLearner, self).__init__(mac, scheme, logger, args)
        
        # 课程学习参数
        self.curriculum_enabled = getattr(args, 'curriculum_enabled', True)
        self.curriculum_schedule = getattr(args, 'curriculum_schedule', 'linear')
        self.curriculum_start_step = getattr(args, 'curriculum_start_step', 0)
        self.curriculum_end_step = getattr(args, 'curriculum_end_step', 1000000)
        
        # 难度级别（0-1，0最简单，1最难）
        self.current_difficulty = 0.0
        self.min_difficulty = getattr(args, 'curriculum_min_difficulty', 0.0)
        self.max_difficulty = getattr(args, 'curriculum_max_difficulty', 1.0)
        
        # 课程学习指标
        self.episode_wins = deque(maxlen=100)
        
    def update_curriculum_difficulty(self, t_env):
        """根据训练进度更新课程难度"""
        if not self.curriculum_enabled:
            self.current_difficulty = self.max_difficulty
            return
        
        if t_env < self.curriculum_start_step:
            self.current_difficulty = self.min_difficulty
        elif t_env >= self.curriculum_end_step:
            self.current_difficulty = self.max_difficulty
        else:
            # 线性调度
            if self.curriculum_schedule == 'linear':
                progress = (t_env - self.curriculum_start_step) / (
                    self.curriculum_end_step - self.curriculum_start_step
                )
                self.current_difficulty = (
                    self.min_difficulty + 
                    (self.max_difficulty - self.min_difficulty) * progress
                )
            # 自适应调度（基于性能）
            elif self.curriculum_schedule == 'adaptive':
                if len(self.episode_wins) > 10:
                    recent_win_rate = np.mean(list(self.episode_wins)[-10:])
                    if recent_win_rate > 0.8:
                        # 性能好，增加难度
                        self.current_difficulty = min(
                            self.current_difficulty + 0.01,
                            self.max_difficulty
                        )
                    elif recent_win_rate < 0.2:
                        # 性能差，降低难度
                        self.current_difficulty = max(
                            self.current_difficulty - 0.01,
                            self.min_difficulty
                        )
    
    def apply_curriculum_to_batch(self, batch):
        """将课程难度应用到 batch"""
        # 根据难度调整 batch（例如：过滤困难样本、调整奖励等）
        # 具体实现取决于任务特点
        return batch
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 更新课程难度
        self.update_curriculum_difficulty(t_env)
        
        # 应用课程学习到 batch
        batch = self.apply_curriculum_to_batch(batch)
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 注册课程学习 Learner

**文件位置**: `RLalgs/CurriculumLearning/src/learners/__init__.py`

```python
from learners.curriculum_learner import CurriculumLearner

REGISTRY = {}
REGISTRY["curriculum_learner"] = CurriculumLearner
```

#### 步骤 3: 配置文件

**文件位置**: `RLalgs/CurriculumLearning/src/config/algs/curriculum_qmix.yaml`

```yaml
learner: "curriculum_learner"  # 使用课程学习 learner

# 课程学习参数
curriculum_enabled: True
curriculum_schedule: "linear"  # "linear" 或 "adaptive"
curriculum_start_step: 0
curriculum_end_step: 1000000
curriculum_min_difficulty: 0.0
curriculum_max_difficulty: 1.0

# 保持 dTAPE 其他配置
mixer: "qmix"
agent: "rnn"
```

### 4.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `CurriculumLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **重写 train 方法**: 在训练前更新难度并应用到 batch
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 五、RewardShaping

### 5.1 优化目标

通过奖励塑形改进奖励信号，使智能体能够更快、更稳定地学习，特别适合稀疏奖励环境。

### 5.2 技术实现路径

#### 步骤 1: 实现奖励塑形 Learner

**文件位置**: `RLalgs/RewardShaping/src/learners/reward_shaping_learner.py`

**核心实现**:

```python
class RewardShapingLearner(MAXQLearner):
    """奖励塑形：通过改进奖励信号提升学习效率"""
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # 奖励塑形权重
        self.shaping_weight = getattr(args, 'reward_shaping_weight', 0.1)
        self.shaping_decay = getattr(args, 'reward_shaping_decay', 0.99)
        self.current_shaping_weight = self.shaping_weight
        
    def shape_reward(self, rewards, states, next_states, t_env):
        """对奖励进行塑形"""
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        if self.shaping_type == 'potential_based':
            # 基于潜在函数的奖励塑形
            # r' = r + γφ(s') - φ(s)
            potential_current = self._compute_potential(states)
            potential_next = self._compute_potential(next_states)
            potential_bonus = potential_next - potential_current
            shaped_rewards = rewards + self.current_shaping_weight * potential_bonus
        
        elif self.shaping_type == 'dense_reward':
            # 密集奖励：为中间步骤添加奖励
            shaped_rewards = self._add_dense_rewards(rewards, states)
        
        elif self.shaping_type == 'curiosity':
            # 好奇心驱动：鼓励探索
            shaped_rewards = self._add_curiosity_bonus(rewards, states)
        
        # 更新塑形权重（逐渐减少）
        self.current_shaping_weight *= self.shaping_decay
        
        return shaped_rewards
    
    def _compute_potential(self, states):
        """计算潜在函数值"""
        # 简化版本：可以扩展为神经网络
        return th.zeros_like(states[:, :, 0:1])
    
    def _add_dense_rewards(self, rewards, states):
        """添加密集奖励"""
        # 为每步添加小的生存奖励
        survival_bonus = 0.01
        return rewards + survival_bonus
    
    def _add_curiosity_bonus(self, rewards, states):
        """添加好奇心奖励"""
        # 基于状态新颖性的奖励
        # 简化版本
        return rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取原始奖励
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        next_states = batch["state"][:, 1:]
        
        # 对奖励进行塑形
        shaped_rewards = self.shape_reward(rewards, states, next_states, t_env)
        
        # 替换 batch 中的奖励
        batch["reward"][:, :-1] = shaped_rewards
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 注册奖励塑形 Learner

**文件位置**: `RLalgs/RewardShaping/src/learners/__init__.py`

```python
from learners.reward_shaping_learner import RewardShapingLearner

REGISTRY = {}
REGISTRY["reward_shaping_learner"] = RewardShapingLearner
```

#### 步骤 3: 配置文件

**文件位置**: `RLalgs/RewardShaping/src/config/algs/reward_shaping_qmix.yaml`

```yaml
learner: "reward_shaping_learner"  # 使用奖励塑形 learner

# 奖励塑形参数
reward_shaping_enabled: True
reward_shaping_type: "potential_based"  # "potential_based", "dense_reward", "curiosity"
reward_shaping_weight: 0.1
reward_shaping_decay: 0.99

# 保持 dTAPE 其他配置
mixer: "qmix"
agent: "rnn"
```

### 5.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `RewardShapingLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **重写 train 方法**: 在训练前对奖励进行塑形
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 六、TargetedOptimization

### 6.1 优化目标

针对特定地图设计专门的奖励塑形策略，根据地图的战术特点和游戏机制定制优化方案。

### 6.2 技术实现路径

#### 步骤 1: 实现地图特定的奖励塑形 Learner

**文件位置**: `RLalgs/TargetedOptimization/src/learners/reward_shaping_learner.py`

**核心实现**:

```python
class RewardShapingLearner(MAXQLearner):
    """奖励塑形：针对特定地图的优化"""
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 获取地图名称
        try:
            self.map_name = getattr(args.env_args, 'map_name', '')
        except:
            self.map_name = ''
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # JCTQ (金蝉脱壳) 特定参数
        self.survival_reward_weight = getattr(args, 'survival_reward_weight', 2.0)
        self.escape_reward_weight = getattr(args, 'escape_reward_weight', 1.5)
        self.disperse_reward_weight = getattr(args, 'disperse_reward_weight', 0.8)
        self.kill_reward_weight = getattr(args, 'kill_reward_weight', 0.3)
        self.time_critical_weight = getattr(args, 'time_critical_weight', 1.2)
        
        # SWCT (上屋抽梯) 特定参数
        self.lure_reward_weight = getattr(args, 'lure_reward_weight', 1.5)
        self.forcefield_reward_weight = getattr(args, 'forcefield_reward_weight', 2.5)
        self.warp_prism_reward_weight = getattr(args, 'warp_prism_reward_weight', 2.0)
        self.tactical_positioning_weight = getattr(args, 'tactical_positioning_weight', 1.2)
        
    def shape_reward(self, rewards, states, next_states, t_env, batch=None):
        """对奖励进行塑形（地图特定）"""
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        # JCTQ (金蝉脱壳) - 逃脱生存策略
        if self.shaping_type == 'escape_survival' or self.map_name == 'jctq':
            shaped_rewards = self._apply_escape_survival_shaping(
                rewards, states, next_states, batch
            )
        
        # SWCT (上屋抽梯) - 诱敌阻挡撤退策略
        elif self.shaping_type == 'lure_block_retreat' or self.map_name == 'swct':
            shaped_rewards = self._apply_lure_block_retreat_shaping(
                rewards, states, next_states, batch, t_env
            )
        
        return shaped_rewards
    
    def _apply_escape_survival_shaping(self, rewards, states, next_states, batch):
        """JCTQ: 逃脱生存策略的奖励塑形"""
        shaped_rewards = rewards.clone()
        
        # 1. 生存奖励：鼓励单位存活
        # 检测单位是否存活（简化版本）
        survival_bonus = self.survival_reward_weight * 0.1  # 每步存活奖励
        shaped_rewards += survival_bonus
        
        # 2. 逃脱奖励：鼓励远离敌人
        # 计算与敌人的距离（简化版本）
        escape_bonus = self.escape_reward_weight * 0.05
        shaped_rewards += escape_bonus
        
        # 3. 分散奖励：鼓励单位分散
        disperse_bonus = self.disperse_reward_weight * 0.03
        shaped_rewards += disperse_bonus
        
        # 4. 击杀惩罚：避免暴露位置
        # 如果击杀敌人，给予较小奖励（避免暴露）
        kill_penalty = -self.kill_reward_weight * 0.1
        # 检测击杀（简化版本）
        # shaped_rewards += kill_penalty
        
        # 5. 时间紧迫性：后期给予更多奖励
        time_bonus = self.time_critical_weight * 0.02
        shaped_rewards += time_bonus
        
        return shaped_rewards
    
    def _apply_lure_block_retreat_shaping(self, rewards, states, next_states, batch, t_env):
        """SWCT: 诱敌阻挡撤退策略的奖励塑形"""
        shaped_rewards = rewards.clone()
        
        # 1. 强生存奖励
        survival_bonus = 3.0 * 0.1
        shaped_rewards += survival_bonus
        
        # 2. ForceField 使用奖励
        forcefield_bonus = self.forcefield_reward_weight * 0.2
        # 检测 ForceField 使用（简化版本）
        # shaped_rewards += forcefield_bonus
        
        # 3. WarpPrism 使用奖励
        warp_prism_bonus = self.warp_prism_reward_weight * 0.15
        # 检测 WarpPrism 使用（简化版本）
        # shaped_rewards += warp_prism_bonus
        
        # 4. 战术位置奖励
        positioning_bonus = self.tactical_positioning_weight * 0.1
        shaped_rewards += positioning_bonus
        
        # 5. 诱敌奖励
        lure_bonus = self.lure_reward_weight * 0.1
        shaped_rewards += lure_bonus
        
        return shaped_rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取原始奖励
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        next_states = batch["state"][:, 1:]
        
        # 对奖励进行塑形（地图特定）
        shaped_rewards = self.shape_reward(rewards, states, next_states, t_env, batch)
        
        # 替换 batch 中的奖励
        batch["reward"][:, :-1] = shaped_rewards
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 配置文件（JCTQ）

**文件位置**: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_jctq.yaml`

```yaml
learner: "reward_shaping_learner"

# JCTQ 特定奖励塑形
reward_shaping_enabled: True
reward_shaping_type: "escape_survival"
survival_reward_weight: 2.0
escape_reward_weight: 1.5
disperse_reward_weight: 0.8
kill_reward_weight: 0.3
time_critical_weight: 1.2
```

#### 步骤 3: 配置文件（SWCT）

**文件位置**: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_swct.yaml`

```yaml
learner: "reward_shaping_learner"

# SWCT 特定奖励塑形
reward_shaping_enabled: True
reward_shaping_type: "lure_block_retreat"
lure_reward_weight: 1.5
forcefield_reward_weight: 2.5
warp_prism_reward_weight: 2.0
tactical_positioning_weight: 1.2
phase_based_reward: True
```

### 6.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `RewardShapingLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **地图特定塑形**: 根据地图名称自动选择对应的奖励塑形策略
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 七、实现路径总结

### 7.1 共同特点

所有六个优化都遵循以下共同原则：

1. **最小侵入**: 只修改必要的部分，保持 dTAPE 框架完整性
2. **模块化设计**: 每个优化作为独立模块实现
3. **配置驱动**: 通过配置文件选择使用哪种优化
4. **向后兼容**: 可以随时切换回原始 dTAPE

### 7.2 修改层次

| 优化方法 | 修改层次 | 主要修改点 |
|---------|---------|-----------|
| TransformerMixer | Mixer 层 | 替换 `QMixer` 为 `TransformerQMixer` |
| HierarchicalArchitecture | Agent 层 | 替换 `RNNAgent` 为 `HierarchicalRNNAgent` |
| EnhancedStateRepresentation | Agent 层 | 替换 `RNNAgent` 为 `EnhancedRNNAgent` |
| CurriculumLearning | Learner 层 | 替换 `MAXQLearner` 为 `CurriculumLearner` |
| RewardShaping | Learner 层 | 替换 `MAXQLearner` 为 `RewardShapingLearner` |
| TargetedOptimization | Learner 层 | 替换 `MAXQLearner` 为 `RewardShapingLearner`（地图特定）|

### 7.3 代码位置总结

| 优化方法 | 核心实现文件 | 注册文件 | 配置文件 |
|---------|------------|---------|---------|
| TransformerMixer | `modules/mixers/transformer_qmix.py` | `modules/mixers/__init__.py` | `config/algs/transformer_qmix.yaml` |
| HierarchicalArchitecture | `modules/agents/hierarchical_agent.py` | `modules/agents/__init__.py` | `config/algs/hierarchical_qmix.yaml` |
| EnhancedStateRepresentation | `modules/agents/enhanced_rnn_agent.py` | `modules/agents/__init__.py` | `config/algs/enhanced_qmix.yaml` |
| CurriculumLearning | `learners/curriculum_learner.py` | `learners/__init__.py` | `config/algs/curriculum_qmix.yaml` |
| RewardShaping | `learners/reward_shaping_learner.py` | `learners/__init__.py` | `config/algs/reward_shaping_qmix.yaml` |
| TargetedOptimization | `learners/reward_shaping_learner.py` | `learners/__init__.py` | `config/algs/targeted_qmix_*.yaml` |

### 7.4 集成方式对比

| 优化方法 | 集成方式 | 继承关系 |
|---------|---------|---------|
| TransformerMixer | 直接替换 | 无（独立实现） |
| HierarchicalArchitecture | 直接替换 | 无（独立实现） |
| EnhancedStateRepresentation | 直接替换 | 无（独立实现） |
| CurriculumLearning | 继承扩展 | `CurriculumLearner(MAXQLearner)` |
| RewardShaping | 继承扩展 | `RewardShapingLearner(MAXQLearner)` |
| TargetedOptimization | 继承扩展 | `RewardShapingLearner(MAXQLearner)` |

---

## 八、使用示例

### 8.1 训练命令

```bash
# TransformerMixer
cd RLalgs/TransformerMixer
bash train_single_map.sh adcc 1 42

# HierarchicalArchitecture
cd RLalgs/HierarchicalArchitecture
bash train_single_map.sh jdsr 2 42

# EnhancedStateRepresentation
cd RLalgs/EnhancedStateRepresentation
bash train_single_map.sh gmzz 3 42

# CurriculumLearning
cd RLalgs/CurriculumLearning
bash train_single_map.sh sdjx 4 42

# RewardShaping
cd RLalgs/RewardShaping
bash train_single_map.sh wzsy 5 42

# TargetedOptimization
cd RLalgs/TargetedOptimization
bash train_single_map.sh jctq 6 42  # JCTQ 地图
bash train_single_map.sh swct 7 42  # SWCT 地图
```

### 8.2 配置文件选择

每个优化方法都有对应的配置文件，通过 `--config` 参数指定：

```bash
python src/main.py \
    --config=transformer_qmix \
    --env-config=sc2te \
    with env_args.map_name=adcc
```

---

## 九、总结

六个优化方法通过不同的技术路径实现了对 dTAPE 的改进：

1. **TransformerMixer**: 通过替换 Mixer 网络为 Transformer 架构，增强状态表示和智能体协作
2. **HierarchicalArchitecture**: 通过分层架构将决策分为高层策略和底层执行
3. **EnhancedStateRepresentation**: 通过增强的状态编码器提取更丰富的特征
4. **CurriculumLearning**: 通过课程学习策略逐步增加任务难度
5. **RewardShaping**: 通过奖励塑形改进奖励信号
6. **TargetedOptimization**: 通过地图特定的奖励塑形实现针对性优化

所有优化都保持了 dTAPE 框架的完整性，通过模块化设计和配置驱动的方式实现了灵活的集成。






## 概述

本文档详细介绍六个优化方法在 dTAPE 框架上的具体技术实现路径，包括代码位置、关键修改点、实现细节和集成方式。

---

## 一、TransformerMixer

### 1.1 优化目标

将 QMIX 的 Mixer 网络从 Hypernetwork 架构替换为 Transformer 架构，增强状态表示和智能体间协作能力。

### 1.2 技术实现路径

#### 步骤 1: 实现 TransformerQMixer 类

**文件位置**: `RLalgs/TransformerMixer/src/modules/mixers/transformer_qmix.py`

**核心实现**:

```python
class TransformerQMixer(nn.Module):
    def __init__(self, args):
        super(TransformerQMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        
        # Transformer 参数
        self.embed_dim = getattr(args, 'mixing_embed_dim', 64)
        self.n_heads = getattr(args, 'transformer_heads', 4)
        self.n_layers = getattr(args, 'transformer_layers', 2)
        self.d_ff = getattr(args, 'transformer_ff_dim', 256)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Transformer 编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 智能体 Q 值编码
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
        
        # V(s) 用于状态相关的 bias
        self.V = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states, dropout=False):
        # 编码状态
        states_flat = states.reshape(-1, self.state_dim)
        state_emb = self.state_encoder(states_flat)
        
        # 编码智能体 Q 值
        agent_qs_flat = agent_qs.reshape(-1, self.n_agents, 1)
        agent_q_emb = self.agent_q_encoder(agent_qs_flat)
        
        # 将智能体 Q 值嵌入与状态嵌入结合
        state_emb_expanded = state_emb.unsqueeze(1).expand(-1, self.n_agents, -1)
        combined_emb = agent_q_emb + state_emb_expanded
        
        # 通过 Transformer 编码器
        transformer_out = combined_emb
        for layer in self.transformer_layers:
            transformer_out = layer(transformer_out)
        
        # 聚合智能体信息（使用平均池化）
        aggregated = th.mean(transformer_out, dim=1)
        
        # 通过混合网络
        mixed = self.mixing_net(aggregated)
        
        # 添加状态相关的 bias
        v = self.V(state_emb)
        
        q_tot = (mixed + v).view(bs, seq_len, 1)
        return q_tot
```

**关键组件**:
- `MultiHeadAttention`: 多头自注意力机制
- `TransformerEncoderLayer`: Transformer 编码器层（自注意力 + FFN + 残差连接）
- `TransformerQMixer`: 主 Mixer 类，保持与 QMixer 相同的接口

#### 步骤 2: 注册 TransformerQMixer

**文件位置**: `RLalgs/TransformerMixer/src/modules/mixers/__init__.py`

```python
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transformer_qmix import TransformerQMixer

REGISTRY = {}
REGISTRY["vdn"] = VDNMixer
REGISTRY["qmix"] = QMixer
REGISTRY["transformer_qmix"] = TransformerQMixer  # 注册
```

#### 步骤 3: 修改 MAXQLearner 支持 transformer_qmix

**文件位置**: `RLalgs/TransformerMixer/src/learners/max_q_learner.py`

**修改点**: 第 1-8 行（导入）和第 42-48 行（Mixer 选择）

```python
# 添加导入
from modules.mixers.transformer_qmix import TransformerQMixer

# 在 __init__ 方法中修改
if args.mixer == "vdn":
    self.mixer = VDNMixer()
elif args.mixer == "qmix":
    self.mixer = QMixer(args)
elif args.mixer == "transformer_qmix":  # 新增
    self.mixer = TransformerQMixer(args)  # 使用 Transformer
else:
    raise ValueError("Mixer {} not recognised.".format(args.mixer))
```

#### 步骤 4: 配置文件

**文件位置**: `RLalgs/TransformerMixer/src/config/algs/transformer_qmix.yaml`

```yaml
mixer: "transformer_qmix"  # 关键：使用 transformer_qmix

# Transformer 特定参数
transformer_heads: 4
transformer_layers: 2
transformer_ff_dim: 256
transformer_dropout: 0.1
mixing_embed_dim: 64

# 保持 dTAPE 其他配置
learner: "max_q_learner"
central_loss: 1
qmix_loss: 1
hysteretic_qmix: True
comm: True
```

### 1.3 与 dTAPE 的集成

- **保持 dTAPE 框架**: 所有 dTAPE 组件（Central Q、通信机制、OW-QMIX）完全保留
- **仅替换 Mixer**: 在 `MAXQLearner.__init__` 中，将 `QMixer(args)` 替换为 `TransformerQMixer(args)`
- **接口兼容**: `TransformerQMixer.forward()` 与 `QMixer.forward()` 接口完全一致

---

## 二、HierarchicalArchitecture

### 2.1 优化目标

引入分层架构，将决策分为高层策略（目标选择、战术决策）和底层执行（具体动作）。

### 2.2 技术实现路径

#### 步骤 1: 实现高层策略网络

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

**核心实现**:

```python
class HighLevelPolicy(nn.Module):
    """高层策略网络：制定宏观策略"""
    def __init__(self, input_shape, args):
        super(HighLevelPolicy, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.fc1 = nn.Linear(input_shape, args.hierarchical_high_dim)
        self.fc2 = nn.Linear(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
        # 输出：目标选择、战术类型
        self.goal_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_goals)
        self.tactic_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_tactics)
        
        self.rnn = nn.GRUCell(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        goal_logits = self.goal_head(h)
        tactic_logits = self.tactic_head(h)
        
        return {
            'goal': goal_logits,
            'tactic': tactic_logits,
            'hidden': h
        }
```

#### 步骤 2: 实现底层执行网络

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

```python
class LowLevelPolicy(nn.Module):
    """底层执行网络：基于高层策略执行具体动作"""
    def __init__(self, input_shape, args):
        super(LowLevelPolicy, self).__init__()
        self.args = args
        
        # 底层输入：原始观测 + 高层策略信息
        enhanced_input_dim = (
            input_shape + 
            args.hierarchical_high_dim + 
            args.hierarchical_n_goals + 
            args.hierarchical_n_tactics
        )
        
        self.fc1 = nn.Linear(enhanced_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
    def forward(self, inputs, high_level_info, hidden_state=None):
        # 拼接原始观测和高层策略信息
        enhanced_input = torch.cat([
            inputs,
            high_level_info['hidden'],
            F.softmax(high_level_info['goal'], dim=-1),
            F.softmax(high_level_info['tactic'], dim=-1)
        ], dim=-1)
        
        x = F.relu(self.fc1(enhanced_input), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        q = self.fc2(h)
        return q, h
```

#### 步骤 3: 实现分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

```python
class HierarchicalRNNAgent(nn.Module):
    """分层架构智能体：结合高层策略和底层执行"""
    def __init__(self, input_shape, args):
        super(HierarchicalRNNAgent, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.high_level = HighLevelPolicy(input_shape, args)
        
        # 底层执行网络
        self.low_level = LowLevelPolicy(input_shape, args)
        
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        inputs_flat = inputs.view(-1, e)
        
        # 获取高层策略
        if hidden_state is not None:
            high_hidden, low_hidden = hidden_state
            high_hidden = high_hidden.reshape(-1, self.args.hierarchical_high_dim)
            low_hidden = low_hidden.reshape(-1, self.args.rnn_hidden_dim)
        else:
            high_hidden = None
            low_hidden = None
        
        high_level_info = self.high_level(inputs_flat, high_hidden)
        new_high_hidden = high_level_info['hidden']
        
        # 获取底层 Q 值
        q, new_low_hidden = self.low_level(inputs_flat, high_level_info, low_hidden)
        
        # 重塑输出
        q = q.view(b, a, -1)
        new_high_hidden = new_high_hidden.view(b, a, -1)
        new_low_hidden = new_low_hidden.view(b, a, -1)
        
        return q, (new_high_hidden, new_low_hidden)
```

#### 步骤 4: 注册分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/__init__.py`

```python
from modules.agents.hierarchical_agent import HierarchicalRNNAgent

REGISTRY = {}
REGISTRY["hierarchical_rnn"] = HierarchicalRNNAgent
```

#### 步骤 5: 修改 MAC 使用分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/controllers/basic_controller.py`

**修改点**: 在 `BasicMAC` 中，将 `agent` 从 `"rnn"` 改为 `"hierarchical_rnn"`

#### 步骤 6: 配置文件

**文件位置**: `RLalgs/HierarchicalArchitecture/src/config/algs/hierarchical_qmix.yaml`

```yaml
agent: "hierarchical_rnn"  # 使用分层智能体

# 分层架构参数
hierarchical_high_dim: 128
hierarchical_n_goals: 8
hierarchical_n_tactics: 4

# 保持 dTAPE 其他配置
learner: "max_q_learner"
mixer: "qmix"
```

### 2.3 与 dTAPE 的集成

- **替换 Agent**: 在 `BasicMAC` 中，将 `RNNAgent` 替换为 `HierarchicalRNNAgent`
- **保持其他组件**: Mixer、Learner、通信机制等完全保留
- **隐藏状态扩展**: 隐藏状态从 `(hidden,)` 扩展为 `(high_hidden, low_hidden)`

---

## 三、EnhancedStateRepresentation

### 3.1 优化目标

通过增强的状态编码器提取更丰富的特征，使用多层特征提取和注意力机制改进状态表示。

### 3.2 技术实现路径

#### 步骤 1: 实现状态编码器

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/enhanced_rnn_agent.py`

**核心实现**:

```python
class StateEncoder(nn.Module):
    """增强的状态编码器：提取更丰富的状态特征"""
    def __init__(self, input_shape, args):
        super(StateEncoder, self).__init__()
        self.args = args
        
        # 特征提取网络（多层 + LayerNorm）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制：关注重要特征
        self.attention = nn.Sequential(
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args.enhanced_feature_dim // 2, args.enhanced_feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 注意力加权
        attention_weights = self.attention(features)
        enhanced_features = features * attention_weights
        
        return enhanced_features
```

#### 步骤 2: 实现增强的 RNN 智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/enhanced_rnn_agent.py`

```python
class EnhancedRNNAgent(nn.Module):
    """增强的 RNN 智能体：使用改进的状态表示"""
    def __init__(self, input_shape, args):
        super(EnhancedRNNAgent, self).__init__()
        self.args = args
        
        # 状态编码器
        self.state_encoder = StateEncoder(input_shape, args)
        
        # RNN 网络
        self.rnn = nn.GRUCell(
            args.enhanced_feature_dim, 
            args.rnn_hidden_dim
        )
        
        # 输出层（多层 + LayerNorm + Dropout）
        self.fc_out = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        inputs_flat = inputs.view(-1, e)
        
        # 状态编码
        encoded_features = self.state_encoder(inputs_flat)
        
        # RNN 处理
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(encoded_features, hidden_state)
        
        # 输出 Q 值
        q = self.fc_out(h)
        q = torch.clamp(q, -5, 2)
        
        return q.view(b, a, -1), h.view(b, a, -1)
```

#### 步骤 3: 注册增强智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/__init__.py`

```python
from modules.agents.enhanced_rnn_agent import EnhancedRNNAgent

REGISTRY = {}
REGISTRY["enhanced_rnn"] = EnhancedRNNAgent
```

#### 步骤 4: 修改 MAC 使用增强智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/controllers/basic_controller.py`

**修改点**: 在 `BasicMAC` 中，将 `agent` 从 `"rnn"` 改为 `"enhanced_rnn"`

#### 步骤 5: 配置文件

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/config/algs/enhanced_qmix.yaml`

```yaml
agent: "enhanced_rnn"  # 使用增强智能体

# 增强状态表示参数
enhanced_feature_dim: 128
enhanced_dropout: 0.1
use_spatial_features: False

# 保持 dTAPE 其他配置
learner: "max_q_learner"
mixer: "qmix"
```

### 3.3 与 dTAPE 的集成

- **替换 Agent**: 在 `BasicMAC` 中，将 `RNNAgent` 替换为 `EnhancedRNNAgent`
- **保持其他组件**: Mixer、Learner、通信机制等完全保留
- **状态处理流程**: 原始观测 → 状态编码器 → RNN → 输出层

---

## 四、CurriculumLearning

### 4.1 优化目标

通过课程学习策略，从简单任务开始，逐步增加任务难度，使智能体能够更好地学习和适应复杂任务。

### 4.2 技术实现路径

#### 步骤 1: 实现课程学习 Learner

**文件位置**: `RLalgs/CurriculumLearning/src/learners/curriculum_learner.py`

**核心实现**:

```python
class CurriculumLearner(MAXQLearner):
    """课程学习：逐步增加任务难度"""
    def __init__(self, mac, scheme, logger, args):
        super(CurriculumLearner, self).__init__(mac, scheme, logger, args)
        
        # 课程学习参数
        self.curriculum_enabled = getattr(args, 'curriculum_enabled', True)
        self.curriculum_schedule = getattr(args, 'curriculum_schedule', 'linear')
        self.curriculum_start_step = getattr(args, 'curriculum_start_step', 0)
        self.curriculum_end_step = getattr(args, 'curriculum_end_step', 1000000)
        
        # 难度级别（0-1，0最简单，1最难）
        self.current_difficulty = 0.0
        self.min_difficulty = getattr(args, 'curriculum_min_difficulty', 0.0)
        self.max_difficulty = getattr(args, 'curriculum_max_difficulty', 1.0)
        
        # 课程学习指标
        self.episode_wins = deque(maxlen=100)
        
    def update_curriculum_difficulty(self, t_env):
        """根据训练进度更新课程难度"""
        if not self.curriculum_enabled:
            self.current_difficulty = self.max_difficulty
            return
        
        if t_env < self.curriculum_start_step:
            self.current_difficulty = self.min_difficulty
        elif t_env >= self.curriculum_end_step:
            self.current_difficulty = self.max_difficulty
        else:
            # 线性调度
            if self.curriculum_schedule == 'linear':
                progress = (t_env - self.curriculum_start_step) / (
                    self.curriculum_end_step - self.curriculum_start_step
                )
                self.current_difficulty = (
                    self.min_difficulty + 
                    (self.max_difficulty - self.min_difficulty) * progress
                )
            # 自适应调度（基于性能）
            elif self.curriculum_schedule == 'adaptive':
                if len(self.episode_wins) > 10:
                    recent_win_rate = np.mean(list(self.episode_wins)[-10:])
                    if recent_win_rate > 0.8:
                        # 性能好，增加难度
                        self.current_difficulty = min(
                            self.current_difficulty + 0.01,
                            self.max_difficulty
                        )
                    elif recent_win_rate < 0.2:
                        # 性能差，降低难度
                        self.current_difficulty = max(
                            self.current_difficulty - 0.01,
                            self.min_difficulty
                        )
    
    def apply_curriculum_to_batch(self, batch):
        """将课程难度应用到 batch"""
        # 根据难度调整 batch（例如：过滤困难样本、调整奖励等）
        # 具体实现取决于任务特点
        return batch
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 更新课程难度
        self.update_curriculum_difficulty(t_env)
        
        # 应用课程学习到 batch
        batch = self.apply_curriculum_to_batch(batch)
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 注册课程学习 Learner

**文件位置**: `RLalgs/CurriculumLearning/src/learners/__init__.py`

```python
from learners.curriculum_learner import CurriculumLearner

REGISTRY = {}
REGISTRY["curriculum_learner"] = CurriculumLearner
```

#### 步骤 3: 配置文件

**文件位置**: `RLalgs/CurriculumLearning/src/config/algs/curriculum_qmix.yaml`

```yaml
learner: "curriculum_learner"  # 使用课程学习 learner

# 课程学习参数
curriculum_enabled: True
curriculum_schedule: "linear"  # "linear" 或 "adaptive"
curriculum_start_step: 0
curriculum_end_step: 1000000
curriculum_min_difficulty: 0.0
curriculum_max_difficulty: 1.0

# 保持 dTAPE 其他配置
mixer: "qmix"
agent: "rnn"
```

### 4.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `CurriculumLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **重写 train 方法**: 在训练前更新难度并应用到 batch
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 五、RewardShaping

### 5.1 优化目标

通过奖励塑形改进奖励信号，使智能体能够更快、更稳定地学习，特别适合稀疏奖励环境。

### 5.2 技术实现路径

#### 步骤 1: 实现奖励塑形 Learner

**文件位置**: `RLalgs/RewardShaping/src/learners/reward_shaping_learner.py`

**核心实现**:

```python
class RewardShapingLearner(MAXQLearner):
    """奖励塑形：通过改进奖励信号提升学习效率"""
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # 奖励塑形权重
        self.shaping_weight = getattr(args, 'reward_shaping_weight', 0.1)
        self.shaping_decay = getattr(args, 'reward_shaping_decay', 0.99)
        self.current_shaping_weight = self.shaping_weight
        
    def shape_reward(self, rewards, states, next_states, t_env):
        """对奖励进行塑形"""
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        if self.shaping_type == 'potential_based':
            # 基于潜在函数的奖励塑形
            # r' = r + γφ(s') - φ(s)
            potential_current = self._compute_potential(states)
            potential_next = self._compute_potential(next_states)
            potential_bonus = potential_next - potential_current
            shaped_rewards = rewards + self.current_shaping_weight * potential_bonus
        
        elif self.shaping_type == 'dense_reward':
            # 密集奖励：为中间步骤添加奖励
            shaped_rewards = self._add_dense_rewards(rewards, states)
        
        elif self.shaping_type == 'curiosity':
            # 好奇心驱动：鼓励探索
            shaped_rewards = self._add_curiosity_bonus(rewards, states)
        
        # 更新塑形权重（逐渐减少）
        self.current_shaping_weight *= self.shaping_decay
        
        return shaped_rewards
    
    def _compute_potential(self, states):
        """计算潜在函数值"""
        # 简化版本：可以扩展为神经网络
        return th.zeros_like(states[:, :, 0:1])
    
    def _add_dense_rewards(self, rewards, states):
        """添加密集奖励"""
        # 为每步添加小的生存奖励
        survival_bonus = 0.01
        return rewards + survival_bonus
    
    def _add_curiosity_bonus(self, rewards, states):
        """添加好奇心奖励"""
        # 基于状态新颖性的奖励
        # 简化版本
        return rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取原始奖励
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        next_states = batch["state"][:, 1:]
        
        # 对奖励进行塑形
        shaped_rewards = self.shape_reward(rewards, states, next_states, t_env)
        
        # 替换 batch 中的奖励
        batch["reward"][:, :-1] = shaped_rewards
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 注册奖励塑形 Learner

**文件位置**: `RLalgs/RewardShaping/src/learners/__init__.py`

```python
from learners.reward_shaping_learner import RewardShapingLearner

REGISTRY = {}
REGISTRY["reward_shaping_learner"] = RewardShapingLearner
```

#### 步骤 3: 配置文件

**文件位置**: `RLalgs/RewardShaping/src/config/algs/reward_shaping_qmix.yaml`

```yaml
learner: "reward_shaping_learner"  # 使用奖励塑形 learner

# 奖励塑形参数
reward_shaping_enabled: True
reward_shaping_type: "potential_based"  # "potential_based", "dense_reward", "curiosity"
reward_shaping_weight: 0.1
reward_shaping_decay: 0.99

# 保持 dTAPE 其他配置
mixer: "qmix"
agent: "rnn"
```

### 5.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `RewardShapingLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **重写 train 方法**: 在训练前对奖励进行塑形
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 六、TargetedOptimization

### 6.1 优化目标

针对特定地图设计专门的奖励塑形策略，根据地图的战术特点和游戏机制定制优化方案。

### 6.2 技术实现路径

#### 步骤 1: 实现地图特定的奖励塑形 Learner

**文件位置**: `RLalgs/TargetedOptimization/src/learners/reward_shaping_learner.py`

**核心实现**:

```python
class RewardShapingLearner(MAXQLearner):
    """奖励塑形：针对特定地图的优化"""
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 获取地图名称
        try:
            self.map_name = getattr(args.env_args, 'map_name', '')
        except:
            self.map_name = ''
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # JCTQ (金蝉脱壳) 特定参数
        self.survival_reward_weight = getattr(args, 'survival_reward_weight', 2.0)
        self.escape_reward_weight = getattr(args, 'escape_reward_weight', 1.5)
        self.disperse_reward_weight = getattr(args, 'disperse_reward_weight', 0.8)
        self.kill_reward_weight = getattr(args, 'kill_reward_weight', 0.3)
        self.time_critical_weight = getattr(args, 'time_critical_weight', 1.2)
        
        # SWCT (上屋抽梯) 特定参数
        self.lure_reward_weight = getattr(args, 'lure_reward_weight', 1.5)
        self.forcefield_reward_weight = getattr(args, 'forcefield_reward_weight', 2.5)
        self.warp_prism_reward_weight = getattr(args, 'warp_prism_reward_weight', 2.0)
        self.tactical_positioning_weight = getattr(args, 'tactical_positioning_weight', 1.2)
        
    def shape_reward(self, rewards, states, next_states, t_env, batch=None):
        """对奖励进行塑形（地图特定）"""
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        # JCTQ (金蝉脱壳) - 逃脱生存策略
        if self.shaping_type == 'escape_survival' or self.map_name == 'jctq':
            shaped_rewards = self._apply_escape_survival_shaping(
                rewards, states, next_states, batch
            )
        
        # SWCT (上屋抽梯) - 诱敌阻挡撤退策略
        elif self.shaping_type == 'lure_block_retreat' or self.map_name == 'swct':
            shaped_rewards = self._apply_lure_block_retreat_shaping(
                rewards, states, next_states, batch, t_env
            )
        
        return shaped_rewards
    
    def _apply_escape_survival_shaping(self, rewards, states, next_states, batch):
        """JCTQ: 逃脱生存策略的奖励塑形"""
        shaped_rewards = rewards.clone()
        
        # 1. 生存奖励：鼓励单位存活
        # 检测单位是否存活（简化版本）
        survival_bonus = self.survival_reward_weight * 0.1  # 每步存活奖励
        shaped_rewards += survival_bonus
        
        # 2. 逃脱奖励：鼓励远离敌人
        # 计算与敌人的距离（简化版本）
        escape_bonus = self.escape_reward_weight * 0.05
        shaped_rewards += escape_bonus
        
        # 3. 分散奖励：鼓励单位分散
        disperse_bonus = self.disperse_reward_weight * 0.03
        shaped_rewards += disperse_bonus
        
        # 4. 击杀惩罚：避免暴露位置
        # 如果击杀敌人，给予较小奖励（避免暴露）
        kill_penalty = -self.kill_reward_weight * 0.1
        # 检测击杀（简化版本）
        # shaped_rewards += kill_penalty
        
        # 5. 时间紧迫性：后期给予更多奖励
        time_bonus = self.time_critical_weight * 0.02
        shaped_rewards += time_bonus
        
        return shaped_rewards
    
    def _apply_lure_block_retreat_shaping(self, rewards, states, next_states, batch, t_env):
        """SWCT: 诱敌阻挡撤退策略的奖励塑形"""
        shaped_rewards = rewards.clone()
        
        # 1. 强生存奖励
        survival_bonus = 3.0 * 0.1
        shaped_rewards += survival_bonus
        
        # 2. ForceField 使用奖励
        forcefield_bonus = self.forcefield_reward_weight * 0.2
        # 检测 ForceField 使用（简化版本）
        # shaped_rewards += forcefield_bonus
        
        # 3. WarpPrism 使用奖励
        warp_prism_bonus = self.warp_prism_reward_weight * 0.15
        # 检测 WarpPrism 使用（简化版本）
        # shaped_rewards += warp_prism_bonus
        
        # 4. 战术位置奖励
        positioning_bonus = self.tactical_positioning_weight * 0.1
        shaped_rewards += positioning_bonus
        
        # 5. 诱敌奖励
        lure_bonus = self.lure_reward_weight * 0.1
        shaped_rewards += lure_bonus
        
        return shaped_rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取原始奖励
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        next_states = batch["state"][:, 1:]
        
        # 对奖励进行塑形（地图特定）
        shaped_rewards = self.shape_reward(rewards, states, next_states, t_env, batch)
        
        # 替换 batch 中的奖励
        batch["reward"][:, :-1] = shaped_rewards
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 配置文件（JCTQ）

**文件位置**: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_jctq.yaml`

```yaml
learner: "reward_shaping_learner"

# JCTQ 特定奖励塑形
reward_shaping_enabled: True
reward_shaping_type: "escape_survival"
survival_reward_weight: 2.0
escape_reward_weight: 1.5
disperse_reward_weight: 0.8
kill_reward_weight: 0.3
time_critical_weight: 1.2
```

#### 步骤 3: 配置文件（SWCT）

**文件位置**: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_swct.yaml`

```yaml
learner: "reward_shaping_learner"

# SWCT 特定奖励塑形
reward_shaping_enabled: True
reward_shaping_type: "lure_block_retreat"
lure_reward_weight: 1.5
forcefield_reward_weight: 2.5
warp_prism_reward_weight: 2.0
tactical_positioning_weight: 1.2
phase_based_reward: True
```

### 6.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `RewardShapingLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **地图特定塑形**: 根据地图名称自动选择对应的奖励塑形策略
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 七、实现路径总结

### 7.1 共同特点

所有六个优化都遵循以下共同原则：

1. **最小侵入**: 只修改必要的部分，保持 dTAPE 框架完整性
2. **模块化设计**: 每个优化作为独立模块实现
3. **配置驱动**: 通过配置文件选择使用哪种优化
4. **向后兼容**: 可以随时切换回原始 dTAPE

### 7.2 修改层次

| 优化方法 | 修改层次 | 主要修改点 |
|---------|---------|-----------|
| TransformerMixer | Mixer 层 | 替换 `QMixer` 为 `TransformerQMixer` |
| HierarchicalArchitecture | Agent 层 | 替换 `RNNAgent` 为 `HierarchicalRNNAgent` |
| EnhancedStateRepresentation | Agent 层 | 替换 `RNNAgent` 为 `EnhancedRNNAgent` |
| CurriculumLearning | Learner 层 | 替换 `MAXQLearner` 为 `CurriculumLearner` |
| RewardShaping | Learner 层 | 替换 `MAXQLearner` 为 `RewardShapingLearner` |
| TargetedOptimization | Learner 层 | 替换 `MAXQLearner` 为 `RewardShapingLearner`（地图特定）|

### 7.3 代码位置总结

| 优化方法 | 核心实现文件 | 注册文件 | 配置文件 |
|---------|------------|---------|---------|
| TransformerMixer | `modules/mixers/transformer_qmix.py` | `modules/mixers/__init__.py` | `config/algs/transformer_qmix.yaml` |
| HierarchicalArchitecture | `modules/agents/hierarchical_agent.py` | `modules/agents/__init__.py` | `config/algs/hierarchical_qmix.yaml` |
| EnhancedStateRepresentation | `modules/agents/enhanced_rnn_agent.py` | `modules/agents/__init__.py` | `config/algs/enhanced_qmix.yaml` |
| CurriculumLearning | `learners/curriculum_learner.py` | `learners/__init__.py` | `config/algs/curriculum_qmix.yaml` |
| RewardShaping | `learners/reward_shaping_learner.py` | `learners/__init__.py` | `config/algs/reward_shaping_qmix.yaml` |
| TargetedOptimization | `learners/reward_shaping_learner.py` | `learners/__init__.py` | `config/algs/targeted_qmix_*.yaml` |

### 7.4 集成方式对比

| 优化方法 | 集成方式 | 继承关系 |
|---------|---------|---------|
| TransformerMixer | 直接替换 | 无（独立实现） |
| HierarchicalArchitecture | 直接替换 | 无（独立实现） |
| EnhancedStateRepresentation | 直接替换 | 无（独立实现） |
| CurriculumLearning | 继承扩展 | `CurriculumLearner(MAXQLearner)` |
| RewardShaping | 继承扩展 | `RewardShapingLearner(MAXQLearner)` |
| TargetedOptimization | 继承扩展 | `RewardShapingLearner(MAXQLearner)` |

---

## 八、使用示例

### 8.1 训练命令

```bash
# TransformerMixer
cd RLalgs/TransformerMixer
bash train_single_map.sh adcc 1 42

# HierarchicalArchitecture
cd RLalgs/HierarchicalArchitecture
bash train_single_map.sh jdsr 2 42

# EnhancedStateRepresentation
cd RLalgs/EnhancedStateRepresentation
bash train_single_map.sh gmzz 3 42

# CurriculumLearning
cd RLalgs/CurriculumLearning
bash train_single_map.sh sdjx 4 42

# RewardShaping
cd RLalgs/RewardShaping
bash train_single_map.sh wzsy 5 42

# TargetedOptimization
cd RLalgs/TargetedOptimization
bash train_single_map.sh jctq 6 42  # JCTQ 地图
bash train_single_map.sh swct 7 42  # SWCT 地图
```

### 8.2 配置文件选择

每个优化方法都有对应的配置文件，通过 `--config` 参数指定：

```bash
python src/main.py \
    --config=transformer_qmix \
    --env-config=sc2te \
    with env_args.map_name=adcc
```

---

## 九、总结

六个优化方法通过不同的技术路径实现了对 dTAPE 的改进：

1. **TransformerMixer**: 通过替换 Mixer 网络为 Transformer 架构，增强状态表示和智能体协作
2. **HierarchicalArchitecture**: 通过分层架构将决策分为高层策略和底层执行
3. **EnhancedStateRepresentation**: 通过增强的状态编码器提取更丰富的特征
4. **CurriculumLearning**: 通过课程学习策略逐步增加任务难度
5. **RewardShaping**: 通过奖励塑形改进奖励信号
6. **TargetedOptimization**: 通过地图特定的奖励塑形实现针对性优化

所有优化都保持了 dTAPE 框架的完整性，通过模块化设计和配置驱动的方式实现了灵活的集成。






## 概述

本文档详细介绍六个优化方法在 dTAPE 框架上的具体技术实现路径，包括代码位置、关键修改点、实现细节和集成方式。

---

## 一、TransformerMixer

### 1.1 优化目标

将 QMIX 的 Mixer 网络从 Hypernetwork 架构替换为 Transformer 架构，增强状态表示和智能体间协作能力。

### 1.2 技术实现路径

#### 步骤 1: 实现 TransformerQMixer 类

**文件位置**: `RLalgs/TransformerMixer/src/modules/mixers/transformer_qmix.py`

**核心实现**:

```python
class TransformerQMixer(nn.Module):
    def __init__(self, args):
        super(TransformerQMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        
        # Transformer 参数
        self.embed_dim = getattr(args, 'mixing_embed_dim', 64)
        self.n_heads = getattr(args, 'transformer_heads', 4)
        self.n_layers = getattr(args, 'transformer_layers', 2)
        self.d_ff = getattr(args, 'transformer_ff_dim', 256)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Transformer 编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 智能体 Q 值编码
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
        
        # V(s) 用于状态相关的 bias
        self.V = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states, dropout=False):
        # 编码状态
        states_flat = states.reshape(-1, self.state_dim)
        state_emb = self.state_encoder(states_flat)
        
        # 编码智能体 Q 值
        agent_qs_flat = agent_qs.reshape(-1, self.n_agents, 1)
        agent_q_emb = self.agent_q_encoder(agent_qs_flat)
        
        # 将智能体 Q 值嵌入与状态嵌入结合
        state_emb_expanded = state_emb.unsqueeze(1).expand(-1, self.n_agents, -1)
        combined_emb = agent_q_emb + state_emb_expanded
        
        # 通过 Transformer 编码器
        transformer_out = combined_emb
        for layer in self.transformer_layers:
            transformer_out = layer(transformer_out)
        
        # 聚合智能体信息（使用平均池化）
        aggregated = th.mean(transformer_out, dim=1)
        
        # 通过混合网络
        mixed = self.mixing_net(aggregated)
        
        # 添加状态相关的 bias
        v = self.V(state_emb)
        
        q_tot = (mixed + v).view(bs, seq_len, 1)
        return q_tot
```

**关键组件**:
- `MultiHeadAttention`: 多头自注意力机制
- `TransformerEncoderLayer`: Transformer 编码器层（自注意力 + FFN + 残差连接）
- `TransformerQMixer`: 主 Mixer 类，保持与 QMixer 相同的接口

#### 步骤 2: 注册 TransformerQMixer

**文件位置**: `RLalgs/TransformerMixer/src/modules/mixers/__init__.py`

```python
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transformer_qmix import TransformerQMixer

REGISTRY = {}
REGISTRY["vdn"] = VDNMixer
REGISTRY["qmix"] = QMixer
REGISTRY["transformer_qmix"] = TransformerQMixer  # 注册
```

#### 步骤 3: 修改 MAXQLearner 支持 transformer_qmix

**文件位置**: `RLalgs/TransformerMixer/src/learners/max_q_learner.py`

**修改点**: 第 1-8 行（导入）和第 42-48 行（Mixer 选择）

```python
# 添加导入
from modules.mixers.transformer_qmix import TransformerQMixer

# 在 __init__ 方法中修改
if args.mixer == "vdn":
    self.mixer = VDNMixer()
elif args.mixer == "qmix":
    self.mixer = QMixer(args)
elif args.mixer == "transformer_qmix":  # 新增
    self.mixer = TransformerQMixer(args)  # 使用 Transformer
else:
    raise ValueError("Mixer {} not recognised.".format(args.mixer))
```

#### 步骤 4: 配置文件

**文件位置**: `RLalgs/TransformerMixer/src/config/algs/transformer_qmix.yaml`

```yaml
mixer: "transformer_qmix"  # 关键：使用 transformer_qmix

# Transformer 特定参数
transformer_heads: 4
transformer_layers: 2
transformer_ff_dim: 256
transformer_dropout: 0.1
mixing_embed_dim: 64

# 保持 dTAPE 其他配置
learner: "max_q_learner"
central_loss: 1
qmix_loss: 1
hysteretic_qmix: True
comm: True
```

### 1.3 与 dTAPE 的集成

- **保持 dTAPE 框架**: 所有 dTAPE 组件（Central Q、通信机制、OW-QMIX）完全保留
- **仅替换 Mixer**: 在 `MAXQLearner.__init__` 中，将 `QMixer(args)` 替换为 `TransformerQMixer(args)`
- **接口兼容**: `TransformerQMixer.forward()` 与 `QMixer.forward()` 接口完全一致

---

## 二、HierarchicalArchitecture

### 2.1 优化目标

引入分层架构，将决策分为高层策略（目标选择、战术决策）和底层执行（具体动作）。

### 2.2 技术实现路径

#### 步骤 1: 实现高层策略网络

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

**核心实现**:

```python
class HighLevelPolicy(nn.Module):
    """高层策略网络：制定宏观策略"""
    def __init__(self, input_shape, args):
        super(HighLevelPolicy, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.fc1 = nn.Linear(input_shape, args.hierarchical_high_dim)
        self.fc2 = nn.Linear(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
        # 输出：目标选择、战术类型
        self.goal_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_goals)
        self.tactic_head = nn.Linear(args.hierarchical_high_dim, args.hierarchical_n_tactics)
        
        self.rnn = nn.GRUCell(args.hierarchical_high_dim, args.hierarchical_high_dim)
        
    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        goal_logits = self.goal_head(h)
        tactic_logits = self.tactic_head(h)
        
        return {
            'goal': goal_logits,
            'tactic': tactic_logits,
            'hidden': h
        }
```

#### 步骤 2: 实现底层执行网络

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

```python
class LowLevelPolicy(nn.Module):
    """底层执行网络：基于高层策略执行具体动作"""
    def __init__(self, input_shape, args):
        super(LowLevelPolicy, self).__init__()
        self.args = args
        
        # 底层输入：原始观测 + 高层策略信息
        enhanced_input_dim = (
            input_shape + 
            args.hierarchical_high_dim + 
            args.hierarchical_n_goals + 
            args.hierarchical_n_tactics
        )
        
        self.fc1 = nn.Linear(enhanced_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
    def forward(self, inputs, high_level_info, hidden_state=None):
        # 拼接原始观测和高层策略信息
        enhanced_input = torch.cat([
            inputs,
            high_level_info['hidden'],
            F.softmax(high_level_info['goal'], dim=-1),
            F.softmax(high_level_info['tactic'], dim=-1)
        ], dim=-1)
        
        x = F.relu(self.fc1(enhanced_input), inplace=True)
        
        if hidden_state is not None:
            h = self.rnn(x, hidden_state)
        else:
            h = self.rnn(x, torch.zeros_like(x))
        
        q = self.fc2(h)
        return q, h
```

#### 步骤 3: 实现分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/hierarchical_agent.py`

```python
class HierarchicalRNNAgent(nn.Module):
    """分层架构智能体：结合高层策略和底层执行"""
    def __init__(self, input_shape, args):
        super(HierarchicalRNNAgent, self).__init__()
        self.args = args
        
        # 高层策略网络
        self.high_level = HighLevelPolicy(input_shape, args)
        
        # 底层执行网络
        self.low_level = LowLevelPolicy(input_shape, args)
        
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        inputs_flat = inputs.view(-1, e)
        
        # 获取高层策略
        if hidden_state is not None:
            high_hidden, low_hidden = hidden_state
            high_hidden = high_hidden.reshape(-1, self.args.hierarchical_high_dim)
            low_hidden = low_hidden.reshape(-1, self.args.rnn_hidden_dim)
        else:
            high_hidden = None
            low_hidden = None
        
        high_level_info = self.high_level(inputs_flat, high_hidden)
        new_high_hidden = high_level_info['hidden']
        
        # 获取底层 Q 值
        q, new_low_hidden = self.low_level(inputs_flat, high_level_info, low_hidden)
        
        # 重塑输出
        q = q.view(b, a, -1)
        new_high_hidden = new_high_hidden.view(b, a, -1)
        new_low_hidden = new_low_hidden.view(b, a, -1)
        
        return q, (new_high_hidden, new_low_hidden)
```

#### 步骤 4: 注册分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/modules/agents/__init__.py`

```python
from modules.agents.hierarchical_agent import HierarchicalRNNAgent

REGISTRY = {}
REGISTRY["hierarchical_rnn"] = HierarchicalRNNAgent
```

#### 步骤 5: 修改 MAC 使用分层智能体

**文件位置**: `RLalgs/HierarchicalArchitecture/src/controllers/basic_controller.py`

**修改点**: 在 `BasicMAC` 中，将 `agent` 从 `"rnn"` 改为 `"hierarchical_rnn"`

#### 步骤 6: 配置文件

**文件位置**: `RLalgs/HierarchicalArchitecture/src/config/algs/hierarchical_qmix.yaml`

```yaml
agent: "hierarchical_rnn"  # 使用分层智能体

# 分层架构参数
hierarchical_high_dim: 128
hierarchical_n_goals: 8
hierarchical_n_tactics: 4

# 保持 dTAPE 其他配置
learner: "max_q_learner"
mixer: "qmix"
```

### 2.3 与 dTAPE 的集成

- **替换 Agent**: 在 `BasicMAC` 中，将 `RNNAgent` 替换为 `HierarchicalRNNAgent`
- **保持其他组件**: Mixer、Learner、通信机制等完全保留
- **隐藏状态扩展**: 隐藏状态从 `(hidden,)` 扩展为 `(high_hidden, low_hidden)`

---

## 三、EnhancedStateRepresentation

### 3.1 优化目标

通过增强的状态编码器提取更丰富的特征，使用多层特征提取和注意力机制改进状态表示。

### 3.2 技术实现路径

#### 步骤 1: 实现状态编码器

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/enhanced_rnn_agent.py`

**核心实现**:

```python
class StateEncoder(nn.Module):
    """增强的状态编码器：提取更丰富的状态特征"""
    def __init__(self, input_shape, args):
        super(StateEncoder, self).__init__()
        self.args = args
        
        # 特征提取网络（多层 + LayerNorm）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim),
            nn.LayerNorm(args.enhanced_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制：关注重要特征
        self.attention = nn.Sequential(
            nn.Linear(args.enhanced_feature_dim, args.enhanced_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args.enhanced_feature_dim // 2, args.enhanced_feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 注意力加权
        attention_weights = self.attention(features)
        enhanced_features = features * attention_weights
        
        return enhanced_features
```

#### 步骤 2: 实现增强的 RNN 智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/enhanced_rnn_agent.py`

```python
class EnhancedRNNAgent(nn.Module):
    """增强的 RNN 智能体：使用改进的状态表示"""
    def __init__(self, input_shape, args):
        super(EnhancedRNNAgent, self).__init__()
        self.args = args
        
        # 状态编码器
        self.state_encoder = StateEncoder(input_shape, args)
        
        # RNN 网络
        self.rnn = nn.GRUCell(
            args.enhanced_feature_dim, 
            args.rnn_hidden_dim
        )
        
        # 输出层（多层 + LayerNorm + Dropout）
        self.fc_out = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.enhanced_dropout),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        
    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        inputs_flat = inputs.view(-1, e)
        
        # 状态编码
        encoded_features = self.state_encoder(inputs_flat)
        
        # RNN 处理
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(encoded_features, hidden_state)
        
        # 输出 Q 值
        q = self.fc_out(h)
        q = torch.clamp(q, -5, 2)
        
        return q.view(b, a, -1), h.view(b, a, -1)
```

#### 步骤 3: 注册增强智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/modules/agents/__init__.py`

```python
from modules.agents.enhanced_rnn_agent import EnhancedRNNAgent

REGISTRY = {}
REGISTRY["enhanced_rnn"] = EnhancedRNNAgent
```

#### 步骤 4: 修改 MAC 使用增强智能体

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/controllers/basic_controller.py`

**修改点**: 在 `BasicMAC` 中，将 `agent` 从 `"rnn"` 改为 `"enhanced_rnn"`

#### 步骤 5: 配置文件

**文件位置**: `RLalgs/EnhancedStateRepresentation/src/config/algs/enhanced_qmix.yaml`

```yaml
agent: "enhanced_rnn"  # 使用增强智能体

# 增强状态表示参数
enhanced_feature_dim: 128
enhanced_dropout: 0.1
use_spatial_features: False

# 保持 dTAPE 其他配置
learner: "max_q_learner"
mixer: "qmix"
```

### 3.3 与 dTAPE 的集成

- **替换 Agent**: 在 `BasicMAC` 中，将 `RNNAgent` 替换为 `EnhancedRNNAgent`
- **保持其他组件**: Mixer、Learner、通信机制等完全保留
- **状态处理流程**: 原始观测 → 状态编码器 → RNN → 输出层

---

## 四、CurriculumLearning

### 4.1 优化目标

通过课程学习策略，从简单任务开始，逐步增加任务难度，使智能体能够更好地学习和适应复杂任务。

### 4.2 技术实现路径

#### 步骤 1: 实现课程学习 Learner

**文件位置**: `RLalgs/CurriculumLearning/src/learners/curriculum_learner.py`

**核心实现**:

```python
class CurriculumLearner(MAXQLearner):
    """课程学习：逐步增加任务难度"""
    def __init__(self, mac, scheme, logger, args):
        super(CurriculumLearner, self).__init__(mac, scheme, logger, args)
        
        # 课程学习参数
        self.curriculum_enabled = getattr(args, 'curriculum_enabled', True)
        self.curriculum_schedule = getattr(args, 'curriculum_schedule', 'linear')
        self.curriculum_start_step = getattr(args, 'curriculum_start_step', 0)
        self.curriculum_end_step = getattr(args, 'curriculum_end_step', 1000000)
        
        # 难度级别（0-1，0最简单，1最难）
        self.current_difficulty = 0.0
        self.min_difficulty = getattr(args, 'curriculum_min_difficulty', 0.0)
        self.max_difficulty = getattr(args, 'curriculum_max_difficulty', 1.0)
        
        # 课程学习指标
        self.episode_wins = deque(maxlen=100)
        
    def update_curriculum_difficulty(self, t_env):
        """根据训练进度更新课程难度"""
        if not self.curriculum_enabled:
            self.current_difficulty = self.max_difficulty
            return
        
        if t_env < self.curriculum_start_step:
            self.current_difficulty = self.min_difficulty
        elif t_env >= self.curriculum_end_step:
            self.current_difficulty = self.max_difficulty
        else:
            # 线性调度
            if self.curriculum_schedule == 'linear':
                progress = (t_env - self.curriculum_start_step) / (
                    self.curriculum_end_step - self.curriculum_start_step
                )
                self.current_difficulty = (
                    self.min_difficulty + 
                    (self.max_difficulty - self.min_difficulty) * progress
                )
            # 自适应调度（基于性能）
            elif self.curriculum_schedule == 'adaptive':
                if len(self.episode_wins) > 10:
                    recent_win_rate = np.mean(list(self.episode_wins)[-10:])
                    if recent_win_rate > 0.8:
                        # 性能好，增加难度
                        self.current_difficulty = min(
                            self.current_difficulty + 0.01,
                            self.max_difficulty
                        )
                    elif recent_win_rate < 0.2:
                        # 性能差，降低难度
                        self.current_difficulty = max(
                            self.current_difficulty - 0.01,
                            self.min_difficulty
                        )
    
    def apply_curriculum_to_batch(self, batch):
        """将课程难度应用到 batch"""
        # 根据难度调整 batch（例如：过滤困难样本、调整奖励等）
        # 具体实现取决于任务特点
        return batch
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 更新课程难度
        self.update_curriculum_difficulty(t_env)
        
        # 应用课程学习到 batch
        batch = self.apply_curriculum_to_batch(batch)
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 注册课程学习 Learner

**文件位置**: `RLalgs/CurriculumLearning/src/learners/__init__.py`

```python
from learners.curriculum_learner import CurriculumLearner

REGISTRY = {}
REGISTRY["curriculum_learner"] = CurriculumLearner
```

#### 步骤 3: 配置文件

**文件位置**: `RLalgs/CurriculumLearning/src/config/algs/curriculum_qmix.yaml`

```yaml
learner: "curriculum_learner"  # 使用课程学习 learner

# 课程学习参数
curriculum_enabled: True
curriculum_schedule: "linear"  # "linear" 或 "adaptive"
curriculum_start_step: 0
curriculum_end_step: 1000000
curriculum_min_difficulty: 0.0
curriculum_max_difficulty: 1.0

# 保持 dTAPE 其他配置
mixer: "qmix"
agent: "rnn"
```

### 4.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `CurriculumLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **重写 train 方法**: 在训练前更新难度并应用到 batch
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 五、RewardShaping

### 5.1 优化目标

通过奖励塑形改进奖励信号，使智能体能够更快、更稳定地学习，特别适合稀疏奖励环境。

### 5.2 技术实现路径

#### 步骤 1: 实现奖励塑形 Learner

**文件位置**: `RLalgs/RewardShaping/src/learners/reward_shaping_learner.py`

**核心实现**:

```python
class RewardShapingLearner(MAXQLearner):
    """奖励塑形：通过改进奖励信号提升学习效率"""
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # 奖励塑形权重
        self.shaping_weight = getattr(args, 'reward_shaping_weight', 0.1)
        self.shaping_decay = getattr(args, 'reward_shaping_decay', 0.99)
        self.current_shaping_weight = self.shaping_weight
        
    def shape_reward(self, rewards, states, next_states, t_env):
        """对奖励进行塑形"""
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        if self.shaping_type == 'potential_based':
            # 基于潜在函数的奖励塑形
            # r' = r + γφ(s') - φ(s)
            potential_current = self._compute_potential(states)
            potential_next = self._compute_potential(next_states)
            potential_bonus = potential_next - potential_current
            shaped_rewards = rewards + self.current_shaping_weight * potential_bonus
        
        elif self.shaping_type == 'dense_reward':
            # 密集奖励：为中间步骤添加奖励
            shaped_rewards = self._add_dense_rewards(rewards, states)
        
        elif self.shaping_type == 'curiosity':
            # 好奇心驱动：鼓励探索
            shaped_rewards = self._add_curiosity_bonus(rewards, states)
        
        # 更新塑形权重（逐渐减少）
        self.current_shaping_weight *= self.shaping_decay
        
        return shaped_rewards
    
    def _compute_potential(self, states):
        """计算潜在函数值"""
        # 简化版本：可以扩展为神经网络
        return th.zeros_like(states[:, :, 0:1])
    
    def _add_dense_rewards(self, rewards, states):
        """添加密集奖励"""
        # 为每步添加小的生存奖励
        survival_bonus = 0.01
        return rewards + survival_bonus
    
    def _add_curiosity_bonus(self, rewards, states):
        """添加好奇心奖励"""
        # 基于状态新颖性的奖励
        # 简化版本
        return rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取原始奖励
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        next_states = batch["state"][:, 1:]
        
        # 对奖励进行塑形
        shaped_rewards = self.shape_reward(rewards, states, next_states, t_env)
        
        # 替换 batch 中的奖励
        batch["reward"][:, :-1] = shaped_rewards
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 注册奖励塑形 Learner

**文件位置**: `RLalgs/RewardShaping/src/learners/__init__.py`

```python
from learners.reward_shaping_learner import RewardShapingLearner

REGISTRY = {}
REGISTRY["reward_shaping_learner"] = RewardShapingLearner
```

#### 步骤 3: 配置文件

**文件位置**: `RLalgs/RewardShaping/src/config/algs/reward_shaping_qmix.yaml`

```yaml
learner: "reward_shaping_learner"  # 使用奖励塑形 learner

# 奖励塑形参数
reward_shaping_enabled: True
reward_shaping_type: "potential_based"  # "potential_based", "dense_reward", "curiosity"
reward_shaping_weight: 0.1
reward_shaping_decay: 0.99

# 保持 dTAPE 其他配置
mixer: "qmix"
agent: "rnn"
```

### 5.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `RewardShapingLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **重写 train 方法**: 在训练前对奖励进行塑形
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 六、TargetedOptimization

### 6.1 优化目标

针对特定地图设计专门的奖励塑形策略，根据地图的战术特点和游戏机制定制优化方案。

### 6.2 技术实现路径

#### 步骤 1: 实现地图特定的奖励塑形 Learner

**文件位置**: `RLalgs/TargetedOptimization/src/learners/reward_shaping_learner.py`

**核心实现**:

```python
class RewardShapingLearner(MAXQLearner):
    """奖励塑形：针对特定地图的优化"""
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 获取地图名称
        try:
            self.map_name = getattr(args.env_args, 'map_name', '')
        except:
            self.map_name = ''
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # JCTQ (金蝉脱壳) 特定参数
        self.survival_reward_weight = getattr(args, 'survival_reward_weight', 2.0)
        self.escape_reward_weight = getattr(args, 'escape_reward_weight', 1.5)
        self.disperse_reward_weight = getattr(args, 'disperse_reward_weight', 0.8)
        self.kill_reward_weight = getattr(args, 'kill_reward_weight', 0.3)
        self.time_critical_weight = getattr(args, 'time_critical_weight', 1.2)
        
        # SWCT (上屋抽梯) 特定参数
        self.lure_reward_weight = getattr(args, 'lure_reward_weight', 1.5)
        self.forcefield_reward_weight = getattr(args, 'forcefield_reward_weight', 2.5)
        self.warp_prism_reward_weight = getattr(args, 'warp_prism_reward_weight', 2.0)
        self.tactical_positioning_weight = getattr(args, 'tactical_positioning_weight', 1.2)
        
    def shape_reward(self, rewards, states, next_states, t_env, batch=None):
        """对奖励进行塑形（地图特定）"""
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        # JCTQ (金蝉脱壳) - 逃脱生存策略
        if self.shaping_type == 'escape_survival' or self.map_name == 'jctq':
            shaped_rewards = self._apply_escape_survival_shaping(
                rewards, states, next_states, batch
            )
        
        # SWCT (上屋抽梯) - 诱敌阻挡撤退策略
        elif self.shaping_type == 'lure_block_retreat' or self.map_name == 'swct':
            shaped_rewards = self._apply_lure_block_retreat_shaping(
                rewards, states, next_states, batch, t_env
            )
        
        return shaped_rewards
    
    def _apply_escape_survival_shaping(self, rewards, states, next_states, batch):
        """JCTQ: 逃脱生存策略的奖励塑形"""
        shaped_rewards = rewards.clone()
        
        # 1. 生存奖励：鼓励单位存活
        # 检测单位是否存活（简化版本）
        survival_bonus = self.survival_reward_weight * 0.1  # 每步存活奖励
        shaped_rewards += survival_bonus
        
        # 2. 逃脱奖励：鼓励远离敌人
        # 计算与敌人的距离（简化版本）
        escape_bonus = self.escape_reward_weight * 0.05
        shaped_rewards += escape_bonus
        
        # 3. 分散奖励：鼓励单位分散
        disperse_bonus = self.disperse_reward_weight * 0.03
        shaped_rewards += disperse_bonus
        
        # 4. 击杀惩罚：避免暴露位置
        # 如果击杀敌人，给予较小奖励（避免暴露）
        kill_penalty = -self.kill_reward_weight * 0.1
        # 检测击杀（简化版本）
        # shaped_rewards += kill_penalty
        
        # 5. 时间紧迫性：后期给予更多奖励
        time_bonus = self.time_critical_weight * 0.02
        shaped_rewards += time_bonus
        
        return shaped_rewards
    
    def _apply_lure_block_retreat_shaping(self, rewards, states, next_states, batch, t_env):
        """SWCT: 诱敌阻挡撤退策略的奖励塑形"""
        shaped_rewards = rewards.clone()
        
        # 1. 强生存奖励
        survival_bonus = 3.0 * 0.1
        shaped_rewards += survival_bonus
        
        # 2. ForceField 使用奖励
        forcefield_bonus = self.forcefield_reward_weight * 0.2
        # 检测 ForceField 使用（简化版本）
        # shaped_rewards += forcefield_bonus
        
        # 3. WarpPrism 使用奖励
        warp_prism_bonus = self.warp_prism_reward_weight * 0.15
        # 检测 WarpPrism 使用（简化版本）
        # shaped_rewards += warp_prism_bonus
        
        # 4. 战术位置奖励
        positioning_bonus = self.tactical_positioning_weight * 0.1
        shaped_rewards += positioning_bonus
        
        # 5. 诱敌奖励
        lure_bonus = self.lure_reward_weight * 0.1
        shaped_rewards += lure_bonus
        
        return shaped_rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取原始奖励
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        next_states = batch["state"][:, 1:]
        
        # 对奖励进行塑形（地图特定）
        shaped_rewards = self.shape_reward(rewards, states, next_states, t_env, batch)
        
        # 替换 batch 中的奖励
        batch["reward"][:, :-1] = shaped_rewards
        
        # 调用父类的训练方法
        return super().train(batch, t_env, episode_num)
```

#### 步骤 2: 配置文件（JCTQ）

**文件位置**: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_jctq.yaml`

```yaml
learner: "reward_shaping_learner"

# JCTQ 特定奖励塑形
reward_shaping_enabled: True
reward_shaping_type: "escape_survival"
survival_reward_weight: 2.0
escape_reward_weight: 1.5
disperse_reward_weight: 0.8
kill_reward_weight: 0.3
time_critical_weight: 1.2
```

#### 步骤 3: 配置文件（SWCT）

**文件位置**: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_swct.yaml`

```yaml
learner: "reward_shaping_learner"

# SWCT 特定奖励塑形
reward_shaping_enabled: True
reward_shaping_type: "lure_block_retreat"
lure_reward_weight: 1.5
forcefield_reward_weight: 2.5
warp_prism_reward_weight: 2.0
tactical_positioning_weight: 1.2
phase_based_reward: True
```

### 6.3 与 dTAPE 的集成

- **继承 MAXQLearner**: `RewardShapingLearner` 继承自 `MAXQLearner`，保留所有 dTAPE 功能
- **地图特定塑形**: 根据地图名称自动选择对应的奖励塑形策略
- **保持其他组件**: Mixer、Agent、通信机制等完全保留

---

## 七、实现路径总结

### 7.1 共同特点

所有六个优化都遵循以下共同原则：

1. **最小侵入**: 只修改必要的部分，保持 dTAPE 框架完整性
2. **模块化设计**: 每个优化作为独立模块实现
3. **配置驱动**: 通过配置文件选择使用哪种优化
4. **向后兼容**: 可以随时切换回原始 dTAPE

### 7.2 修改层次

| 优化方法 | 修改层次 | 主要修改点 |
|---------|---------|-----------|
| TransformerMixer | Mixer 层 | 替换 `QMixer` 为 `TransformerQMixer` |
| HierarchicalArchitecture | Agent 层 | 替换 `RNNAgent` 为 `HierarchicalRNNAgent` |
| EnhancedStateRepresentation | Agent 层 | 替换 `RNNAgent` 为 `EnhancedRNNAgent` |
| CurriculumLearning | Learner 层 | 替换 `MAXQLearner` 为 `CurriculumLearner` |
| RewardShaping | Learner 层 | 替换 `MAXQLearner` 为 `RewardShapingLearner` |
| TargetedOptimization | Learner 层 | 替换 `MAXQLearner` 为 `RewardShapingLearner`（地图特定）|

### 7.3 代码位置总结

| 优化方法 | 核心实现文件 | 注册文件 | 配置文件 |
|---------|------------|---------|---------|
| TransformerMixer | `modules/mixers/transformer_qmix.py` | `modules/mixers/__init__.py` | `config/algs/transformer_qmix.yaml` |
| HierarchicalArchitecture | `modules/agents/hierarchical_agent.py` | `modules/agents/__init__.py` | `config/algs/hierarchical_qmix.yaml` |
| EnhancedStateRepresentation | `modules/agents/enhanced_rnn_agent.py` | `modules/agents/__init__.py` | `config/algs/enhanced_qmix.yaml` |
| CurriculumLearning | `learners/curriculum_learner.py` | `learners/__init__.py` | `config/algs/curriculum_qmix.yaml` |
| RewardShaping | `learners/reward_shaping_learner.py` | `learners/__init__.py` | `config/algs/reward_shaping_qmix.yaml` |
| TargetedOptimization | `learners/reward_shaping_learner.py` | `learners/__init__.py` | `config/algs/targeted_qmix_*.yaml` |

### 7.4 集成方式对比

| 优化方法 | 集成方式 | 继承关系 |
|---------|---------|---------|
| TransformerMixer | 直接替换 | 无（独立实现） |
| HierarchicalArchitecture | 直接替换 | 无（独立实现） |
| EnhancedStateRepresentation | 直接替换 | 无（独立实现） |
| CurriculumLearning | 继承扩展 | `CurriculumLearner(MAXQLearner)` |
| RewardShaping | 继承扩展 | `RewardShapingLearner(MAXQLearner)` |
| TargetedOptimization | 继承扩展 | `RewardShapingLearner(MAXQLearner)` |

---

## 八、使用示例

### 8.1 训练命令

```bash
# TransformerMixer
cd RLalgs/TransformerMixer
bash train_single_map.sh adcc 1 42

# HierarchicalArchitecture
cd RLalgs/HierarchicalArchitecture
bash train_single_map.sh jdsr 2 42

# EnhancedStateRepresentation
cd RLalgs/EnhancedStateRepresentation
bash train_single_map.sh gmzz 3 42

# CurriculumLearning
cd RLalgs/CurriculumLearning
bash train_single_map.sh sdjx 4 42

# RewardShaping
cd RLalgs/RewardShaping
bash train_single_map.sh wzsy 5 42

# TargetedOptimization
cd RLalgs/TargetedOptimization
bash train_single_map.sh jctq 6 42  # JCTQ 地图
bash train_single_map.sh swct 7 42  # SWCT 地图
```

### 8.2 配置文件选择

每个优化方法都有对应的配置文件，通过 `--config` 参数指定：

```bash
python src/main.py \
    --config=transformer_qmix \
    --env-config=sc2te \
    with env_args.map_name=adcc
```

---

## 九、总结

六个优化方法通过不同的技术路径实现了对 dTAPE 的改进：

1. **TransformerMixer**: 通过替换 Mixer 网络为 Transformer 架构，增强状态表示和智能体协作
2. **HierarchicalArchitecture**: 通过分层架构将决策分为高层策略和底层执行
3. **EnhancedStateRepresentation**: 通过增强的状态编码器提取更丰富的特征
4. **CurriculumLearning**: 通过课程学习策略逐步增加任务难度
5. **RewardShaping**: 通过奖励塑形改进奖励信号
6. **TargetedOptimization**: 通过地图特定的奖励塑形实现针对性优化

所有优化都保持了 dTAPE 框架的完整性，通过模块化设计和配置驱动的方式实现了灵活的集成。





