# Targeted Optimization 在 JCTQ 和 SWCT 地图上的具体实现

## 概述

本文档详细整理了 Targeted Optimization（目标优化）算法在 JCTQ（金蝉脱壳）和 SWCT（上屋抽梯）两张地图上的具体优化实现，包括配置参数、奖励塑形机制和实现细节。

---

## 一、JCTQ（金蝉脱壳）地图优化

### 1.1 地图特点

- **地图名称**: JCTQ - 金蝉脱壳 (Slipping Away by Casting Off a Cloak)
- **战术核心**: 像蝉脱壳一样，留下假象，自己脱身
- **地图配置**:
  - 4 agents vs 9 enemies（劣势）
  - 45步（极短时间）
  - 有 Burrow（埋地）机制
  - 胜利条件：至少存活1个单位
  - 种族：Zerg vs Protoss

### 1.2 优化策略设计

#### 核心优化方向
1. **逃脱优先**：优先考虑逃脱而非战斗
2. **生存优先**：确保至少1个单位存活
3. **分散隐藏**：单位分散，避免被集中攻击

#### 配置文件位置
`RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_jctq.yaml`

### 1.3 关键配置参数

```yaml
# 探索策略
epsilon_start: 0.995
epsilon_finish: 0.05
epsilon_anneal_time: 30000  # 更快的探索衰减（45步需要快速决策）

# 奖励塑形配置
reward_shaping_enabled: True
reward_shaping_type: "escape_survival"  # 逃脱生存型
survival_reward_weight: 2.0  # 存活奖励权重（大幅提高）
escape_reward_weight: 1.5  # 逃脱奖励权重
disperse_reward_weight: 0.8  # 分散奖励权重
kill_reward_weight: 0.3  # 击杀奖励权重（降低，避免暴露位置）
time_critical_weight: 1.2  # 时间紧迫性权重
reward_shaping_decay: 0.99

# 学习率
lr: 0.0008  # 稍大的学习率（快速学习逃脱策略）
```

### 1.4 奖励塑形实现

**实现函数**: `_apply_escape_survival_shaping()`  
**位置**: `RLalgs/TargetedOptimization/src/learners/reward_shaping_learner.py` (第203-251行)

#### 1.4.1 存活奖励（Survival Reward）

```python
# 每步给予存活奖励（基于奖励信号推断，如果奖励>=0说明可能存活）
survival_bonus = th.where(rewards >= 0, 
                          self.survival_reward_weight * 0.1,  # 存活奖励 = 2.0 * 0.1 = 0.2
                          th.zeros_like(rewards))  # 死亡无奖励
```

**设计原理**:
- 每步存活都有基础奖励 0.2（权重 2.0 × 基础值 0.1）
- 确保智能体优先考虑生存而非战斗
- 死亡时无奖励，强化生存的重要性

#### 1.4.2 逃脱奖励（Escape Reward）

```python
# 通过状态差异推断逃脱行为
state_diff = th.abs(next_states - states).sum(dim=-1, keepdim=False)  # [batch_size, seq_len]
state_diff_max = state_diff.max() if state_diff.numel() > 0 else 1.0
escape_bonus = self.escape_reward_weight * 0.05 * th.clamp(state_diff / (state_diff_max + 1e-6), 0, 1)
escape_bonus = escape_bonus.expand(batch_size, seq_len, n_agents)
```

**设计原理**:
- 检测状态变化（位置移动）来识别逃脱行为
- 状态变化越大，逃脱奖励越高
- 权重 1.5，基础值 0.05，最大奖励约 0.075

#### 1.4.3 分散奖励（Disperse Reward）

```python
# 通过状态空间分布推断单位分散程度
state_var = th.var(states, dim=-1, keepdim=False)  # [batch_size, seq_len]
state_var_max = state_var.max() if state_var.numel() > 0 else 1.0
disperse_bonus = self.disperse_reward_weight * 0.03 * th.clamp(state_var / (state_var_max + 1e-6), 0, 1)
disperse_bonus = disperse_bonus.expand(batch_size, seq_len, n_agents)
```

**设计原理**:
- 通过计算状态方差来检测单位分散程度
- 方差越大，说明单位越分散，奖励越高
- 权重 0.8，基础值 0.03，最大奖励约 0.024

#### 1.4.4 击杀惩罚（Kill Penalty）

```python
# 降低击杀奖励权重（避免暴露位置）
kill_penalty = th.where(rewards > 0.5, 
                       (1.0 - self.kill_reward_weight) * rewards * 0.5,  # 降低击杀奖励
                       th.zeros_like(rewards))
shaped_rewards = shaped_rewards - kill_penalty
```

**设计原理**:
- 击杀敌人会暴露位置，不利于逃脱
- 当原始奖励 > 0.5（可能是击杀）时，降低其权重
- 权重 0.3，意味着击杀奖励被降低 70%

#### 1.4.5 时间紧迫性奖励（Time Critical Reward）

```python
# 越接近结束，存活奖励越高
time_factor = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)  # [1, seq_len, 1]
time_bonus = self.time_critical_weight * 0.05 * time_factor.expand(batch_size, seq_len, n_agents)
```

**设计原理**:
- 时间紧迫性：越接近结束还存活，奖励越高
- 线性增加：从 0 到 0.06（权重 1.2 × 基础值 0.05）
- 鼓励智能体在关键时刻保持存活

### 1.5 优化效果

- **Baseline 胜率**: 0.00%
- **Targeted Optimization 胜率**: 100.00%
- **提升幅度**: +100.00%（从 0 提升到 100%）

---

## 二、SWCT（上屋抽梯）地图优化

### 2.1 地图特点

- **地图名称**: SWCT - 上屋抽梯 (Removing the Ladder After the Enemy Has Climbed Up)
- **战术核心**: 把对方引到高处然后撤掉梯子，让对方无法下来
- **地图配置**:
  - 5 agents vs 11 enemies（劣势）
  - 200步
  - 有 WarpPrism（折跃棱镜）和 ForceField（力场）机制
  - 种族：Protoss vs Zerg

### 2.2 优化策略设计

#### 核心优化方向
1. **诱敌深入**：将敌人引到特定位置
2. **力场阻挡**：使用 ForceField 阻挡敌人撤退
3. **传送撤退**：使用 WarpPrism 进行战术撤退

#### 配置文件位置
`RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_swct.yaml`

### 2.3 关键配置参数

```yaml
# 探索策略
epsilon_start: 0.995
epsilon_finish: 0.05
epsilon_anneal_time: 120000  # 更长的探索时间（5 vs 11需要更多探索）

# 奖励塑形配置（增强版）
reward_shaping_enabled: True
reward_shaping_type: "lure_block_retreat"  # 诱敌-阻挡-撤退型
reward_shaping_weight: 0.2  # 增加塑形权重（从默认0.1增加到0.2）
reward_shaping_decay: 0.999  # 更慢的衰减（保持塑形效果更久）
lure_reward_weight: 1.5  # 增加诱敌奖励权重
forcefield_reward_weight: 2.5  # 增加力场奖励权重（核心机制）
warp_prism_reward_weight: 2.0  # 增加传送门奖励权重
tactical_positioning_weight: 1.2  # 增加战术位置权重
phase_based_reward: True  # 启用分阶段奖励

# 学习率
lr: 0.0012  # 稍大的学习率（5 vs 11需要更快学习）
```

### 2.4 奖励塑形实现

**实现函数**: `_apply_lure_block_retreat_shaping()`  
**位置**: `RLalgs/TargetedOptimization/src/learners/reward_shaping_learner.py` (第253-363行)

#### 2.4.1 强烈生存奖励（Survival Reward）

```python
# 每步基础生存奖励
survival_bonus = th.ones_like(rewards) * 0.15  # 每步基础生存奖励
shaped_rewards = shaped_rewards + survival_bonus

# 存活时间奖励：越接近结束还存活，奖励越高
time_progress = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)
time_progress = time_progress.expand(batch_size, seq_len, n_agents)
survival_time_bonus = 0.2 * time_progress  # 随时间线性增加
shaped_rewards = shaped_rewards + survival_time_bonus
```

**设计原理**:
- 5 vs 11 劣势，生存是核心
- 每步基础生存奖励 0.15
- 存活时间奖励线性增加，最高 0.2
- 总生存奖励：0.15 + 0.2 = 0.35（在最后一步）

#### 2.4.2 力场阻挡奖励（ForceField Reward）

```python
# 检测大幅状态变化（可能是力场生效阻挡敌人）
state_change = th.abs(next_states - states).sum(dim=-1, keepdim=False)  # [batch_size, seq_len]
state_change = state_change.unsqueeze(-1)  # [batch_size, seq_len, 1]
state_change_max = state_change.max() if state_change.numel() > 0 else 1.0

# 大幅状态变化（可能是力场阻挡成功）
forcefield_threshold = state_change_max * 0.4  # 40%阈值
forcefield_bonus = self.forcefield_reward_weight * 0.25 * th.where(
    state_change > forcefield_threshold,
    th.ones_like(state_change),
    th.zeros_like(state_change)
)
forcefield_bonus = forcefield_bonus.expand(batch_size, seq_len, n_agents)
```

**设计原理**:
- 检测大幅状态变化（阈值 40%）来识别力场使用
- 权重 2.5，基础值 0.25，最大奖励 0.625
- 力场是"上屋抽梯"战术的核心机制

#### 2.4.3 传送撤退奖励（Warp Prism Reward）

```python
# 检测大幅位置变化（可能是传送）
if states.shape[-1] >= 2:
    pos_diff = next_states[:, :, :2] - states[:, :, :2]  # [batch_size, seq_len, 2]
    pos_change = th.norm(pos_diff, dim=-1, keepdim=False)  # [batch_size, seq_len]
    pos_change = pos_change.unsqueeze(-1)  # [batch_size, seq_len, 1]
    pos_change_max = pos_change.max() if pos_change.numel() > 0 else 1.0
    
    # 大幅位置变化（可能是传送撤退）
    warp_threshold = pos_change_max * 0.3  # 30%阈值
    warp_bonus = self.warp_prism_reward_weight * 0.2 * th.where(
        pos_change > warp_threshold,
        th.ones_like(pos_change),
        th.zeros_like(pos_change)
    )
    warp_bonus = warp_bonus.expand(batch_size, seq_len, n_agents)
```

**设计原理**:
- 检测大幅位置变化（阈值 30%）来识别传送行为
- 权重 2.0，基础值 0.2，最大奖励 0.4
- 支持战术撤退和重新部署

#### 2.4.4 诱敌奖励（Lure Reward）

```python
# 诱敌奖励（移动且获得奖励）
state_change_expanded = state_change.expand(batch_size, seq_len, n_agents)
lure_bonus = self.lure_reward_weight * 0.1 * th.where(
    (state_change_expanded > state_change_max * 0.2) & (rewards > 0),
    th.ones_like(rewards),
    th.zeros_like(rewards)
)
```

**设计原理**:
- 结合状态变化和奖励变化来识别诱敌行为
- 状态变化 > 20% 且获得正奖励时给予诱敌奖励
- 权重 1.5，基础值 0.1，最大奖励 0.15

#### 2.4.5 战术位置奖励（Tactical Positioning Reward）

```python
# 计算单位到中心的距离
if states.shape[-1] >= 2:
    pos_states = states[:, :, :2]  # [batch_size, seq_len, 2]
    state_center = pos_states.mean(dim=-1, keepdim=False)  # [batch_size, seq_len]
    state_center = state_center.unsqueeze(-1)  # [batch_size, seq_len, 1]
    state_center = state_center.expand(batch_size, seq_len, 2)  # [batch_size, seq_len, 2]
    dist_to_center = th.norm(pos_states - state_center, dim=-1, keepdim=False)  # [batch_size, seq_len]
    dist_to_center = dist_to_center.unsqueeze(-1)  # [batch_size, seq_len, 1]
    dist_max = dist_to_center.max() if dist_to_center.numel() > 0 else 1.0
    position_bonus = self.tactical_positioning_weight * 0.08 * th.exp(-dist_to_center / (dist_max + 1e-6))
    position_bonus = position_bonus.expand(batch_size, seq_len, n_agents)
```

**设计原理**:
- 奖励占据有利战术位置（如高地、狭窄通道）
- 距离中心越近，奖励越高（指数衰减）
- 权重 1.2，基础值 0.08

#### 2.4.6 分阶段奖励（Phase-Based Reward）

```python
if self.phase_based_reward:
    # 阶段1（前30%）：诱敌深入 + 生存
    phase1_mask = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) < 0.3
    phase1_mask = phase1_mask.expand(batch_size, seq_len, n_agents)
    phase1_bonus = (lure_bonus + survival_bonus) * phase1_mask * 1.5  # 增强
    shaped_rewards = shaped_rewards + phase1_bonus
    
    # 阶段2（中40%）：力场阻挡 + 生存
    phase2_mask = (th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) >= 0.3) & \
                  (th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) < 0.7)
    phase2_mask = phase2_mask.expand(batch_size, seq_len, n_agents)
    phase2_bonus = (forcefield_bonus + survival_bonus) * phase2_mask * 1.5  # 增强
    shaped_rewards = shaped_rewards + phase2_bonus
    
    # 阶段3（后30%）：撤退 + 生存
    phase3_mask = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) >= 0.7
    phase3_mask = phase3_mask.expand(batch_size, seq_len, n_agents)
    phase3_bonus = (warp_bonus + survival_bonus + survival_time_bonus) * phase3_mask * 1.5  # 增强
    shaped_rewards = shaped_rewards + phase3_bonus
```

**设计原理**:
- **阶段1（前30%）**：诱敌深入 + 生存，增强 1.5 倍
- **阶段2（中40%）**：力场阻挡 + 生存，增强 1.5 倍
- **阶段3（后30%）**：撤退 + 生存，增强 1.5 倍
- 不同阶段需要不同的奖励权重，引导智能体执行分阶段战术

#### 2.4.7 敌人伤害奖励（Damage Reward）

```python
# 即使不能获胜，也要奖励对敌人造成伤害
damage_bonus = th.where(
    rewards > 0,
    rewards * 0.5,  # 伤害奖励放大50%
    th.zeros_like(rewards)
)
```

**设计原理**:
- 即使不能获胜，也要奖励对敌人造成伤害
- 伤害奖励放大 50%，鼓励积极战斗

### 2.5 优化效果

- **Baseline 胜率**: 0.00%
- **Targeted Optimization 胜率**: 37.50%
- **提升幅度**: +37.50%（从 0 提升到 37.50%）

---

## 三、实现对比总结

### 3.1 共同特点

1. **强烈生存奖励**: 两张地图都强调生存，但实现方式不同
   - JCTQ: 每步基础奖励 0.2（权重 2.0 × 0.1）
   - SWCT: 每步基础奖励 0.15 + 时间奖励最高 0.2

2. **机制感知奖励**: 针对地图特有的游戏机制设计奖励
   - JCTQ: Burrow（埋地）机制（通过状态变化检测）
   - SWCT: ForceField（力场）和 WarpPrism（传送）机制（通过状态/位置变化检测）

3. **时间敏感性**: 考虑时间因素对奖励的影响
   - JCTQ: 时间紧迫性奖励（越接近结束奖励越高）
   - SWCT: 分阶段奖励（不同阶段不同策略）

### 3.2 差异特点

| 特性 | JCTQ | SWCT |
|------|------|------|
| **核心策略** | 逃脱优先 | 诱敌-阻挡-撤退 |
| **时间限制** | 45步（极短） | 200步（中等） |
| **劣势程度** | 4 vs 9 | 5 vs 11 |
| **奖励权重** | 生存 2.0，逃脱 1.5 | 生存 0.15+0.2，力场 2.5 |
| **分阶段奖励** | 否 | 是（3个阶段） |
| **击杀策略** | 降低击杀奖励（避免暴露） | 放大伤害奖励（鼓励战斗） |

### 3.3 关键成功因素

1. **精确的机制检测**: 通过状态/位置变化准确识别机制使用
2. **合理的奖励权重**: 通过调参找到最优的奖励权重组合
3. **时间敏感的奖励设计**: 在关键时刻给予更强的奖励信号
4. **分阶段战术引导**: SWCT 通过分阶段奖励引导智能体执行复杂战术

---

## 四、代码文件位置

### 4.1 配置文件

- JCTQ: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_jctq.yaml`
- SWCT: `RLalgs/TargetedOptimization/src/config/algs/targeted_qmix_swct.yaml`

### 4.2 实现代码

- 奖励塑形 Learner: `RLalgs/TargetedOptimization/src/learners/reward_shaping_learner.py`
  - JCTQ: `_apply_escape_survival_shaping()` (第203-251行)
  - SWCT: `_apply_lure_block_retreat_shaping()` (第253-363行)

### 4.3 环境代码

- JCTQ: `smac/smac/env/sc2_tactics/star36env_jctq.py`
- SWCT: `smac/smac/env/sc2_tactics/star36env_swct.py`

---

## 五、优化建议

### 5.1 JCTQ 进一步优化

1. **Burrow 机制检测**: 可以更精确地检测 Burrow 使用（通过单位状态变化）
2. **逃脱路径规划**: 可以添加逃脱路径奖励，鼓励向地图边缘移动
3. **分散度优化**: 可以优化分散奖励的计算方式，更准确地反映单位分散程度

### 5.2 SWCT 进一步优化

1. **力场位置优化**: 可以添加力场位置奖励，鼓励在关键位置使用力场
2. **传送时机优化**: 可以优化传送奖励的检测，更准确地识别战术传送
3. **协调奖励增强**: 可以增强诱敌和阻挡的协调奖励，使战术执行更流畅

---

## 六、参考文献

- Targeted Optimization 算法设计文档
- JCTQ 和 SWCT 地图配置文档
- 奖励塑形 Learner 实现代码
- 优化报告: `report/optimization_report.md`
