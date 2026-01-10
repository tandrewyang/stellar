# 基于地图名称含义的新优化方案

## 问题分析

jctq和swct在所有算法中胜率均为0.00%，说明需要根据地图名称的战术含义来设计优化方案。

---

## jctq (金蝉脱壳) - 新优化方案

### 地图名称含义
**"金蝉脱壳"**：像蝉脱壳一样，留下假象，自己脱身。核心是**逃脱和生存**。

### 地图特点
- 4 agents vs 9 enemies（劣势）
- 45步（极短时间）
- 有Burrow（埋地）机制
- 胜利条件：至少存活1个单位

### 核心优化策略

#### 1. **逃脱优先策略 (Escape-First Strategy)**
- **奖励设计**：大幅奖励存活和逃脱，而非击杀
- **存活奖励**：每个单位存活到结束 +1.0
- **逃脱奖励**：使用Burrow逃脱 +0.5
- **分散奖励**：单位分散（避免被一网打尽）+0.3
- **击杀惩罚**：击杀敌人奖励降低（因为会暴露位置）

#### 2. **时间紧迫性感知 (Time-Critical Awareness)**
- **时间奖励衰减**：越接近结束，存活奖励越高
- **早期逃脱奖励**：前15步逃脱 +0.8，后30步逃脱 +0.3
- **时间压力编码**：在状态中加入剩余步数/总步数比例

#### 3. **分散-隐藏-逃脱战术 (Disperse-Hide-Escape)**
- **分散奖励**：单位之间距离越远，奖励越高
- **隐藏奖励**：使用Burrow隐藏 +0.4
- **逃脱路径规划**：奖励向地图边缘移动

#### 4. **生存优先动作空间**
- 优先学习：移动、Burrow、分散
- 降低学习：攻击、聚集

### 配置文件更新

```yaml
# 针对jctq的"金蝉脱壳"策略
reward_shaping_enabled: True
reward_shaping_type: "escape_survival"  # 逃脱生存型
survival_reward_weight: 2.0  # 存活奖励权重（大幅提高）
escape_reward_weight: 1.5  # 逃脱奖励权重
disperse_reward_weight: 0.8  # 分散奖励权重
kill_reward_weight: 0.3  # 击杀奖励权重（降低）
time_critical_weight: 1.2  # 时间紧迫性权重
epsilon_anneal_time: 30000  # 更快的探索衰减（45步需要快速决策）
lr: 0.0008  # 稍大的学习率（快速学习逃脱策略）
```

---

## swct (上屋抽梯) - 新优化方案

### 地图名称含义
**"上屋抽梯"**：把对方引到高处然后撤掉梯子，让对方无法下来。核心是**诱敌深入 + 战术撤退**。

### 地图特点
- 5 agents vs 11 enemies（劣势）
- 200步
- 有WarpPrism（传送门）和ForceField（力场）机制

### 核心优化策略

#### 1. **诱敌-阻挡-撤退策略 (Lure-Block-Retreat)**
- **诱敌奖励**：将敌人引到特定位置 +0.4
- **力场阻挡奖励**：使用ForceField阻挡敌人 +0.6
- **传送撤退奖励**：使用WarpPrism撤退 +0.5
- **战术位置奖励**：占据有利位置（高地、狭窄通道）+0.3

#### 2. **力场战术使用 (ForceField Tactics)**
- **阻挡奖励**：ForceField成功阻挡敌人 +0.8
- **分割奖励**：ForceField分割敌人阵型 +0.6
- **撤退掩护奖励**：ForceField掩护撤退 +0.5

#### 3. **传送门战术撤退 (WarpPrism Tactical Evacuation)**
- **撤退奖励**：使用WarpPrism撤退 +0.5
- **重新部署奖励**：撤退后重新部署到有利位置 +0.4
- **单位保护奖励**：通过传送保护单位 +0.6

#### 4. **分阶段战术 (Phased Tactics)**
- **阶段1（前50步）**：诱敌深入，奖励将敌人引到力场区域
- **阶段2（中100步）**：使用力场阻挡，奖励战术阻挡
- **阶段3（后50步）**：撤退和重新部署，奖励传送撤退

### 配置文件更新

```yaml
# 针对swct的"上屋抽梯"策略
reward_shaping_enabled: True
reward_shaping_type: "lure_block_retreat"  # 诱敌-阻挡-撤退型
lure_reward_weight: 1.2  # 诱敌奖励权重
forcefield_reward_weight: 2.0  # 力场奖励权重（核心机制）
warp_prism_reward_weight: 1.5  # 传送门奖励权重
tactical_positioning_weight: 1.0  # 战术位置权重
phase_based_reward: True  # 启用分阶段奖励
epsilon_anneal_time: 80000  # 中等探索衰减
lr: 0.001  # 标准学习率
```

---

## 实现要点

### 1. 奖励函数修改
需要在learner中实现地图特定的奖励塑形：
- jctq: `escape_survival_reward_shaping()`
- swct: `lure_block_retreat_reward_shaping()`

### 2. 状态编码增强
- jctq: 添加时间紧迫性、单位分散度、逃脱路径
- swct: 添加力场状态、传送门状态、敌人位置分布

### 3. 动作空间优化
- jctq: 优先学习逃脱相关动作
- swct: 优先学习力场和传送门使用

### 4. 课程学习
- jctq: 从简单逃脱到复杂分散逃脱
- swct: 从单一机制使用到组合战术

---

## 预期效果

### jctq (金蝉脱壳)
- **目标胜率**: 从0.00%提升到至少30%+
- **关键指标**: 存活单位数、逃脱成功率

### swct (上屋抽梯)
- **目标胜率**: 从0.00%提升到至少40%+
- **关键指标**: 力场使用效率、传送撤退成功率

