# RewardShaping - 奖励塑形优化

## 优化思路

通过**奖励塑形（Reward Shaping）**改进奖励信号，使智能体能够更快、更稳定地学习，特别适合稀疏奖励环境。

## 核心改进

### 1. 奖励塑形策略

#### 基于潜在函数的奖励塑形（Potential-Based）
- **原理**: 使用潜在函数φ(s)来塑形奖励
- **公式**: r' = r + γφ(s') - φ(s)
- **优势**: 保证最优策略不变（policy invariant）
- **应用**: 鼓励接近目标状态，避免危险状态

#### 密集奖励（Dense Reward）
- **原理**: 为中间步骤添加小的奖励信号
- **优势**: 提供更频繁的学习信号
- **应用**: 每步给予小的生存奖励、进度奖励等

#### 好奇心驱动（Curiosity）
- **原理**: 鼓励探索未知状态
- **优势**: 提高探索效率
- **应用**: 基于预测误差或状态新颖性的奖励

### 2. 技术细节

#### 奖励塑形流程
```
原始奖励 → 奖励塑形函数 → 塑形后奖励 → 训练
```

#### 关键参数
- `reward_shaping_enabled`: True - 是否启用奖励塑形
- `reward_shaping_type`: "potential_based" - 塑形类型
- `reward_shaping_weight`: 0.1 - 塑形奖励权重
- `reward_shaping_decay`: 0.99 - 塑形权重衰减率

#### 塑形类型
1. **potential_based**: 基于潜在函数的奖励塑形
2. **dense_reward**: 密集奖励
3. **curiosity**: 好奇心驱动

### 3. 优势

1. **更快学习**: 提供更频繁、更有用的学习信号
2. **稳定训练**: 减少稀疏奖励带来的训练不稳定
3. **策略不变**: 基于潜在函数的塑形保证最优策略不变
4. **灵活调整**: 可以根据任务特点选择不同的塑形方式

### 4. 预期效果

- 更快的收敛速度
- 更高的任务完成率
- 更稳定的训练过程
- 更好的探索能力

## 使用方法

### 训练单个地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/RewardShaping/train_single_map.sh adcc 5 42
```

### 训练所有HLSMAC地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/RewardShaping/train_hlsmac.sh 5 42
```

## 配置文件

主要配置文件: `src/config/algs/reward_shaping_qmix.yaml`

### 关键参数说明

- `learner: "reward_shaping_learner"` - 使用奖励塑形learner
- `reward_shaping_enabled: True` - 启用奖励塑形
- `reward_shaping_type: "potential_based"` - 塑形类型
- `reward_shaping_weight: 0.1` - 塑形奖励权重（建议0.05-0.2）
- `reward_shaping_decay: 0.99` - 塑形权重衰减率

### 调优建议

1. **塑形类型选择**:
   - **potential_based**: 适合需要保证策略不变的任务
   - **dense_reward**: 适合稀疏奖励环境
   - **curiosity**: 适合需要大量探索的任务

2. **权重设置**:
   - 初始权重建议0.05-0.2
   - 过大可能干扰原始奖励信号
   - 过小可能效果不明显

3. **衰减率**:
   - 建议0.99-0.999
   - 逐渐减少塑形影响，让模型依赖原始奖励

## 实现细节

### 奖励塑形流程
```
原始奖励 r
    ↓
塑形函数 f(s, s', r)
    ↓
塑形后奖励 r' = r + α * f(s, s', r)
    ↓
训练使用 r'
```

### 关键创新点
1. **多种塑形方式**: 支持多种奖励塑形策略
2. **权重衰减**: 逐渐减少塑形影响
3. **策略不变性**: 基于潜在函数的塑形保证最优策略不变
4. **易于扩展**: 可以添加更多塑形策略

## 与Baseline对比

| 特性 | Baseline | RewardShaping |
|------|----------|---------------|
| 奖励信号 | 稀疏 | 密集/塑形 |
| 学习速度 | 中等 | 更快 |
| 训练稳定性 | 中等 | 更高 |
| 探索能力 | 中等 | 更强（curiosity）|
| 策略不变性 | N/A | 保证（potential_based）|

## 实验建议

1. **塑形类型对比**: 测试不同塑形类型的效果
2. **权重调优**: 测试不同塑形权重的影响
3. **消融实验**: 测试奖励塑形对性能的贡献
4. **任务特定**: 针对不同HLSMAC地图设计特定塑形

## 扩展方向

1. **学习潜在函数**: 使用神经网络学习潜在函数
2. **自适应塑形**: 根据训练进度自动调整塑形
3. **多目标塑形**: 同时优化多个目标
4. **逆强化学习**: 从专家演示中学习奖励函数

## 参考文献

- Policy Invariant Reward Shaping
- Potential-Based Shaping and Q-Value Initialization
- Intrinsic Motivation and Curiosity-Driven Learning

