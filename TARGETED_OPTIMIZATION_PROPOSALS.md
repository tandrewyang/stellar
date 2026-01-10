# 针对困难地图的优化方案设计

## 问题分析

以下6个地图在所有优化算法中胜率均为0.00%，说明现有优化方法对这些地图无效：

1. **dhls (调虎离山)**: 16 agents vs 11 enemies, 200步，Z vs T，有NydusCanal（虫洞）机制
2. **swct (上屋抽梯)**: 5 agents vs 11 enemies, 200步，P vs Z，有WarpPrism和ForceField机制
3. **tlhz (偷梁换柱)**: 4 agents vs 2 enemies, 300步，Z vs P，有建造机制（BuildHatchery, BuildChamber, TrainZerg）
4. **yqgz (欲擒故纵)**: 24 agents vs 8 enemies, 150步，Z vs T
5. **jctq (金蝉脱壳)**: 4 agents vs 9 enemies, 45步，Z vs P，有Burrow机制，至少存活1个
6. **fkwz (反客为主)**: 5 agents vs 7 enemies, 300步，P vs P，有建造机制

## 地图特点分类

### 1. 特殊机制地图（需要理解机制）
- **dhls**: 虫洞传送机制
- **swct**: 传送门和力场机制
- **jctq**: 埋地机制

### 2. 少对多地图（需要精确战术）
- **jctq**: 4 vs 9，45步（极短时间）
- **swct**: 5 vs 11

### 3. 建造类地图（需要长期规划）
- **tlhz**: 300步，需要建造和训练单位
- **fkwz**: 300步，需要建造和资源管理

### 4. 大规模地图（需要协调）
- **yqgz**: 24 agents vs 8 enemies

## 针对性优化方案

---

## 方案1: 机制感知强化学习 (Mechanism-Aware RL)

### 适用地图
- **dhls** (虫洞机制)
- **swct** (传送门和力场机制)
- **jctq** (埋地机制)

### 核心思想
这些地图有特殊的游戏机制，需要智能体理解并有效利用这些机制。

### 实现方案

#### 1.1 机制状态编码器
```python
class MechanismAwareEncoder(nn.Module):
    """
    专门编码特殊机制的状态
    - dhls: NydusCanal位置、可用性、连接状态
    - swct: WarpPrism位置、ForceField位置和持续时间
    - jctq: Burrow状态、敌人位置
    """
    def __init__(self, mechanism_type):
        # 机制特定的特征提取
        # 机制可用性、位置、状态等
        pass
```

#### 1.2 机制动作空间扩展
- 为特殊机制设计专门的动作空间
- 例如：虫洞使用、传送门操作、埋地/出土

#### 1.3 机制奖励塑形
```python
# 奖励机制使用
mechanism_reward = {
    'dhls': {
        'nydus_used': +0.1,  # 使用虫洞
        'nydus_successful': +0.5,  # 成功传送
        'tactical_positioning': +0.2  # 战术位置
    },
    'swct': {
        'forcefield_block': +0.3,  # 力场阻挡敌人
        'warp_prism_evacuation': +0.2  # 传送门撤离
    },
    'jctq': {
        'burrow_survival': +0.5,  # 埋地存活
        'burrow_escape': +0.3  # 埋地逃脱
    }
}
```

### 训练策略
1. **预训练阶段**: 在简化环境中学习机制使用
2. **课程学习**: 从简单机制使用到复杂战术组合
3. **模仿学习**: 从专家演示中学习机制使用模式

---

## 方案2: 分层战术规划 (Hierarchical Tactical Planning)

### 适用地图
- **jctq** (4 vs 9, 45步，极短时间)
- **swct** (5 vs 11，少对多)

### 核心思想
少对多需要精确的战术执行，需要高层战术规划和底层精确执行。

### 实现方案

#### 2.1 战术层（高层）
```python
class TacticalPlanner(nn.Module):
    """
    战术规划层：
    - 目标选择（优先攻击目标）
    - 队形选择（分散/集中/包围）
    - 时机选择（何时进攻/撤退）
    """
    def plan(self, state):
        # 输出战术目标
        return {
            'target_enemy': enemy_id,
            'formation': 'surround',
            'timing': 'now'
        }
```

#### 2.2 执行层（底层）
```python
class TacticalExecutor(nn.Module):
    """
    战术执行层：
    - 精确位置控制
    - 同步动作执行
    - 实时调整
    """
    def execute(self, tactical_plan, state):
        # 将战术计划转换为具体动作
        pass
```

#### 2.3 时间感知网络
```python
class TimeAwareNetwork(nn.Module):
    """
    对于jctq（45步），需要时间感知：
    - 剩余步数编码
    - 紧急度评估
    - 快速决策机制
    """
    def __init__(self):
        self.time_encoder = TimeEncoder()
        self.urgency_estimator = UrgencyEstimator()
```

### 训练策略
1. **两阶段训练**: 先训练战术规划，再训练执行
2. **时间压力训练**: 逐步减少可用时间
3. **失败案例分析**: 重点学习失败案例

---

## 方案3: 长期规划与资源管理 (Long-term Planning & Resource Management)

### 适用地图
- **tlhz** (300步，建造机制)
- **fkwz** (300步，建造和资源管理)

### 核心思想
这些地图需要长期规划，包括资源管理、建造顺序、单位训练等。

### 实现方案

#### 3.1 宏观策略网络
```python
class MacroStrategyNetwork(nn.Module):
    """
    宏观策略：
    - 资源分配（何时建造、何时训练）
    - 建造顺序（优先级）
    - 经济与军事平衡
    """
    def __init__(self):
        self.resource_allocator = ResourceAllocator()
        self.build_order_planner = BuildOrderPlanner()
        self.economy_military_balancer = EconomyMilitaryBalancer()
```

#### 3.2 多时间尺度学习
```python
class MultiTimescaleLearning:
    """
    多时间尺度：
    - 短期（1-10步）：单位控制
    - 中期（10-50步）：战术执行
    - 长期（50-300步）：战略规划
    """
    def forward(self, state, time_horizon):
        if time_horizon == 'short':
            return self.short_term_network(state)
        elif time_horizon == 'medium':
            return self.medium_term_network(state)
        else:
            return self.long_term_network(state)
```

#### 3.3 资源状态编码
```python
class ResourceStateEncoder(nn.Module):
    """
    资源状态编码：
    - 当前资源量
    - 资源获取速率
    - 资源消耗计划
    - 资源瓶颈预测
    """
    def encode(self, state):
        # 编码资源相关信息
        pass
```

### 训练策略
1. **分阶段训练**: 先学习资源管理，再学习战术
2. **长期奖励**: 设计长期奖励信号
3. **专家演示**: 从专家建造顺序中学习

---

## 方案4: 大规模协调优化 (Large-scale Coordination)

### 适用地图
- **yqgz** (24 agents vs 8 enemies)

### 核心思想
大规模单位需要有效的协调机制，避免单位之间的冲突和重复。

### 实现方案

#### 4.1 分层协调机制
```python
class HierarchicalCoordination(nn.Module):
    """
    分层协调：
    - 小队划分（将24个单位分成若干小队）
    - 小队间协调（避免重复攻击）
    - 小队内协调（队形、同步）
    """
    def __init__(self):
        self.squad_planner = SquadPlanner(n_squads=6)  # 24/4 = 6个小队
        self.inter_squad_coordinator = InterSquadCoordinator()
        self.intra_squad_coordinator = IntraSquadCoordinator()
```

#### 4.2 注意力机制优化
```python
class EfficientAttention(nn.Module):
    """
    高效注意力：
    - 局部注意力（只关注附近的单位）
    - 分层注意力（先小队内，再小队间）
    - 稀疏注意力（减少计算量）
    """
    def forward(self, x):
        # 使用局部注意力减少计算复杂度
        pass
```

#### 4.3 目标分配网络
```python
class TargetAssignmentNetwork(nn.Module):
    """
    目标分配：
    - 避免多个单位攻击同一目标
    - 优化目标分配（最大化伤害）
    - 动态调整（目标死亡后重新分配）
    """
    def assign_targets(self, agents, enemies):
        # 匈牙利算法或学习式分配
        pass
```

### 训练策略
1. **渐进式训练**: 从少单位到多单位
2. **协调奖励**: 奖励良好的协调行为
3. **冲突惩罚**: 惩罚单位间的冲突

---

## 方案5: 组合优化方案

### 适用地图
- **dhls** + **swct** + **jctq**: 机制感知 + 分层战术
- **tlhz** + **fkwz**: 长期规划 + 资源管理
- **yqgz**: 大规模协调

### 实现方案

#### 5.1 模块化架构
```python
class ModularOptimization(nn.Module):
    """
    模块化架构，可以组合不同模块：
    - 机制感知模块（用于特殊机制地图）
    - 战术规划模块（用于少对多地图）
    - 长期规划模块（用于建造类地图）
    - 协调模块（用于大规模地图）
    """
    def __init__(self, map_type):
        if map_type in ['dhls', 'swct', 'jctq']:
            self.add_module('mechanism_aware', MechanismAwareEncoder())
        if map_type in ['jctq', 'swct']:
            self.add_module('tactical_planner', TacticalPlanner())
        if map_type in ['tlhz', 'fkwz']:
            self.add_module('macro_strategy', MacroStrategyNetwork())
        if map_type == 'yqgz':
            self.add_module('coordination', HierarchicalCoordination())
```

#### 5.2 地图特定配置
```yaml
# 针对不同地图的配置
dhls:
  use_mechanism_aware: true
  mechanism_type: 'nydus_canal'
  use_tactical_planning: false
  use_long_term_planning: false

jctq:
  use_mechanism_aware: true
  mechanism_type: 'burrow'
  use_tactical_planning: true
  time_aware: true
  max_steps: 45

tlhz:
  use_long_term_planning: true
  use_resource_management: true
  max_steps: 300

yqgz:
  use_coordination: true
  n_squads: 6
  use_efficient_attention: true
```

---

## 实施建议

### 优先级排序
1. **高优先级**（最容易见效）:
   - **jctq**: 时间感知 + 战术规划（45步，需要快速决策）
   - **swct**: 机制感知 + 战术规划（5 vs 11，需要精确执行）

2. **中优先级**:
   - **dhls**: 机制感知（虫洞机制）
   - **yqgz**: 大规模协调（24个单位）

3. **低优先级**（最复杂）:
   - **tlhz**: 长期规划 + 资源管理（建造机制）
   - **fkwz**: 长期规划 + 资源管理（建造和资源）

### 实施步骤
1. **第一阶段**: 实现机制感知模块，测试在dhls、swct、jctq上的效果
2. **第二阶段**: 实现战术规划模块，测试在jctq、swct上的效果
3. **第三阶段**: 实现长期规划模块，测试在tlhz、fkwz上的效果
4. **第四阶段**: 实现大规模协调模块，测试在yqgz上的效果
5. **第五阶段**: 组合优化，测试所有地图

### 评估指标
- **胜率提升**: 主要指标
- **机制使用率**: 对于机制地图，机制使用频率
- **战术执行质量**: 对于战术地图，战术执行准确性
- **资源利用效率**: 对于建造地图，资源利用率
- **协调质量**: 对于大规模地图，单位协调度

---

## 技术细节

### 网络架构建议
1. **机制感知**: 使用图神经网络（GNN）编码机制状态
2. **战术规划**: 使用Transformer编码战术序列
3. **长期规划**: 使用LSTM/GRU编码时间序列
4. **大规模协调**: 使用局部注意力 + 全局协调

### 训练技巧
1. **课程学习**: 从简单到复杂
2. **模仿学习**: 从专家演示中学习
3. **多任务学习**: 在相关地图间共享知识
4. **强化学习**: 通过试错学习

### 超参数建议
- **学习率**: 机制感知和战术规划需要较小学习率（1e-4）
- **批次大小**: 大规模地图需要较小批次（32-64）
- **探索率**: 机制地图需要更高探索率（0.3-0.5）

---

## 预期效果

- **jctq**: 从0%提升到30-50%（时间感知 + 战术规划）
- **swct**: 从0%提升到20-40%（机制感知 + 战术规划）
- **dhls**: 从0%提升到15-30%（机制感知）
- **yqgz**: 从0%提升到10-25%（大规模协调）
- **tlhz**: 从0%提升到5-15%（长期规划，最困难）
- **fkwz**: 从0%提升到5-15%（长期规划，最困难）

---

## 总结

这6个困难地图需要针对性的优化方案，而不是通用的优化方法。关键是根据地图特点设计专门的模块：

1. **机制地图**: 需要理解并利用特殊机制
2. **少对多地图**: 需要精确的战术规划和执行
3. **建造地图**: 需要长期规划和资源管理
4. **大规模地图**: 需要有效的协调机制

通过模块化设计，可以针对不同地图组合不同的优化模块，实现最佳效果。


