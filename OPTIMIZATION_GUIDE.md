# HLSMAC 算法优化指南

基于论文 https://arxiv.org/html/2509.12927v1 的优化建议

## 一、论文关键发现

### 1. 评估指标与胜率的相关性（R²分析）
根据论文Figure 13，指标与胜率的相关性排序：

1. **CTD (Critical Target Damage)** - 相关性最高 ⭐⭐⭐⭐⭐
2. **TPF (Target Proximity Frequency)** - 高度相关 ⭐⭐⭐⭐
3. **USR (Unit Survival Rate)** - 高度相关 ⭐⭐⭐⭐
4. **TDA (Target Directional Alignment)** - 中等相关 ⭐⭐⭐
5. **AUF (Ability Utilization Frequency)** - 相关性较弱，但能区分人类玩家 ⭐⭐

### 2. 表现优秀的算法
根据论文Table 31（USR数据）和胜率数据：

**综合表现优秀**:
- **FOP**: 多个场景达到1.0胜率
- **sTAPE**: 部分场景完美表现
- **dTAPE**: 稳定表现（0.96-1.0）
- **RIIT**: 部分场景表现良好

**场景特定表现**:
- **adcc**: FOP, sTAPE表现优秀
- **dhls**: 多个算法达到1.0
- **fkwz**: FOP, Qatten表现优秀
- **gmzz**: 多个算法表现良好
- **jctq**: FOP, sTAPE表现优秀
- **jdsr**: sTAPE, DOP表现优秀
- **sdjx**: sTAPE, OWQMIX表现优秀
- **swct**: 多个算法达到1.0
- **tlhz**: FOP, sTAPE表现优秀
- **wwjz**: 多个算法表现良好
- **wzsy**: Qatten, dTAPE表现优秀
- **yqgz**: 多个算法达到1.0

---

## 二、优化方向建议

### 方向1: 提高关键目标伤害 (CTD) ⭐⭐⭐⭐⭐

**原理**: CTD与胜率相关性最高

**实现方法**:
1. **改进目标选择策略**
   - 识别关键目标（如敌方建筑、关键单位）
   - 优先攻击对胜利影响最大的目标
   - 实现目标优先级排序

2. **优化攻击分配**
   - 集中火力攻击关键目标
   - 避免分散攻击
   - 实现协同攻击机制

3. **奖励塑形**
   - 增加对关键目标伤害的奖励
   - 设计分层奖励：关键目标 > 普通单位 > 建筑

**代码修改位置**:
- `src/modules/agents/`: 改进智能体目标选择
- `src/learners/`: 添加CTD相关奖励
- `src/components/`: 实现目标优先级计算

---

### 方向2: 提高目标接近频率 (TPF) ⭐⭐⭐⭐

**原理**: TPF与胜率高度相关

**实现方法**:
1. **改进路径规划**
   - 优化单位移动到关键位置的路径
   - 考虑地形和障碍物
   - 实现智能路径选择

2. **位置奖励机制**
   - 奖励接近关键位置的单位
   - 设计位置相关的奖励函数
   - 实现位置价值估计

3. **协同移动**
   - 多个单位协同移动到目标位置
   - 实现编队移动
   - 优化移动时机

**代码修改位置**:
- `src/controllers/`: 改进移动控制逻辑
- `src/modules/`: 添加位置价值估计模块
- `src/learners/`: 添加位置相关奖励

---

### 方向3: 提高单位存活率 (USR) ⭐⭐⭐⭐

**原理**: USR与胜率高度相关

**实现方法**:
1. **改进战术执行**
   - 优化单位控制，减少不必要的损失
   - 实现撤退机制
   - 优化单位站位

2. **防御策略**
   - 识别危险情况
   - 实现单位保护机制
   - 优化单位分组

3. **奖励机制**
   - 奖励单位存活
   - 惩罚单位损失
   - 实现存活率相关的奖励

**代码修改位置**:
- `src/controllers/`: 改进单位控制
- `src/modules/agents/`: 添加防御策略
- `src/learners/`: 添加存活率奖励

---

### 方向4: 能力利用优化 ⭐⭐

**原理**: 虽然AUF与胜率相关性较弱，但能区分人类玩家

**实现方法**:
1. **识别关键能力**
   - 分析每个场景的关键能力
   - 实现能力重要性评估
   - 设计能力使用策略

2. **能力使用奖励**
   - 奖励正确使用关键能力
   - 惩罚错误使用能力
   - 实现能力使用频率监控

3. **能力组合优化**
   - 优化能力使用时机
   - 实现能力组合策略
   - 设计能力协同机制

**代码修改位置**:
- `src/modules/`: 添加能力管理模块
- `src/controllers/`: 改进能力使用逻辑
- `src/learners/`: 添加能力使用奖励

---

## 三、具体优化方案

### 方案1: 基于CTD的奖励塑形

```python
# 在learner中添加CTD奖励
def compute_ctd_reward(self, batch, rewards):
    """
    计算关键目标伤害奖励
    """
    # 识别关键目标（根据地图类型）
    critical_targets = self.get_critical_targets(batch)
    
    # 计算对关键目标的伤害
    ctd = self.compute_critical_damage(batch, critical_targets)
    
    # 添加CTD奖励（权重可调）
    ctd_reward = ctd * self.args.ctd_reward_weight
    
    return rewards + ctd_reward
```

### 方案2: 基于TPF的位置奖励

```python
# 添加位置价值估计
class PositionValueEstimator(nn.Module):
    """
    估计位置价值，用于TPF优化
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 位置价值网络
        
    def forward(self, positions, map_state):
        # 估计每个位置的价值
        return position_values
```

### 方案3: 基于USR的存活奖励

```python
# 在reward计算中添加存活奖励
def compute_survival_reward(self, batch):
    """
    计算单位存活奖励
    """
    # 计算单位存活率
    survival_rate = self.compute_survival_rate(batch)
    
    # 存活奖励（鼓励保持单位存活）
    survival_reward = survival_rate * self.args.survival_reward_weight
    
    return survival_reward
```

---

## 四、实验设计建议

### 1. 基线对比
- **原始dTAPE**: 作为baseline
- **优化版本**: 应用上述优化
- **对比指标**: 胜率、CTD、TPF、USR

### 2. 消融实验
- **实验1**: 仅添加CTD奖励
- **实验2**: 仅添加TPF奖励
- **实验3**: 仅添加USR奖励
- **实验4**: 组合所有优化

### 3. 场景分析
- **分析每个场景**: 哪些优化最有效
- **场景特定优化**: 针对不同场景的优化策略
- **失败案例分析**: 分析失败原因，针对性改进

---

## 五、参考论文中的成功案例

### 1. FOP的成功经验
- **特点**: 在多个场景达到1.0胜率
- **可能原因**: 
  - 有效的策略分解
  - 良好的协同机制
  - 优化的奖励设计

### 2. sTAPE的成功经验
- **特点**: 部分场景完美表现
- **可能原因**:
  - 有效的通信机制
  - 良好的策略学习
  - 优化的动作选择

### 3. dTAPE的稳定表现
- **特点**: 稳定在0.96-1.0
- **优势**: 
  - 确定性策略
  - 稳定的性能
  - 良好的泛化能力

---

## 六、实施步骤

### 阶段1: 分析当前性能
1. 运行原始dTAPE在12个地图上
2. 记录胜率、CTD、TPF、USR等指标
3. 分析失败案例

### 阶段2: 实施优化
1. 选择1-2个优化方向（建议从CTD开始）
2. 实现优化代码
3. 在单个地图上测试

### 阶段3: 全面评估
1. 在所有12个地图上训练
2. 对比原始baseline
3. 分析改进效果

### 阶段4: 消融实验
1. 测试各个优化的独立效果
2. 优化组合策略
3. 最终评估

---

## 七、注意事项

1. **不要过度优化**: 避免过拟合特定场景
2. **保持通用性**: 优化应该适用于多个场景
3. **平衡指标**: 不要只关注胜率，也要考虑其他指标
4. **可解释性**: 记录优化思路，便于报告撰写

---

## 八、论文链接

- **完整论文**: https://arxiv.org/html/2509.12927v1
- **关键章节**: Section 5 (Evaluation Metrics and Results)

