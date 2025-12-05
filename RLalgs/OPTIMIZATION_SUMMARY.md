# 算法优化版本汇总

本文档汇总了所有实现的算法优化版本及其特点。

## 优化版本列表

### 1. TransformerMixer
- **路径**: `RLalgs/TransformerMixer/`
- **优化方向**: 模型结构改进 - Transformer架构的Mixer网络
- **核心改进**: 
  - 使用多头自注意力机制
  - Transformer编码器层
  - 智能体间注意力
- **配置文件**: `src/config/algs/transformer_qmix.yaml`
- **训练脚本**: `train_single_map.sh`

### 2. HierarchicalArchitecture
- **路径**: `RLalgs/HierarchicalArchitecture/`
- **优化方向**: 模型结构改进 - 分层架构
- **核心改进**:
  - 高层策略网络（目标选择、战术决策）
  - 底层执行网络（具体动作）
  - 分层决策机制
- **配置文件**: `src/config/algs/hierarchical_qmix.yaml`
- **训练脚本**: `train_single_map.sh`

### 3. EnhancedStateRepresentation
- **路径**: `RLalgs/EnhancedStateRepresentation/`
- **优化方向**: 特征工程优化
- **核心改进**:
  - 增强的状态编码器
  - 多层特征提取
  - 注意力机制
  - 层归一化
- **配置文件**: `src/config/algs/enhanced_qmix.yaml`
- **训练脚本**: `train_single_map.sh`

### 4. CurriculumLearning./start_missing_tasks.sh

- **路径**: `RLalgs/CurriculumLearning/`
- **优化方向**: 训练策略改进 - 课程学习
- **核心改进**:
  - 线性难度调度
  - 自适应难度调度
  - 逐步增加任务难度
- **配置文件**: `src/config/algs/curriculum_qmix.yaml`
- **训练脚本**: `train_single_map.sh`

### 5. RewardShaping
- **路径**: `RLalgs/RewardShaping/`
- **优化方向**: 训练策略改进 - 奖励塑形
- **核心改进**:
  - 基于潜在函数的奖励塑形
  - 密集奖励
  - 好奇心驱动
- **配置文件**: `src/config/algs/reward_shaping_qmix.yaml`
- **训练脚本**: `train_single_map.sh`

## 优化方向分类

### 模型结构改进
1. **TransformerMixer**: Transformer架构的Mixer网络
2. **HierarchicalArchitecture**: 分层架构（高层+底层）

### 特征工程优化
3. **EnhancedStateRepresentation**: 改进状态表示

### 训练策略改进
4. **CurriculumLearning**: 课程学习
5. **RewardShaping**: 奖励塑形

## 使用方法

### 训练单个地图
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
bash RLalgs/<优化版本>/train_single_map.sh <map_name> <gpu_id> <seed>
```

### 示例
```bash
# TransformerMixer (GPU 1)
bash RLalgs/TransformerMixer/train_single_map.sh adcc 1 42

# HierarchicalArchitecture (GPU 2)
bash RLalgs/HierarchicalArchitecture/train_single_map.sh adcc 2 42

# EnhancedStateRepresentation (GPU 3)
bash RLalgs/EnhancedStateRepresentation/train_single_map.sh adcc 3 42

# CurriculumLearning (GPU 4)
bash RLalgs/CurriculumLearning/train_single_map.sh adcc 4 42

# RewardShaping (GPU 5)
bash RLalgs/RewardShaping/train_single_map.sh adcc 5 42
```

## 对比实验建议

### 1. 与Baseline对比
所有优化版本都应该与原始dTAPE算法在相同条件下进行对比。

### 2. 消融实验
- 测试各个组件的贡献
- 测试不同参数设置的影响

### 3. 组合实验
- 可以尝试组合不同的优化方法
- 例如：TransformerMixer + RewardShaping

### 4. 不同地图测试
- 在不同HLSMAC地图上测试效果
- 分析哪些优化适合哪些地图

## 性能指标

建议记录以下指标进行对比：
- **胜率（Win Rate）**: 最重要的指标
- **训练速度**: 收敛到特定胜率所需的步数
- **训练稳定性**: TD误差、奖励方差等
- **计算开销**: 训练时间、内存使用等

## 注意事项

1. **训练过程一致**: 所有优化版本使用相同的训练流程
2. **超参数调优**: 每个优化版本可能需要不同的超参数
3. **资源管理**: 注意不同优化版本的计算资源需求
4. **可复现性**: 使用相同的随机种子确保可复现

## 扩展方向

### 可以继续实现的优化
1. **规则+学习融合**: 结合规则系统和学习系统
2. **搜索+学习**: 结合搜索算法（如MCTS）和学习
3. **大模型决策**: 结合大语言模型进行决策
4. **多任务学习**: 在不同HLSMAC地图间共享知识
5. **元学习**: 快速适应新地图

## 文档结构

每个优化版本都包含：
- `OPTIMIZATION_DESCRIPTION.md`: 详细的优化说明
- `src/config/algs/<config>.yaml`: 配置文件
- `train_single_map.sh`: 训练脚本
- 相应的代码实现

## 联系与反馈

如有问题或建议，请参考各优化版本的详细文档。

