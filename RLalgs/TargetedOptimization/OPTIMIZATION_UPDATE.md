# 针对性优化更新说明

## 更新内容

### jctq (金蝉脱壳) - 重新设计
**问题**: 胜率0%，无提升
**策略**: "金蝉脱壳" - 伪装、转移注意力、逃脱

**优化调整**:
1. 增加通信维度 (3→4) - 需要更好的协调来分散注意力
2. 更大的网络维度 (mixing_embed_dim: 64→128, hypernet_embed: 128→256)
3. 更高的奖励塑形权重 (0.2→0.3) - 关键策略需要强化
4. 更频繁的目标网络更新 (200→100) - 快速学习
5. 更高的td_lambda (0.6→0.8) - 长期规划，存活策略
6. 更快的探索衰减 (50000→30000) - 45步需要快速收敛
7. 更高的初始探索 (0.995→0.99) - 需要快速学习

### swct (上屋抽梯) - 进一步优化
**当前状态**: 胜率9.38%，有提升但不够
**策略**: "上屋抽梯" - 切断退路、迫使前进、包围战术

**优化调整**:
1. 增加通信维度 (3→4) - 需要更好的协调来包围敌人
2. 更大的网络维度 (mixing_embed_dim: 64→128, hypernet_embed: 128→256)
3. 更高的奖励塑形权重 (0.15→0.25) - 战术策略需要强化
4. 更高的td_lambda (0.6→0.7) - 长期战术规划
5. 更慢的衰减 (默认→0.999) - 策略需要持续

## 启动命令

### jctq (GPU4) - 重新训练
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
CUDA_VISIBLE_DEVICES=4 python3 -u src/main.py \
    --config=targeted_qmix_jctq \
    --env-config=sc2te \
    with env_args.map_name=jctq seed=42 t_max=2005000 \
    batch_size_run=1 use_tensorboard=True save_model=True \
    save_model_interval=500000 2>&1 | tee ../results/train_logs/jctq_targeted_qmix_jctq_golden_cicada/train.log

### swct (GPU5) - 继续优化
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
CUDA_VISIBLE_DEVICES=5 python3 -u src/main.py \
    --config=targeted_qmix_swct \
    --env-config=sc2te \
    with env_args.map_name=swct seed=42 t_max=2005000 \
    batch_size_run=1 use_tensorboard=True save_model=True \
    save_model_interval=500000 2>&1 | tee ../results/train_logs/swct_targeted_qmix_swct_remove_ladder/train.log

