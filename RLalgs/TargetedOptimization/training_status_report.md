# TargetedOptimization 四个地图训练状态报告

生成时间: 2025-12-14

## 一、训练状态概览

### 1.1 总体情况

| 地图 | 实验名称 | 运行编号 | 训练进度 | 当前Episode | 测试次数 | 最新胜率 | 最高胜率 |
|------|---------|---------|---------|------------|---------|---------|---------|
| TLHZ | targeted_qmix_tlhz_long_term_planning | 6 | 41,100 / 2,005,000 (2.0%) | 136 | 4 | 0.00% | 0.00% |
| DHLS | targeted_qmix_dhls_env=4_adam_td_lambda | 11 | 40,473 / 2,005,000 (2.0%) | 304 | 4 | 0.00% | 0.00% |
| FKWZ | targeted_qmix_fkwz_active_offensive | 3 | 235 / 2,005,000 (0.0%) | N/A | 0 | N/A | N/A |
| YQGZ | targeted_qmix_yqgz_env=4_adam_td_lambda | 8 | 30,300 / 2,005,000 (1.5%) | 201 | 3 | 0.00% | 0.00% |

### 1.2 关键发现

1. **训练进度**: 所有训练都处于早期阶段（0-2%），距离完成还有很长的路要走
2. **胜率表现**: 所有地图的胜率都是0%，这是训练早期的正常现象
3. **测试频率**: 测试次数较少（0-4次），说明训练刚开始
4. **FKWZ异常**: FKWZ的训练进度极低（235步），可能存在问题

## 二、详细训练数据

### 2.1 TLHZ (调虎离山) - Long Term Planning

**优化策略**: 长期规划型奖励塑形
- **Reward Shaping Weight**: 0.6
- **Reward Shaping Decay**: 0.9999
- **Damage Reward Factor**: 2.5
- **Survival Reward Weight**: 3.5

**训练数据**:
- 训练步数: 41,100 / 2,005,000 (2.0%)
- 当前Episode: 136
- 测试次数: 4
- 最新胜率: 0.00%
- 最新Reward: -5.6924

**最近测试结果**:
- t_env=300: Win Rate=0.00%
- t_env=10,500: Win Rate=0.00%
- t_env=20,700: Win Rate=0.00%
- t_env=40,800: Win Rate=0.00%

**分析**: 
- 训练正常进行，但胜率仍为0
- Reward为负值，说明智能体还在学习基础行为
- 需要更多训练步数才能看到效果

### 2.2 DHLS (打草惊蛇) - Environment 4 Adam TD Lambda

**优化策略**: 环境4配置 + Adam优化器 + TD Lambda
- **Reward Shaping Weight**: 0.6
- **Reward Shaping Decay**: 0.9998
- **Damage Reward Factor**: 2.0
- **Division Reward Weight**: 4.0
- **Survival Reward Weight**: 3.5
- **学习率**: 0.0012

**训练数据**:
- 训练步数: 40,473 / 2,005,000 (2.0%)
- 当前Episode: 304
- 测试次数: 4
- 最新胜率: 0.00%
- 最新Reward: 0.0000

**最近测试结果**:
- t_env=68: Win Rate=0.00%
- t_env=10,182: Win Rate=0.00%
- t_env=20,334: Win Rate=0.00%
- t_env=40,289: Win Rate=0.00%

**分析**:
- 训练正常，Episode数相对较高（304）
- Reward为0，可能需要调整奖励函数
- Division Reward Weight较高（4.0），强调分割战术

### 2.3 FKWZ (反客为主) - Active Offensive

**优化策略**: 主动进攻型奖励塑形
- **Reward Shaping Weight**: 0.6
- **Reward Shaping Decay**: 0.9999
- **Damage Reward Factor**: 2.2
- **Warp Prism Reward Weight**: 5.0
- **Warpgate Train Reward Weight**: 6.0

**训练数据**:
- 训练步数: 235 / 2,005,000 (0.0%)
- 当前Episode: N/A
- 测试次数: 0
- 最新胜率: N/A
- 最新Reward: N/A

**分析**:
- ⚠️ **异常**: 训练进度极低，可能训练已停止或出现问题
- 需要检查训练进程是否还在运行
- 可能需要重启训练

### 2.4 YQGZ (以逸待劳) - Environment 4 Adam TD Lambda

**优化策略**: 环境4配置 + Adam优化器 + TD Lambda
- **Reward Shaping Weight**: 0.6
- **Reward Shaping Decay**: 0.9998
- **Damage Reward Factor**: 2.2
- **Survival Reward Weight**: 3.5

**训练数据**:
- 训练步数: 30,300 / 2,005,000 (1.5%)
- 当前Episode: 201
- 测试次数: 3
- 最新胜率: 0.00%
- 最新Reward: 0.0000

**最近测试结果**:
- t_env=150: Win Rate=0.00%
- t_env=10,200: Win Rate=0.00%
- t_env=20,250: Win Rate=0.00%

**分析**:
- 训练正常进行，进度略慢于其他地图
- Reward为0，可能需要调整奖励权重

## 三、Reward统计

### 3.1 Reward趋势

| 地图 | 最新Reward | 最高Reward | Reward趋势 |
|------|-----------|-----------|-----------|
| TLHZ | -5.6924 | N/A | 负值，需要改进 |
| DHLS | 0.0000 | N/A | 零值，可能需要调整 |
| FKWZ | N/A | N/A | 无数据 |
| YQGZ | 0.0000 | N/A | 零值，可能需要调整 |

### 3.2 Reward分析

1. **TLHZ**: Reward为负值，说明智能体还在学习基础行为，可能需要更多探索奖励
2. **DHLS/YQGZ**: Reward为0，可能是奖励函数设计问题，或者训练步数太少
3. **FKWZ**: 无数据，训练可能未正常进行

## 四、优化建议

### 4.1 立即行动项

#### 1. 检查FKWZ训练状态
```bash
# 检查FKWZ训练进程
ps aux | grep fkwz
# 检查训练日志
tail -f results/sacred/fkwz/targeted_qmix_fkwz_active_offensive/*/cout.txt
```

**建议**: 
- 如果训练已停止，需要重启
- 检查是否有错误信息
- 确认GPU资源是否充足

#### 2. 调整Reward函数

**TLHZ**:
- 当前Reward为负值，建议增加探索奖励
- 考虑增加生存奖励权重（当前3.5可能不够）
- 建议: `survival_reward_weight: 4.0-5.0`

**DHLS/YQGZ**:
- Reward为0可能表示奖励信号太弱
- 建议增加damage_reward_factor
- 建议: `damage_reward_factor: 2.5-3.0`

### 4.2 中期优化建议

#### 1. 调整Reward Shaping权重

当前所有地图都使用`reward_shaping_weight: 0.6`，建议根据训练进度动态调整：

- **早期（0-10%）**: 0.6-0.8（强引导）
- **中期（10-50%）**: 0.4-0.6（逐步减少）
- **后期（50%+）**: 0.2-0.4（依赖原始奖励）

#### 2. 优化测试频率

当前测试频率较低，建议：
- 每10,000步测试一次（当前约每10,000-20,000步）
- 增加早期测试频率，便于监控训练效果

#### 3. 学习率调整

**DHLS**使用0.0012，其他使用0.001，建议：
- 如果训练不稳定，降低学习率到0.0008-0.001
- 如果收敛太慢，可以尝试0.0012-0.0015

### 4.3 长期优化建议

#### 1. 地图特定优化

**TLHZ (调虎离山)**:
- 重点优化长期规划能力
- 增加对战略目标的奖励
- 考虑使用curriculum learning逐步增加难度

**DHLS (打草惊蛇)**:
- 强调分割和包围战术
- Division Reward Weight已经较高（4.0），可以保持
- 考虑增加对敌人分散的奖励

**FKWZ (反客为主)**:
- 强调主动进攻和资源控制
- Warpgate Train Reward Weight已经很高（6.0），可以保持
- 需要确保训练正常进行

**YQGZ (以逸待劳)**:
- 强调防御和资源积累
- 增加对资源保护的奖励
- 考虑增加对敌人消耗的奖励

#### 2. 训练策略优化

1. **Curriculum Learning**: 
   - 从简单场景开始，逐步增加难度
   - 可以设置多个训练阶段

2. **Early Stopping**:
   - 如果胜率长期为0，考虑调整策略
   - 设置胜率阈值，低于阈值时调整参数

3. **Multi-objective Optimization**:
   - 同时优化胜率、资源效率、战术多样性
   - 使用加权和或Pareto优化

#### 3. 监控和调试

1. **实时监控**:
   - 使用TensorBoard监控训练曲线
   - 设置告警机制，胜率异常时通知

2. **定期检查**:
   - 每24小时检查一次训练状态
   - 记录关键指标（胜率、Reward、Loss等）

3. **A/B测试**:
   - 对比不同配置的效果
   - 保留最佳配置

## 五、预期时间线

### 5.1 训练完成时间估算

假设当前训练速度保持不变：

| 地图 | 当前进度 | 预计完成时间 |
|------|---------|------------|
| TLHZ | 2.0% | 约50倍当前时间 |
| DHLS | 2.0% | 约50倍当前时间 |
| FKWZ | 0.0% | 需要重启 |
| YQGZ | 1.5% | 约67倍当前时间 |

**注意**: 实际时间可能因GPU资源、训练稳定性等因素而变化

### 5.2 关键里程碑

- **10%进度**: 应该能看到初步的胜率提升（预计1-2周）
- **25%进度**: 胜率应该有明显改善（预计1个月）
- **50%进度**: 应该接近或达到可接受的胜率（预计2个月）
- **100%进度**: 完成训练，评估最终性能（预计4个月）

## 六、总结

### 6.1 当前状态

1. **训练正常**: TLHZ、DHLS、YQGZ训练正常进行，但都处于早期阶段
2. **胜率为0**: 这是训练早期的正常现象，需要更多训练步数
3. **FKWZ异常**: 需要立即检查并可能重启训练

### 6.2 下一步行动

1. **立即**: 检查并修复FKWZ训练
2. **短期**: 调整Reward函数，增加探索奖励
3. **中期**: 优化Reward Shaping权重和学习率
4. **长期**: 实施Curriculum Learning和Multi-objective Optimization

### 6.3 风险提示

1. **训练时间**: 预计需要数月才能完成训练
2. **资源消耗**: 需要持续占用GPU资源
3. **胜率风险**: 如果长期胜率为0，可能需要重新设计奖励函数

---

**报告生成时间**: 2025-12-14
**下次更新建议**: 训练进度达到10%时

