# HLSMAC 论文学习总结

## 论文信息
- **标题**: HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making
- **作者**: Xingxing Hong, Yungong Wang, Dexin Jin, Ye Yuan, Ximing Huang, Zijian Wu, Wenxin Li
- **链接**: https://arxiv.org/html/2509.12927v1

---

## 一、核心创新点

### 1. 设计理念
- **从微操到战略**: HLSMAC 强调高级战略决策，而非 SMAC 的微操（micromanagement）
- **融入人类智慧**: 基于中国古典兵法《三十六计》设计场景
- **12个精心设计的场景**: 每个场景对应一个具体的计策

### 2. 关键区别（vs SMAC）
- **更大的地图尺寸和更丰富的地形元素**
- **扩展的单位和建筑能力**
- **多样化的对手策略**
- **重新定义的游戏终止条件**

---

## 二、12个HLSMAC场景

根据论文，12个场景对应12个计策：

| 场景ID | 中文名称 | 英文名称 | 对应计策 |
|--------|---------|---------|---------|
| adcc | 暗度陈仓 | Advancing Secretly by an Unknown Path | 第8计 |
| dhls | 调虎离山 | Lure the Tiger Out of the Mountains | 第15计 |
| fkwz | 反客为主 | Turn from Guest into Host | 第30计 |
| gmzz | 关门捉贼 | Shut the Door to Catch the Thief | 第22计 |
| jctq | 金蝉脱壳 | Slipping Away by Casting Off a Cloak | 第21计 |
| jdsr | 借刀杀人 | Kill with a Borrowed Knife | 第3计 |
| sdjx | 声东击西 | Make a Feint to the East While Attacking in the West | 第6计 |
| swct | 上屋抽梯 | Pull Down the Ladder After the Ascent | 第28计 |
| tlhz | 偷梁换柱 | Steal the Beams and Pillars | 第25计 |
| wwjz | 围魏救赵 | Besiege Wei to Rescue Zhao | 第2计 |
| wzsy | 无中生有 | Create Something from Nothing | 第7计 |
| yqgz | 欲擒故纵 | In Order to Capture, One Must Let Loose | 第16计 |

---

## 三、评估框架

### 1. PyMARL框架集成
- HLSMAC环境实现
- 与PyMARL接口对接
- 支持现有的MARL算法

### 2. LLM-PySC2框架
- 扩展LLM智能体配置
- 与LLM智能体接口对接
- 评估LLM在战略决策中的表现

---

## 四、评估指标

### 1. 传统指标
- **胜率 (Win Rate)**: 基础性能指标

### 2. HLSMAC特定指标（论文创新）

#### 2.1 能力利用率 (Ability Utilization Frequency, AUF)
- **定义**: 评估智能体对关键能力的利用频率
- **意义**: 区分人类玩家和当前方法
- **发现**: 人类玩家有目的地利用关键能力，而当前方法缺乏这种能力

#### 2.2 单位存活率 (Unit Survival Rate, USR)
- **定义**: 评估单位在战斗中的存活情况
- **意义**: 反映战术执行的有效性

#### 2.3 关键目标伤害 (Critical Target Damage, CTD)
- **定义**: 对关键目标的伤害程度
- **相关性**: 与胜率相关性最高（R²分析）

#### 2.4 目标接近频率 (Target Proximity Frequency, TPF)
- **定义**: 智能体接近目标位置的频率
- **相关性**: 与胜率高度相关

#### 2.5 目标方向对齐 (Target Directional Alignment, TDA)
- **定义**: 行动方向与目标方向的对齐程度
- **相关性**: 与胜率中等相关

#### 2.6 推进效率 (Advancement Efficiency)
- **定义**: 战略推进的效率

---

## 五、实验结果（21个MARL算法）

### 1. 评估的算法列表
根据论文表格，评估了以下算法：

**经典算法**:
- IQL (Independent Q-Learning)
- COMA
- VDN
- QMIX
- QTRAN

**改进算法**:
- VMIX
- MAVEN
- CWQMIX (CW-QMIX)
- OWQMIX (OW-QMIX)
- DOP
- LICA
- Qatten
- QPLEX
- FOP
- RIIT
- RODE
- ROMA
- RESQ
- RESZ
- dTAPE
- sTAPE

### 2. 关键发现

#### 2.1 胜率表现
- **最佳算法**: 不同场景表现不同
- **FOP**: 在多个场景表现优秀
- **dTAPE**: 在部分场景表现良好
- **sTAPE**: 在某些场景达到1.0胜率

#### 2.2 单位存活率 (USR)
根据Table 31的数据：
- **FOP**: 在多个场景达到1.0
- **sTAPE**: 在多个场景表现优秀
- **dTAPE**: 在部分场景表现良好（0.96-1.0）

#### 2.3 指标相关性分析
根据Figure 13的R²分析：
1. **CTD (Critical Target Damage)**: 与胜率相关性最高
2. **TPF (Target Proximity Frequency)**: 高度相关
3. **USR (Unit Survival Rate)**: 高度相关
4. **TDA (Target Directional Alignment)**: 中等相关
5. **AUF (Ability Utilization Frequency)**: 相关性较弱，但对区分人类玩家有价值

---

## 六、LLM-PySC2框架结果

### 1. GPT-3.5的表现
- **理解能力**: 能够识别部分场景的战略意图
- **执行能力**: 行动选择效果不佳，难以成功完成目标
- **示例场景理解**:
  - gmzz: 正确识别"关门捉贼"策略，知道利用补给站阻挡
  - swct: 识别单位能力，知道使用力场阻挡
  - 其他场景: 理解有限

### 2. 人类玩家表现
- **胜率**: 在多个场景达到1.0
- **能力利用**: 有目的地利用关键能力
- **战略执行**: 更好地遵循计策思路

---

## 七、对作业的启示

### 1. 算法选择建议
根据论文结果，以下算法在HLSMAC上表现较好：
- **FOP**: 多个场景表现优秀
- **sTAPE**: 部分场景达到完美表现
- **dTAPE**: 稳定表现
- **RIIT**: 在部分场景表现良好

### 2. 优化方向
根据论文发现，可以从以下方面优化：

#### 2.1 提高关键目标伤害 (CTD)
- 改进目标选择策略
- 优化攻击优先级

#### 2.2 提高目标接近频率 (TPF)
- 改进路径规划
- 优化单位移动策略

#### 2.3 提高单位存活率 (USR)
- 改进战术执行
- 优化单位控制

#### 2.4 能力利用优化
- 虽然AUF与胜率相关性较弱，但能区分人类玩家
- 可以设计奖励机制鼓励正确使用关键能力

### 3. 评估重点
- **主要指标**: 胜率（作业要求）
- **辅助指标**: CTD, TPF, USR（可用于分析）
- **创新点**: 可以关注能力利用和战略遵循度

---

## 八、论文中的关键场景示例

### Example 1: Besiege Wei to Rescue Zhao (wwjz)
- **策略**: 围魏救赵
- **要点**: 通过攻击次要目标来解救主要目标

### Example 2: Lure Your Enemy onto the Roof, Then Take Away the Ladder (swct)
- **策略**: 上屋抽梯
- **要点**: 利用地形和单位能力（如力场）困住敌人

### Example 3: Kill with a Borrowed Sword (jdsr)
- **策略**: 借刀杀人
- **要点**: 利用敌人或环境来攻击目标

---

## 九、实验设置参考

### 1. 训练参数
- **训练步数**: 通常2,000,000+步
- **测试回合数**: 32个episode
- **随机种子**: 多个种子取平均

### 2. 评估方法
- **贪心评估**: test_greedy=True
- **多次运行**: 多个随机种子
- **指标计算**: 胜率、USR、CTD等

---

## 十、对当前项目的指导

### 1. 算法选择
- **已选择**: dTAPE（论文中表现稳定）
- **优化方向**: 参考FOP和sTAPE的成功经验

### 2. 评估重点
- **主要**: 胜率提升（作业要求）
- **辅助**: 可以分析CTD、TPF等指标
- **创新**: 可以关注能力利用的改进

### 3. 实验设计
- **12个地图**: 全部训练和测试
- **多次运行**: 使用不同随机种子
- **对比实验**: 与原始baseline对比

---

## 参考文献

[1] Hong, Xingxing et al. "HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making." ArXiv abs/2509.12927 (2025).

[2] Samvelyan, Mikayel et al. "The StarCraft Multi-Agent Challenge." ArXiv abs/1902.04043 (2019).

[3] Vinyals, Oriol et al. "StarCraft II: A New Challenge for Reinforcement Learning." ArXiv abs/1708.04782 (2017).

