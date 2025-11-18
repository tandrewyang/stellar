# HLSMAC 训练日志分析指南

## 正常日志特征

### 1. StarCraft II 启动阶段 ✅

**正常日志序列**:
```
[INFO] Launching SC2: /share/project/ytz/StarCraftII/Versions/Base75689/SC2_x64
[INFO] Connecting to: ws://127.0.0.1:41607/sc2api, attempt: 0
Starting up...
Startup Phase 1 complete
Startup Phase 2 complete
Startup Phase 3 complete. Ready for commands.
```

**说明**: 
- SC2 进程正常启动
- WebSocket 连接建立成功
- 启动阶段按顺序完成

---

### 2. 游戏初始化阶段 ✅

**正常日志**:
```
Game has started.
Using default stable ids, none found at: /share/project/ytz/StarCraftII/stableid.json
Successfully loaded stable ids: GameData\stableid.json
Sending ResponseJoinGame
```

**说明**:
- ✅ 游戏已启动
- ✅ 稳定ID加载成功（从GameData目录）
- ✅ 加入游戏请求已发送

**注意**: 
- `stableid.json` 未找到是正常的，会使用默认的稳定ID
- 如果需要在根目录放置 `stableid.json`，可以从GitHub下载

---

### 3. 训练进行阶段 ✅

**正常日志**:
```
target_mean: -0.0046	td_error_abs: 0.5631
test_battle_won_mean: 0.0000	test_dead_allies_mean: 1.1250
```

**说明**:
- ✅ 训练正在进行
- ✅ TD误差在正常范围
- ✅ 测试指标在记录

---

### 4. 警告信息（可忽略）⚠️

**PyTorch 警告**:
```
UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated
```

**说明**:
- ⚠️ 这是PyTorch版本兼容性警告
- ⚠️ 不影响训练，但未来版本可能会报错
- ⚠️ 可以忽略，或后续修复代码

**TensorFlow 警告**:
```
oneDNN custom operations are on. You may see slightly different numerical results
```

**说明**:
- ⚠️ TensorFlow的数值计算警告
- ⚠️ 不影响训练
- ⚠️ 可以忽略

---

## 异常日志特征

### 1. 连接失败 ❌

**异常日志**:
```
[ERROR] Failed to connect to SC2
[ERROR] Connection timeout
```

**解决方法**:
- 检查SC2PATH是否正确
- 检查SC2进程是否正常启动
- 检查端口是否被占用

---

### 2. 地图加载失败 ❌

**异常日志**:
```
NoMapError: Map doesn't exist: adcc
FileNotFoundError: Map file not found
```

**解决方法**:
- 检查地图文件是否在 `$SC2PATH/Maps/` 目录
- 检查地图名称是否正确
- 确认已应用地图别名映射修复

---

### 3. 训练错误 ❌

**异常日志**:
```
RuntimeError: CUDA out of memory
ValueError: Invalid action
```

**解决方法**:
- CUDA内存不足：减小batch_size
- 无效动作：检查动作空间配置

---

## 当前日志状态分析

根据你提供的日志，**状态正常** ✅：

1. ✅ **SC2启动成功**: 
   - 版本: B75689 (SC2.4.10)
   - 连接: ws://127.0.0.1:41607/sc2api
   - 启动阶段: 全部完成

2. ✅ **游戏初始化成功**:
   - Game has started
   - 稳定ID加载成功
   - ResponseJoinGame已发送

3. ✅ **训练正在进行**:
   - 有训练指标输出
   - TD误差在正常范围
   - 测试指标在记录

4. ⚠️ **警告信息**:
   - PyTorch索引警告（可忽略）
   - TensorFlow oneDNN警告（可忽略）

---

## 日志监控建议

### 关键指标监控

1. **训练指标**:
   - `target_mean`: 目标值均值（应该逐渐收敛）
   - `td_error_abs`: TD误差绝对值（应该逐渐减小）
   - `test_battle_won_mean`: 测试胜率（应该逐渐提高）

2. **游戏状态**:
   - `Game has started`: 确认游戏正常启动
   - `test_dead_allies_mean`: 单位存活情况

3. **错误监控**:
   - 任何 `[ERROR]` 或 `Exception` 都需要注意
   - 连接失败需要立即处理

---

## 日志文件位置

### Sacred日志
```
RLalgs/dTAPE/results/sacred/<map_name>/<algorithm>/<run_id>/
├── cout.txt          # 标准输出（包含训练日志）
├── config.json       # 配置信息
├── run.json          # 运行信息
└── info.json         # 详细指标数据
```

### TensorBoard日志
```
RLalgs/dTAPE/results/<map_name>_<algorithm>_seed<seed>/tboard/
```

---

## 快速检查命令

```bash
# 检查最新训练日志
tail -f RLalgs/dTAPE/results/sacred/adcc/ow_qmix_env=4_adam_td_lambda/*/cout.txt

# 检查是否有错误
grep -i "error\|failed\|exception" RLalgs/dTAPE/results/sacred/*/*/cout.txt

# 检查训练进度
grep "test_battle_won_mean" RLalgs/dTAPE/results/sacred/*/*/cout.txt | tail -5

# 检查游戏启动
grep "Game has started" RLalgs/dTAPE/results/sacred/*/*/cout.txt
```

---

## 总结

**你的日志完全正常** ✅

- StarCraft II 成功启动
- 游戏正常初始化
- 训练正在进行
- 只有一些可忽略的警告

可以继续训练，无需担心！

