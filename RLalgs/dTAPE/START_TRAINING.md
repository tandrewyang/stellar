# 🚀 开始训练 - PZYY & LDTJ 地图

## ✅ 所有配置已完成

- ✅ 地图文件已放置在正确位置
- ✅ 环境代码已创建
- ✅ 地图参数已注册
- ✅ Python 环境已配置 (py310_sc2)
- ✅ 训练脚本已准备就绪

---

## 🎯 立即开始训练

### 方式 1：前台运行（推荐首次测试）

```bash
# 训练 PZYY (GPU 6)
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_pzyy_gpu6_foreground.sh
```

或

```bash
# 训练 LDTJ (GPU 7)
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_ldtj_gpu7_foreground.sh
```

**特点：**
- 实时查看训练输出
- 按 Ctrl+C 即可停止
- 适合调试和验证

---

### 方式 2：使用 tmux 后台运行（推荐长时间训练）

```bash
# 启动 PZYY 训练
tmux new -s pzyy_train
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_pzyy_gpu6_foreground.sh
# 按 Ctrl+B 然后按 D 分离

# 启动 LDTJ 训练
tmux new -s ldtj_train
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_ldtj_gpu7_foreground.sh
# 按 Ctrl+B 然后按 D 分离
```

**重新连接查看训练：**
```bash
tmux attach -t pzyy_train  # 查看 PZYY 训练
tmux attach -t ldtj_train  # 查看 LDTJ 训练
```

**列出所有会话：**
```bash
tmux ls
```

---

### 方式 3：使用 nohup 后台运行

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_pzyy_gpu6.sh
./train_ldtj_gpu7.sh
```

**查看日志：**
```bash
tail -f results/train_logs/pzyy_dtape/train_*.log
tail -f results/train_logs/ldtj_dtape/train_*.log
```

---

## 📊 监控训练

### 查看运行状态
```bash
ps aux | grep main.py | grep -E 'pzyy|ldtj'
```

### 查看 GPU 使用情况
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控
```

### 停止训练
```bash
# 停止 PZYY
pkill -f 'map_name=pzyy'

# 停止 LDTJ
pkill -f 'map_name=ldtj'

# 或者在 tmux 中按 Ctrl+C
```

---

## ⚙️ 训练配置

### PZYY (抛砖引玉)
- **单位**: 10 枪兵 + 1 寡妇雷 vs 22 小狗 + 1 监察者
- **特殊机制**: 寡妇雷埋地/出地
- **GPU**: 6
- **训练步数**: 2,005,000

### LDTJ (李代桃僵)
- **单位**: 5 异龙 + 2 孢子爬虫 vs 4 寡妇雷 + 1 攻城坦克
- **特殊机制**: 孢子爬虫不能移动
- **GPU**: 7
- **训练步数**: 2,005,000

---

## 🔍 故障排查

### 如果遇到 "Connection already closed" 错误：
1. 检查 SC2 是否正常安装：`ls -lh $SC2PATH`
2. 检查地图文件：`ls -lh $SC2PATH/Maps/Tactics_Maps/`
3. 尝试减少 `batch_size_run` 或调整其他参数

### 如果遇到 Python 模块缺失：
```bash
conda activate py310_sc2
conda list | grep torch
```

### 查看完整错误日志：
```bash
tail -100 results/train_logs/pzyy_dtape/train_*.log
```

---

## 📈 预期输出

正常训练时，你应该看到类似以下的输出：

```
========================================
dTAPE 训练 - PZYY (抛砖引玉) - 前台运行
========================================
地图: pzyy
GPU: 6
种子: 42
SC2PATH: /share/project/ytz/StarCraftII
========================================

[INFO] Initializing environment...
[INFO] Map loaded successfully
[INFO] Training started...
[DEBUG] Episode 1, Reward: ...
...
```

---

## 🎉 开始训练！

选择一个方式，立即开始训练：

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_pzyy_gpu6_foreground.sh
```

**祝训练顺利！** 🚀

