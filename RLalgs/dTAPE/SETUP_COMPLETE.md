# ✅ 新地图配置完成清单

## 📋 PZYY (抛砖引玉) 和 LDTJ (李代桃僵) 地图配置

### ✅ 已完成的配置

#### 1. 地图文件 ✅
- [x] `/share/project/ytz/StarCraftII/Maps/Tactics_Maps/pzyy.SC2Map` (88 KB)
- [x] `/share/project/ytz/StarCraftII/Maps/Tactics_Maps/ldtj.SC2Map` (74 KB)

#### 2. 环境代码 ✅
- [x] `/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/star36env_pzyy.py`
  - 枪兵 (Marine) + 寡妇雷 (Widow Mine, 支持埋地/出地)
  - vs 小狗 (Zergling) + 监察者 (Overseer)
  - 特殊动作：Burrow/Unburrow
  
- [x] `/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/star36env_ldtj.py`
  - 异龙 (Mutalisk) + 孢子爬虫 (Spore Crawler, 不能移动)
  - vs 寡妇雷 (Widow Mine) + 攻城坦克 (Siege Tank)

#### 3. 环境注册 ✅
- [x] `/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src/envs/__init__.py`
  - 添加了 `SC2TacticsPZYYEnv` 和 `SC2TacticsLDTJEnv` 的 import
  - 添加了对应的环境注册逻辑

#### 4. 地图别名 ✅
- [x] `/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/maps/__init__.py`
  - 添加了 `"pzyy": "pzyy_te"` 和 `"ldtj": "ldtj_te"` 别名映射

#### 5. 地图参数注册 ✅
- [x] `/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/maps/sc2_tactics_maps.py`
  - 添加了 `pzyy_te` 地图配置 (11 agents, 25 enemies)
  - 添加了 `ldtj_te` 地图配置 (7 agents, 5 enemies)

#### 6. 代码依赖修复 ✅
- [x] `episode_buffer.py` - 已复制并修复缩进错误

#### 7. 训练脚本 ✅
- [x] `train_pzyy_gpu6_foreground.sh` - PZYY 前台训练
- [x] `train_ldtj_gpu7_foreground.sh` - LDTJ 前台训练
- [x] `train_pzyy_gpu6.sh` - PZYY 后台训练
- [x] `train_ldtj_gpu7.sh` - LDTJ 后台训练
- [x] `start_new_maps_training.sh` - 批量启动脚本

---

## 🚀 启动训练

### 方式 1：前台运行（推荐用于调试）

```bash
# PZYY (GPU 6)
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_pzyy_gpu6_foreground.sh

# LDTJ (GPU 7)
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_ldtj_gpu7_foreground.sh
```

### 方式 2：使用 tmux 同时运行

```bash
# PZYY
tmux new -s pzyy_train
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_pzyy_gpu6_foreground.sh
# 按 Ctrl+B 然后 D 分离

# LDTJ
tmux new -s ldtj_train
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./train_ldtj_gpu7_foreground.sh
# 按 Ctrl+B 然后 D 分离

# 重新连接
tmux attach -t pzyy_train
tmux attach -t ldtj_train
```

### 方式 3：后台运行

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
./start_new_maps_training.sh
```

---

## 📊 监控训练

### 查看运行状态
```bash
ps aux | grep main.py | grep -E 'pzyy|ldtj'
```

### 查看日志
```bash
# PZYY
tail -f /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/results/train_logs/pzyy_dtape/train_*.log

# LDTJ
tail -f /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/results/train_logs/ldtj_dtape/train_*.log
```

### 停止训练
```bash
# 停止 PZYY
pkill -f 'map_name=pzyy'

# 停止 LDTJ
pkill -f 'map_name=ldtj'
```

---

## 🎯 训练参数

- **算法**: dTAPE
- **训练步数**: 2,005,000
- **种子**: 42
- **GPU**: PZYY (GPU 6), LDTJ (GPU 7)
- **批量大小**: 1
- **保存间隔**: 每 500,000 步
- **TensorBoard**: 已启用

---

## ✅ 所有配置已完成！准备开始训练！🚀

