# HLSMAC 快速开始指南

## 5分钟快速开始

### 0. 快速设置环境变量（推荐）

如果只是想快速设置环境变量而不修改配置文件：

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
source set_env.sh  # 临时设置（仅当前会话）
```

### 1. 环境配置（首次运行，永久设置）

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC

# 运行环境配置脚本（自动检测shell类型）
bash setup_env.sh

# 如果使用Zsh（推荐）
bash setup_env_zsh.sh
source ~/.zshrc

# 如果使用Bash
source ~/.bashrc
```

**注意**: 如果看到 `.bashrc` 相关的错误（如 `shopt: command not found`），这是因为你使用的是 zsh 而不是 bash。请使用 `setup_env_zsh.sh` 脚本。

### 2. 训练单个地图（测试）

```bash
# 训练"暗度陈仓"地图（约需数小时）
bash train_single_map.sh adcc 0 42
bash train_single_map.sh jdsr 0 42
# 查看训练日志
tail -f results/adcc_d_tape_seed42/train.log
```

### 3. 评测模型

```bash
# 评测训练好的模型
bash evaluate_hlsmac.sh adcc results/adcc_d_tape_seed42/models/episode_2000000.pt 0
```

## 完整训练流程

### 步骤1: 环境检查

```bash
# 检查SC2PATH
echo $SC2PATH
# 应该输出: /share/project/ytz/StarCraftII

# 检查地图文件
ls $SC2PATH/Maps/*.SC2Map | wc -l
# 应该输出: 12 (HLSMAC的12张地图)
```

### 步骤2: 训练所有地图

```bash
# 使用原始dTAPE算法训练
bash train_hlsmac.sh

# 或使用优化版本
ALG_CONFIG=d_tape_improved bash train_hlsmac.sh
```

### 步骤3: 评测所有模型

```bash
# 评测所有地图的最新模型
bash evaluate_all_maps.sh
```

### 步骤4: 查看结果

```bash
# 查看TensorBoard
tensorboard --logdir=results/train_logs

# 查看评测结果
cat results/evaluation_*/adcc_eval.log
```

## 常用命令

### 训练相关

```bash
# 训练单个地图（指定GPU和种子）
bash train_single_map.sh <map_name> <gpu_id> <seed>

# 训练所有地图（后台运行）
nohup bash train_hlsmac.sh > train_all.log 2>&1 &

# 查看训练进度
tail -f results/train_logs/*/train.log
```

### 评测相关

```bash
# 评测单个模型
bash evaluate_hlsmac.sh <map_name> <checkpoint_path> <gpu_id>

# 评测所有模型
bash evaluate_all_maps.sh

# 评测指定步数的模型
LOAD_STEP=1000000 bash evaluate_all_maps.sh
```

### 模型管理

```bash
# 列出所有训练好的模型
find results -name "*.pt" -type f

# 查看模型大小
du -h results/*/models/*.pt

# 删除旧模型（保留最新）
find results -name "*.pt" -mtime +7 -delete
```

## 参数调优建议

### 训练时间不足时

```bash
# 减少训练步数
T_MAX=1000000 bash train_single_map.sh adcc 0 42
```

### GPU内存不足时

修改配置文件 `RLalgs/dTAPE/src/config/algs/d_tape_improved.yaml`:
```yaml
batch_size: 64  # 从128减小
batch_size_run: 1
```

### 加快训练速度

```bash
# 使用多个GPU并行训练不同地图
CUDA_VISIBLE_DEVICES=0 bash train_single_map.sh adcc 0 42 &
CUDA_VISIBLE_DEVICES=1 bash train_single_map.sh dhls 1 42 &
CUDA_VISIBLE_DEVICES=2 bash train_single_map.sh fkwz 2 42 &
```

## 故障排除

### 问题0: Sacred Git 信息错误（已修复）

**错误信息**: `ValueError: Reference at 'refs/heads/master' does not exist`

**状态**: ✅ 已修复 - 已在代码中添加 Git 错误处理

如果再次遇到此问题，代码会自动捕获并忽略 Git 错误，不影响训练。

### 问题1: 地图参数未找到（已修复）

**错误信息**: `ValueError: Map parameters for 'adcc' not found in map_param_registry`

**状态**: ✅ 已修复 - 已添加地图名称别名映射

地图注册表中使用 `adcc_te` 格式，但训练脚本使用 `adcc`。已自动处理别名映射。

### 问题2: Protobuf 版本错误（已修复）

**错误信息**: `TypeError: Descriptors cannot be created directly`

**状态**: ✅ 已修复 - protobuf 已降级到 3.20.3

如果再次遇到此问题：
```bash
# 降级 protobuf
pip install protobuf==3.20.3 --force-reinstall

# 或使用修复脚本
bash fix_protobuf.sh

# 验证修复
python3 -c "from s2clientprotocol import sc2api_pb2; print('✓ 成功')"
```

**注意**: 如果同时使用 TensorFlow，可能会有版本冲突警告，但不影响 StarCraft II 使用。

### 问题1: 找不到地图文件

```bash
# 手动复制地图
cp Tactics_Maps/HLSMAC_Maps/*.SC2Map $SC2PATH/Maps/
```

### 问题2: 模块导入错误

```bash
# 重新设置PYTHONPATH
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac:$PYTHONPATH"
```

### 问题3: SC2启动失败

```bash
# 检查SC2PATH
echo $SC2PATH
ls $SC2PATH/Versions/

# 检查权限
chmod +x $SC2PATH/Versions/*/SC2_x64
```

## 下一步

1. 查看完整文档: `README.md`
2. 了解算法优化: 查看 `RLalgs/dTAPE/src/config/algs/d_tape_improved.yaml`
3. 分析训练结果: 使用TensorBoard查看训练曲线
4. 提交作业: 按照作业要求整理模型、报告和代码

