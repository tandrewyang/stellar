# 四个地图优化训练启动命令

## 快速启动（一键启动所有训练）

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
bash start_all_optimized_training.sh
```

这将自动在后台启动所有四个地图的训练：
- **DHLS** (调虎离山): GPU 4
- **YQGZ** (欲擒故纵): GPU 5
- **TLHZ** (偷梁换柱): GPU 6
- **FKWZ** (反客为主): GPU 7

---

## 单独启动命令

### 1. DHLS (调虎离山) - GPU 4

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
CUDA_VISIBLE_DEVICES=4 python3 -u src/main.py \
    --config=targeted_qmix_dhls \
    --env-config=sc2te \
    with env_args.map_name=dhls \
    seed=42 \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000
```

或者使用训练脚本：
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
bash train_single_map.sh dhls 4 42
```

---

### 2. YQGZ (欲擒故纵) - GPU 5

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
CUDA_VISIBLE_DEVICES=5 python3 -u src/main.py \
    --config=targeted_qmix_yqgz \
    --env-config=sc2te \
    with env_args.map_name=yqgz \
    seed=42 \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000
```

或者使用训练脚本：
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
bash train_single_map.sh yqgz 5 42
```

---

### 3. TLHZ (偷梁换柱) - GPU 6

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
CUDA_VISIBLE_DEVICES=6 python3 -u src/main.py \
    --config=targeted_qmix_tlhz \
    --env-config=sc2te \
    with env_args.map_name=tlhz \
    seed=42 \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000
```

或者使用训练脚本：
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
bash train_single_map.sh tlhz 6 42
```

---

### 4. FKWZ (反客为主) - GPU 7

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
CUDA_VISIBLE_DEVICES=7 python3 -u src/main.py \
    --config=targeted_qmix_fkwz \
    --env-config=sc2te \
    with env_args.map_name=fkwz \
    seed=42 \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000
```

或者使用训练脚本：
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/TargetedOptimization
bash train_single_map.sh fkwz 7 42
```

---

## 后台运行（推荐）

### 使用 nohup 后台运行单个训练：

```bash
# DHLS
nohup bash train_single_map.sh dhls 4 42 > logs/dhls_gpu4.log 2>&1 &

# YQGZ
nohup bash train_single_map.sh yqgz 5 42 > logs/yqgz_gpu5.log 2>&1 &

# TLHZ
nohup bash train_single_map.sh tlhz 6 42 > logs/tlhz_gpu6.log 2>&1 &

# FKWZ
nohup bash train_single_map.sh fkwz 7 42 > logs/fkwz_gpu7.log 2>&1 &
```

### 查看后台训练状态：

```bash
# 查看进程
ps aux | grep train_single_map

# 查看GPU使用情况
nvidia-smi

# 查看日志
tail -f logs/dhls_gpu4.log
tail -f logs/yqgz_gpu5.log
tail -f logs/tlhz_gpu6.log
tail -f logs/fkwz_gpu7.log
```

---

## 优化说明

本次优化针对所有四个地图进行了大幅增强：

1. **获胜检测奖励**: Episode结束时获胜给予10.0奖励
2. **生存奖励**: 从0.2提升到0.5
3. **战术奖励**: 各战术动作奖励提升100-200%
4. **伤害奖励**: 对敌人造成伤害的奖励放大1.5-2.0倍
5. **阶段奖励**: 前期/中期/后期奖励增强150-300%

配置文件权重也已大幅提升：
- 奖励塑形权重: 0.3 → 0.5
- 奖励塑形衰减: 0.9995 → 0.9998
- 各专项奖励权重提升50-100%

---

## 训练监控

训练日志保存在：
- `results/train_logs/{map_name}_{config_name}/train.log`

Sacred实验结果保存在：
- `results/sacred/{map_name}/{config_name}/`

TensorBoard日志：
```bash
tensorboard --logdir=results/tb_logs
```

