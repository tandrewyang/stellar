#!/bin/bash
# dTAPE - 训练 PZYY (抛砖引玉) 地图

MAP_NAME="pzyy"
GPU_ID=6
SEED=42

# 设置SC2PATH（如果未设置）
if [ -z "$SC2PATH" ]; then
    DEFAULT_SC2PATH="/share/project/ytz/StarCraftII"
    if [ -d "$DEFAULT_SC2PATH" ]; then
        export SC2PATH="$DEFAULT_SC2PATH"
        echo "已自动设置 SC2PATH=$SC2PATH"
    else
        echo "错误: SC2PATH环境变量未设置，且默认路径不存在: $DEFAULT_SC2PATH"
        echo "请设置: export SC2PATH=/path/to/StarCraftII"
        exit 1
    fi
else
    echo "使用 SC2PATH=$SC2PATH"
fi

# 进入算法目录
cd "$(dirname "$0")"

# 激活 conda 环境
source /share/project/miniconda3/etc/profile.d/conda.sh
conda activate py310_sc2

# 设置protobuf环境变量（解决版本兼容性问题）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=========================================="
echo "dTAPE 训练 - PZYY (抛砖引玉)"
echo "=========================================="
echo "地图: $MAP_NAME"
echo "GPU: $GPU_ID"
echo "种子: $SEED"
echo "SC2PATH: $SC2PATH"
echo "=========================================="

# 创建日志目录
LOG_DIR="results/train_logs/${MAP_NAME}_dtape"
mkdir -p "$LOG_DIR"

# 运行训练（后台运行）
nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
    --config=d_tape \
    --env-config=sc2te \
    with env_args.map_name=$MAP_NAME \
    seed=$SEED \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000 \
    > "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# 获取进程ID
PID=$!
echo "训练已在后台启动！"
echo "进程 ID: $PID"
echo "日志目录: $LOG_DIR"
echo "查看日志: tail -f $LOG_DIR/train_*.log"
echo "停止训练: kill $PID"

