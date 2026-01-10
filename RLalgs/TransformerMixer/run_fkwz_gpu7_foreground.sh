#!/bin/bash
# TransformerMixer - 训练fkwz地图，GPU 7，前台输出日志

MAP_NAME="fkwz"
GPU_ID=7
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

# 设置protobuf环境变量（解决版本兼容性问题）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=========================================="
echo "TransformerMixer 训练 - fkwz"
echo "=========================================="
echo "地图: $MAP_NAME"
echo "GPU: $GPU_ID"
echo "种子: $SEED"
echo "SC2PATH: $SC2PATH"
echo "=========================================="

# 创建日志目录
LOG_DIR="results/train_logs/${MAP_NAME}_transformer_qmix"
mkdir -p "$LOG_DIR"

# 前台运行，同时输出到终端和日志文件
CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
    --config=transformer_qmix \
    --env-config=sc2te \
    with env_args.map_name=$MAP_NAME \
    seed=$SEED \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000 \
    2>&1 | tee "$LOG_DIR/train.log"

echo "训练完成！日志保存在: $LOG_DIR/train.log"


