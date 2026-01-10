#!/bin/bash
# TargetedOptimization - 训练单个HLSMAC地图（针对性优化）

if [ $# -lt 1 ]; then
    echo "用法: $0 <map_name> [gpu_id] [seed]"
    echo "示例: $0 jctq 4 42"
    echo "支持的地图: jctq, swct, dhls, yqgz, tlhz, fkwz"
    exit 1
fi

MAP_NAME=$1
GPU_ID=${2:-0}
SEED=${3:-42}

# 根据地图选择配置文件
case $MAP_NAME in
    jctq)
        CONFIG_NAME="targeted_qmix_jctq"
        ;;
    swct)
        CONFIG_NAME="targeted_qmix_swct"
        ;;
    dhls)
        CONFIG_NAME="targeted_qmix_dhls"
        ;;
    yqgz)
        CONFIG_NAME="targeted_qmix_yqgz"
        ;;
    tlhz)
        CONFIG_NAME="targeted_qmix_tlhz"
        ;;
    fkwz)
        CONFIG_NAME="targeted_qmix_fkwz"
        ;;
    *)
        echo "错误: 不支持的地图 $MAP_NAME"
        echo "支持的地图: jctq, swct, dhls, yqgz, tlhz, fkwz"
        exit 1
        ;;
esac

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
echo "TargetedOptimization 训练"
echo "=========================================="
echo "地图: $MAP_NAME"
echo "配置文件: $CONFIG_NAME"
echo "GPU: $GPU_ID"
echo "种子: $SEED"
echo "SC2PATH: $SC2PATH"
echo "=========================================="

# 创建日志目录
LOG_DIR="../results/train_logs/${MAP_NAME}_${CONFIG_NAME}"
mkdir -p "$LOG_DIR"

# 运行训练（前台运行，实时显示输出，同时保存日志）
# 使用python3 -u确保无缓冲输出（实时显示）
# 使用tee同时输出到终端和文件
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u src/main.py \
    --config=$CONFIG_NAME \
    --env-config=sc2te \
    with env_args.map_name=$MAP_NAME \
    seed=$SEED \
    t_max=2005000 \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000 \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "训练完成！日志保存在: $LOG_DIR/train.log"

