#!/bin/bash
# DOP训练单个HLSMAC地图的脚本

if [ $# -lt 1 ]; then
    echo "用法: $0 <map_name> [gpu_id] [seed]"
    echo "地图列表: adcc, dhls, fkwz, gmzz, jctq, jdsr, sdjx, swct, tlhz, wwjz, wzsy, yqgz"
    exit 1
fi

MAP_NAME=$1
GPU_ID=${2:-0}
SEED=${3:-42}
ALG_CONFIG=${ALG_CONFIG:-"dop"}
T_MAX=${T_MAX:-2005000}

# 进入算法目录
cd "$(dirname "$0")"

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

echo "训练地图: $MAP_NAME"
echo "GPU: $GPU_ID"
echo "种子: $SEED"
echo "算法: $ALG_CONFIG"

# 创建结果目录
RESULT_DIR="../results/${MAP_NAME}_${ALG_CONFIG}_seed${SEED}"
mkdir -p "$RESULT_DIR"

# 运行训练
CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
    --config=$ALG_CONFIG \
    --env-config=sc2te \
    with env_args.map_name=$MAP_NAME \
    seed=$SEED \
    t_max=$T_MAX \
    batch_size_run=1 \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=500000 \
    local_results_path="$RESULT_DIR"

