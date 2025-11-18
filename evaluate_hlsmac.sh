#!/bin/bash
# HLSMAC模型评测脚本

if [ $# -lt 2 ]; then
    echo "用法: $0 <map_name> <checkpoint_path> [gpu_id]"
    echo "示例: $0 adcc ../../results/adcc_d_tape/models/episode_2000000.pt 0"
    exit 1
fi

MAP_NAME=$1
CHECKPOINT_PATH=$2
GPU_ID=${3:-0}
ALG_CONFIG=${ALG_CONFIG:-"d_tape"}
TEST_NEPISODE=${TEST_NEPISODE:-32}

# 进入算法目录
cd "$(dirname "$0")/RLalgs/dTAPE"

# 检查文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 模型文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

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
fi

echo "=========================================="
echo "评测模型"
echo "=========================================="
echo "地图: $MAP_NAME"
echo "模型: $CHECKPOINT_PATH"
echo "测试回合数: $TEST_NEPISODE"
echo "=========================================="

# 运行评测
CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
    --config=$ALG_CONFIG \
    --env-config=sc2te \
    with env_args.map_name=$MAP_NAME \
    evaluate=True \
    checkpoint_path="$CHECKPOINT_PATH" \
    test_nepisode=$TEST_NEPISODE \
    test_greedy=True \
    save_replay=True \
    env_args.replay_dir="$SC2PATH/Replays" \
    env_args.replay_prefix="${MAP_NAME}_eval"

echo ""
echo "评测完成！回放文件保存在: $SC2PATH/Replays/"

