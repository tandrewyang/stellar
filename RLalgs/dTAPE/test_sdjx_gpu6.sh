#!/bin/bash
# 临时测试脚本 - 使用已存在的 SDJX 地图验证环境

MAP_NAME="sdjx"
GPU_ID=6
SEED=42

export SC2PATH="/share/project/ytz/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$(dirname "$0")"

echo "=========================================="
echo "测试 dTAPE 环境 - 使用 SDJX 地图"
echo "=========================================="
echo "地图: $MAP_NAME"
echo "GPU: $GPU_ID"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
    --config=d_tape \
    --env-config=sc2te \
    with env_args.map_name=$MAP_NAME \
    seed=$SEED \
    t_max=50000 \
    batch_size_run=1 \
    test_interval=5000 \
    log_interval=1000

