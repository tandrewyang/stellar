#!/bin/bash
# 测试已存在的地图（SDJX）是否能正常工作

source /share/project/miniconda3/etc/profile.d/conda.sh
conda activate py310_sc2

export SC2PATH="/share/project/ytz/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$(dirname "$0")"

echo "=========================================="
echo "测试已存在的 SDJX 地图"
echo "=========================================="

timeout 120 python3 src/main.py \
    --config=d_tape \
    --env-config=sc2te \
    with env_args.map_name=sdjx \
    seed=42 \
    t_max=50000 \
    batch_size_run=1 \
    test_interval=10000 \
    log_interval=5000 2>&1 | tee test_sdjx.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 124 ]; then
    echo "✅ 测试成功！SDJX 地图可以正常加载和运行"
    echo "这说明基础环境没问题，问题可能在新地图文件上"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成"
else
    echo "❌ 测试失败，退出码: $EXIT_CODE"
    echo "基础环境可能有问题"
fi
echo "=========================================="



