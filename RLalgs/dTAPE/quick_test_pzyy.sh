#!/bin/bash
# 快速测试 PZYY 地图（只启动环境，不训练）

source /share/project/miniconda3/etc/profile.d/conda.sh
conda activate py310_sc2

export SC2PATH="/share/project/ytz/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$(dirname "$0")"

echo "=========================================="
echo "快速测试 - PZYY 地图"
echo "=========================================="

# 只运行很短时间来测试地图加载
timeout 90 python3 src/main.py \
    --config=d_tape \
    --env-config=sc2te \
    with env_args.map_name=pzyy \
    seed=42 \
    t_max=5000 \
    batch_size_run=1 \
    test_interval=1000 2>&1 | head -100

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "检查输出上方的错误信息："
echo "  - 如果看到 'ConnectionError' 或 'SC2 crashed' → 地图文件问题"
echo "  - 如果看到 NaN 或 IndexError → 环境代码问题" 
echo "  - 如果看到训练开始 → 成功！"
echo "=========================================="



