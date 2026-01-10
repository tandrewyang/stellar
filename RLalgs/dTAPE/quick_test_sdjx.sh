#!/bin/bash
# 快速测试 - 使用已知可工作的 SDJX 地图

export SC2PATH="/share/project/ytz/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$(dirname "$0")"

echo "快速测试 - 使用 SDJX 地图验证环境..."
echo ""

CUDA_VISIBLE_DEVICES=6 timeout 60 python3 src/main.py \
    --config=d_tape \
    --env-config=sc2te \
    with env_args.map_name=sdjx \
    seed=42 \
    t_max=10000 \
    batch_size_run=1 \
    test_interval=5000 \
    log_interval=1000

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo ""
    echo "✅ 测试成功！环境可以正常启动（超时退出是正常的）"
    echo "现在可以尝试运行新地图了"
elif [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 测试完成！"
else
    echo ""
    echo "❌ 测试失败！退出码: $EXIT_CODE"
    echo "请检查环境配置"
fi

