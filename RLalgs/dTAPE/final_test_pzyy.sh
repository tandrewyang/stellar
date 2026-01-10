#!/bin/bash
# 最终测试 - PZYY 地图完整训练流程（限时2分钟）

source /share/project/miniconda3/etc/profile.d/conda.sh
conda activate py310_sc2

export SC2PATH="/share/project/ytz/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$(dirname "$0")"

echo "=========================================="
echo "最终测试 - PZYY 完整训练流程（120秒超时）"
echo "=========================================="
echo "如果120秒内没有崩溃，说明配置成功！"
echo ""

timeout 120 python3 src/main.py \
    --config=d_tape \
    --env-config=sc2te \
    with env_args.map_name=pzyy \
    seed=42 \
    t_max=100000 \
    batch_size_run=1 2>&1 | tee test_pzyy_full.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 124 ]; then
    echo "✅ 成功！训练运行了120秒没有崩溃"
    echo "PZYY 地图配置正确，可以开始正式训练了！"
    echo ""
    echo "开始训练命令："
    echo "  ./train_pzyy_gpu6_foreground.sh"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "训练正常结束"
else
    echo "❌ 出错了，退出码: $EXIT_CODE"
    echo "请检查上方的错误信息"
fi
echo "=========================================="



