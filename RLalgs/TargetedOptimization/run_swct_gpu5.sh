#!/bin/bash
# 启动swct地图训练 (GPU5) - 前台运行，实时显示
# swct特点: 5 vs 11, 200步, WarpPrism和ForceField机制
# 优化: 机制感知 + 战术规划

cd "$(dirname "$0")"
echo "启动 swct 训练 (GPU5) - 前台运行"
echo "按 Ctrl+C 可停止训练"
echo ""
bash train_single_map.sh swct 5 42

