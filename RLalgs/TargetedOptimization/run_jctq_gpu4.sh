#!/bin/bash
# 启动jctq地图训练 (GPU4) - 前台运行，实时显示
# jctq特点: 4 vs 9, 45步, Burrow机制
# 优化: 时间感知 + 战术规划 + 机制感知

cd "$(dirname "$0")"
echo "启动 jctq 训练 (GPU4) - 前台运行"
echo "按 Ctrl+C 可停止训练"
echo ""
bash train_single_map.sh jctq 4 42

