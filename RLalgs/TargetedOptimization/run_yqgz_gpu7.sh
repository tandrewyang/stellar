#!/bin/bash
# 启动yqgz地图训练 (GPU7) - 前台运行，实时显示
# yqgz特点: 24 vs 8, 150步
# 优化: 大规模协调

cd "$(dirname "$0")"
echo "启动 yqgz 训练 (GPU7) - 前台运行"
echo "按 Ctrl+C 可停止训练"
echo ""
bash train_single_map.sh yqgz 7 42

