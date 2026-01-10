#!/bin/bash
# 启动fkwz地图训练 (GPU5) - 前台运行，实时显示
# fkwz特点: 5 vs 7, 300步, 建造机制

echo "启动 fkwz 训练 (GPU5) - 前台运行"

# 调用通用训练脚本
bash train_single_map.sh fkwz 5 42

