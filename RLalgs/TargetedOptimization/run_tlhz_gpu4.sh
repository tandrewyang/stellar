#!/bin/bash
# 启动tlhz地图训练 (GPU4) - 前台运行，实时显示
# tlhz特点: 4 vs 2, 300步, 有建造机制（BuildHatchery, BuildChamber, TrainZerg）

cd "$(dirname "$0")"

echo "启动 tlhz 训练 (GPU4) - 前台运行"
echo "地图: tlhz (偷梁换柱)"
echo "优化策略: 长期规划 + 建造奖励 + 训练奖励"
echo ""

bash train_single_map.sh tlhz 4 42

